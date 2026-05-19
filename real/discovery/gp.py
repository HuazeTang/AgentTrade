"""Genetic Programming engine for factor discovery.

Evolves a population of expression trees, selecting for predictive power
(IC), robustness (IC IR, stability), and parsimony (complexity penalty).

Typical usage:
    engine = GPEngine(population_size=200, max_generations=50)
    best = engine.evolve(data, forward_returns, existing_factors)
    # best is a list of (tree, factor_cls, fitness, metrics) tuples
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from discovery.compiler import compile_expr, compile_and_validate
from discovery.expr import (
    Expr, VarExpr, ConstExpr, UnaryOp, BinaryOp,
    RollingOp, CrossSectionalOp, GroupedCrossSectionalOp, TimeSeriesOp,
    random_expr, collect_all_nodes,
    ROLLING_OPS, ROLLING_WINDOWS, UNARY_OPS, BINARY_OPS,
    CS_OPS, CS_GROUP_OPS, GROUP_FIELDS, TS_OPS, TS_PERIODS, TERMINAL_FIELDS,
)
from discovery.operators import operator_registry
from discovery.validate import FactorValidator

logger = logging.getLogger(__name__)


class Individual(NamedTuple):
    tree: Expr
    factor_name: str
    factor_cls: type | None  # compiled Factor class
    fitness: float
    ic_mean: float
    ic_ir: float
    hit_rate: float
    auto_corr: float
    complexity: int
    depth: int
    generation: int = 0  # which generation this individual was created in


@dataclass
class GenerationStats:
    """Per-generation population statistics for evolution tracking."""
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    worst_fitness: float
    std_fitness: float
    best_ic: float
    mean_ic: float
    best_ir: float
    mean_ir: float
    best_depth: int
    mean_depth: float
    best_nodes: int
    mean_nodes: float
    valid_count: int
    total_count: int
    hall_of_fame_size: int
    stall_count: int
    elapsed_seconds: float


@dataclass
class GPConfig:
    """Configuration for the GP engine."""
    population_size: int = 200
    max_generations: int = 50
    tournament_size: int = 7
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    elite_count: int = 5
    max_depth: int = 6
    max_complexity: int = 30
    parsimony_penalty: float = 0.001  # per node
    terminal_prob: float = 0.3  # probability of picking terminal during mutation

    # Terminal set: data columns usable as leaf nodes.
    # Defaults to TERMINAL_FIELDS (raw OHLCV + derived). Extend with factor names
    # to let GP combine existing factors with raw fields.
    terminals: list[str] | None = None  # None = use TERMINAL_FIELDS from expr.py

    # Mutation weights
    subtree_mutation_weight: float = 0.30
    window_mutation_weight: float = 0.20
    operator_mutation_weight: float = 0.20
    constant_mutation_weight: float = 0.15
    unary_mutation_weight: float = 0.15

    # Fitness weights
    ic_mean_weight: float = 0.4
    ic_ir_weight: float = 0.2
    stability_weight: float = 0.2  # min IC across periods
    hit_rate_weight: float = 0.2

    # Early stopping
    early_stop_generations: int = 10  # stop if no improvement in N gens
    min_fitness_improvement: float = 0.001


class GPEngine:
    """Genetic Programming engine for factor evolution.

    Usage:
        engine = GPEngine(config=GPConfig(population_size=200))
        results = engine.evolve(data, forward_returns, existing_factors)
        for individual in results[:5]:
            print(f"{individual.factor_name}: fitness={individual.fitness:.4f}")
    """

    def __init__(self, config: GPConfig | None = None):
        self.config = config or GPConfig()
        self._validator = FactorValidator()
        self._generation = 0
        self._best_fitness: float = -np.inf
        self._stall_count = 0
        self._hall_of_fame: list[Individual] = []
        self._history: list[GenerationStats] = []

    # ── Main evolution loop ────────────────────────────────────────────────

    def evolve(
        self,
        data: pd.DataFrame,
        forward_returns: pd.Series,
        existing_factors: pd.DataFrame | None = None,
        callback: Callable[[int, list[Individual]], None] | None = None,
    ) -> list[Individual]:
        """Evolve a population of factor expressions.

        Args:
            data: Multi-indexed (trade_date, symbol) OHLCV DataFrame.
            forward_returns: Same-index forward return Series.
            existing_factors: DataFrame of existing factor values for novelty check.
            callback: Optional function called after each generation with
                      (generation, population).

        Returns:
            List of Individuals sorted by fitness (best first).
        """
        cfg = self.config
        self._generation = 0
        self._best_fitness = -np.inf
        self._stall_count = 0
        self._hall_of_fame = []

        # Initialize population
        logger.info("Initializing population of %d ...", cfg.population_size)
        population = self._initialize_population()

        # Evaluate initial population
        logger.info("Evaluating initial population ...")
        population = self._evaluate_population(
            population, data, forward_returns, existing_factors
        )
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        self._update_hall_of_fame(population[:cfg.elite_count])
        self._best_fitness = population[0].fitness
        self._record_generation_stats(population, 0.0)

        if callback:
            callback(0, population)

        # Evolution loop
        for gen in range(1, cfg.max_generations + 1):
            self._generation = gen
            t_start = time.time()

            # Elitism: keep best N valid individuals (skip NaN fitness)
            valid_pop = [ind for ind in population if not np.isnan(ind.fitness) and ind.fitness > -900]
            if len(valid_pop) >= 1:
                elites = valid_pop[:cfg.elite_count]
            else:
                elites = population[:cfg.elite_count]

            # Generate offspring
            offspring = []
            while len(offspring) < cfg.population_size - cfg.elite_count:
                # Selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                child_tree1 = parent1.tree.clone()
                child_tree2 = parent2.tree.clone()

                # Crossover
                if random.random() < cfg.crossover_prob:
                    child_tree1, child_tree2 = self._crossover(child_tree1, child_tree2)

                # Mutation
                if random.random() < cfg.mutation_prob:
                    child_tree1 = self._mutate(child_tree1)
                if random.random() < cfg.mutation_prob:
                    child_tree2 = self._mutate(child_tree2)

                # Enforce depth/complexity limits
                if child_tree1.depth() <= cfg.max_depth and child_tree1.node_count() <= cfg.max_complexity:
                    offspring.append(child_tree1)
                if len(offspring) < cfg.population_size - cfg.elite_count:
                    if child_tree2.depth() <= cfg.max_depth and child_tree2.node_count() <= cfg.max_complexity:
                        offspring.append(child_tree2)

            # Evaluate offspring
            new_pop = self._evaluate_population(
                offspring, data, forward_returns, existing_factors
            )

            # New population = elites + evaluated offspring
            population = elites + new_pop
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            self._update_hall_of_fame(population[:cfg.elite_count])

            # Progress
            best = population[0]
            elapsed = time.time() - t_start
            logger.info(
                "Gen %3d: best_fitness=%.4f IC=%.4f IR=%.3f depth=%d nodes=%d (%s) [%.1fs]",
                gen, best.fitness, best.ic_mean, best.ic_ir,
                best.depth, best.complexity, best.factor_name, elapsed,
            )
            self._record_generation_stats(population, elapsed)

            # Early stopping
            if best.fitness - self._best_fitness < cfg.min_fitness_improvement:
                self._stall_count += 1
            else:
                self._stall_count = 0
                self._best_fitness = best.fitness

            if self._stall_count >= cfg.early_stop_generations:
                logger.info("Early stopping at generation %d (no improvement for %d gens)",
                            gen, cfg.early_stop_generations)
                break

            if callback:
                callback(gen, population)

        # Return hall of fame sorted by fitness
        self._hall_of_fame.sort(key=lambda ind: ind.fitness, reverse=True)
        return self._hall_of_fame

    # ── Initialization ─────────────────────────────────────────────────────

    def _initialize_population(self) -> list[Expr]:
        """Generate initial population using ramped half-and-half."""
        cfg = self.config
        terminals = cfg.terminals or TERMINAL_FIELDS
        population = []
        for i in range(cfg.population_size):
            depth = random.randint(2, min(cfg.max_depth, 4))
            method = "grow" if i % 3 != 0 else "full"
            tree = random_expr(max_depth=depth, method=method, terminals=terminals)
            if tree.node_count() <= cfg.max_complexity:
                population.append(tree)
            else:
                population.append(random_expr(max_depth=2, method="grow", terminals=terminals))

        return population

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _evaluate_population(
        self,
        trees: list[Expr],
        data: pd.DataFrame,
        forward_returns: pd.Series,
        existing_factors: pd.DataFrame | None,
    ) -> list[Individual]:
        """Evaluate each tree: compile → compute → validate → fitness."""
        individuals = []
        for tree in trees:
            ind = self._evaluate_one(tree, data, forward_returns, existing_factors, self._generation)
            individuals.append(ind)
        return individuals

    def _evaluate_one(
        self,
        tree: Expr,
        data: pd.DataFrame,
        forward_returns: pd.Series,
        existing_factors: pd.DataFrame | None,
        generation: int = 0,
    ) -> Individual:
        """Evaluate a single expression tree."""
        cfg = self.config
        complexity = tree.node_count()
        depth = tree.depth()

        # Reject trivial trees — raw fields or single constants have no structure
        if depth <= 1:
            return Individual(
                tree=tree, factor_name="trivial", factor_cls=None,
                fitness=-999, ic_mean=0, ic_ir=0, hit_rate=0, auto_corr=1,
                complexity=complexity, depth=depth, generation=generation,
            )

        # Try to compile and compute
        try:
            factor_cls, factor_values = compile_and_validate(
                tree, data, factor_name=None
            )
        except Exception as e:
            logger.debug("Compile failed for %s: %s", repr(tree), e)
            return Individual(
                tree=tree, factor_name="invalid", factor_cls=None,
                fitness=-999, ic_mean=0, ic_ir=0, hit_rate=0, auto_corr=1,
                complexity=complexity, depth=depth, generation=generation,
            )

        # Validate
        try:
            result = self._validator.validate(
                factor_values=factor_values,
                forward_returns=forward_returns,
                factor_name=factor_cls.meta.name,
                existing_factors=existing_factors,
            )
        except Exception as e:
            logger.debug("Validation failed for %s: %s", repr(tree), e)
            return Individual(
                tree=tree, factor_name=factor_cls.meta.name, factor_cls=factor_cls,
                fitness=-999, ic_mean=0, ic_ir=0, hit_rate=0, auto_corr=1,
                complexity=complexity, depth=depth, generation=generation,
            )

        ic_mean = result.ic_mean
        ic_ir = result.ic_ir
        hit_rate = result.hit_rate
        auto_corr = result.auto_corr if not np.isnan(result.auto_corr) else 1.0

        # Bail early if core metrics are invalid
        if np.isnan(ic_mean) or np.isnan(ic_ir):
            return Individual(
                tree=tree, factor_name=factor_cls.meta.name, factor_cls=factor_cls,
                fitness=-999, ic_mean=0, ic_ir=0, hit_rate=0, auto_corr=1,
                complexity=complexity, depth=depth, generation=generation,
            )

        # Reject near-constant signals (e.g. lt() comparisons with non-overlapping
        # operand ranges).  IC_std ≈ 0 produces spuriously high IR.
        _ic_std = getattr(result, "ic_std", np.nan)
        if not np.isnan(_ic_std) and _ic_std < 0.005:
            return Individual(
                tree=tree, factor_name=factor_cls.meta.name, factor_cls=factor_cls,
                fitness=-999, ic_mean=ic_mean, ic_ir=ic_ir,
                hit_rate=hit_rate, auto_corr=auto_corr,
                complexity=complexity, depth=depth, generation=generation,
            )

        # Stability: min IC across sub-periods
        period_ics = [
            v.get("mean", 0) for v in result.ic_by_period.values()
        ]
        min_period_ic = min(period_ics) if period_ics else ic_mean
        if np.isnan(min_period_ic):
            min_period_ic = 0.0

        # Cap IR contribution to prevent degenerate near-constant signals
        # (e.g. lt() comparisons where operands never overlap in range)
        # from dominating fitness with IR → ∞.
        _ir_capped = min(max(ic_ir, 0), 5.0)

        # Fitness components
        fitness = (
            cfg.ic_mean_weight * abs(ic_mean) +
            cfg.ic_ir_weight * _ir_capped +
            cfg.stability_weight * min_period_ic +
            cfg.hit_rate_weight * hit_rate
        )

        # Guard against NaN fitness from any source
        if np.isnan(fitness):
            fitness = -999.0

        # Penalties
        if not np.isnan(auto_corr) and auto_corr > self._validator.max_auto_corr:
            fitness -= 0.1 * (auto_corr - self._validator.max_auto_corr)
        fitness -= cfg.parsimony_penalty * complexity
        if not result.passed:
            fitness -= 0.2  # penalty for not meeting all criteria

        return Individual(
            tree=tree,
            factor_name=factor_cls.meta.name,
            factor_cls=factor_cls,
            fitness=fitness,
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            auto_corr=auto_corr,
            complexity=complexity,
            depth=depth,
        )

    # ── Selection ──────────────────────────────────────────────────────────

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Tournament selection: pick best among k randomly chosen.

        Skips individuals with NaN/invalid fitness to prevent them from
        becoming parents and propagating degenerate trees.
        """
        cfg = self.config
        k = min(cfg.tournament_size, len(population))
        # Prefer valid individuals
        valid = [ind for ind in population if not np.isnan(ind.fitness) and ind.fitness > -900]
        if len(valid) >= 2:
            candidates = random.sample(valid, min(k, len(valid)))
        else:
            candidates = random.sample(population, k)
        return max(candidates, key=lambda ind: ind.fitness)

    # ── Crossover ──────────────────────────────────────────────────────────

    def _crossover(self, tree1: Expr, tree2: Expr) -> tuple[Expr, Expr]:
        """Subtree crossover: swap a random node from tree1 with one from tree2."""
        nodes1 = collect_all_nodes(tree1)
        nodes2 = collect_all_nodes(tree2)

        if len(nodes1) < 2 or len(nodes2) < 2:
            return tree1, tree2

        # Pick random non-root nodes
        node1 = random.choice(nodes1[1:] if len(nodes1) > 1 else nodes1)
        node2 = random.choice(nodes2[1:] if len(nodes2) > 1 else nodes2)

        # Swap: replace node1 in tree1 with node2, node2 in tree2 with node1
        new_tree1 = tree1.replace_child(node1, node2.clone())
        new_tree2 = tree2.replace_child(node2, node1.clone())

        # Enforce depth limits
        cfg = self.config
        if new_tree1.depth() > cfg.max_depth or new_tree1.node_count() > cfg.max_complexity:
            new_tree1 = tree1
        if new_tree2.depth() > cfg.max_depth or new_tree2.node_count() > cfg.max_complexity:
            new_tree2 = tree2

        return new_tree1, new_tree2

    # ── Mutation ───────────────────────────────────────────────────────────

    def _mutate(self, tree: Expr) -> Expr:
        """Apply one of several mutation operators at random."""
        cfg = self.config
        choices = ["subtree", "window", "operator", "constant", "unary"]
        weights = [
            cfg.subtree_mutation_weight,
            cfg.window_mutation_weight,
            cfg.operator_mutation_weight,
            cfg.constant_mutation_weight,
            cfg.unary_mutation_weight,
        ]
        op = random.choices(choices, weights=weights, k=1)[0]

        if op == "subtree":
            return self._mutate_subtree(tree)
        elif op == "window":
            return self._mutate_window(tree)
        elif op == "operator":
            return self._mutate_operator(tree)
        elif op == "constant":
            return self._mutate_constant(tree)
        elif op == "unary":
            return self._mutate_unary(tree)
        return tree

    def _mutate_subtree(self, tree: Expr) -> Expr:
        """Replace a random subtree with a new random tree."""
        nodes = collect_all_nodes(tree)
        if not nodes:
            return tree
        target = random.choice(nodes)
        new_subtree = random_expr(
            max_depth=random.randint(1, 3),
            method=random.choice(["grow", "full"]),
            terminals=self.config.terminals or TERMINAL_FIELDS,
        )
        return tree.replace_child(target, new_subtree)

    def _mutate_window(self, tree: Expr) -> Expr:
        """Change the window/periods/group_field parameter on parameterized nodes."""
        all_nodes = collect_all_nodes(tree)
        param_nodes = [n for n in all_nodes if isinstance(n, (RollingOp, TimeSeriesOp, GroupedCrossSectionalOp))]
        if not param_nodes:
            return tree

        target = random.choice(param_nodes)
        if isinstance(target, RollingOp):
            target.window = random.choice(ROLLING_WINDOWS)
        elif isinstance(target, TimeSeriesOp):
            target.periods = random.choice(TS_PERIODS)
        elif isinstance(target, GroupedCrossSectionalOp):
            target.group_field = random.choice(GROUP_FIELDS)

        return tree

    def _mutate_operator(self, tree: Expr) -> Expr:
        """Change operator type on a node while keeping the same arity."""
        all_nodes = collect_all_nodes(tree)
        op_nodes = [
            n for n in all_nodes
            if isinstance(n, (UnaryOp, BinaryOp, RollingOp, CrossSectionalOp,
                             GroupedCrossSectionalOp, TimeSeriesOp))
        ]
        if not op_nodes:
            return tree

        target = random.choice(op_nodes)

        if isinstance(target, UnaryOp):
            new_op = random.choice([o for o in UNARY_OPS if o != target.op])
            target.op = new_op
        elif isinstance(target, BinaryOp):
            new_op = random.choice([o for o in BINARY_OPS if o != target.op])
            target.op = new_op
        elif isinstance(target, RollingOp):
            new_op = random.choice([o for o in ROLLING_OPS if o != target.op])
            target.op = new_op
        elif isinstance(target, CrossSectionalOp):
            new_op = random.choice([o for o in CS_OPS if o != target.op])
            target.op = new_op
        elif isinstance(target, GroupedCrossSectionalOp):
            new_op = random.choice([o for o in CS_GROUP_OPS if o != target.op])
            target.op = new_op
        elif isinstance(target, TimeSeriesOp):
            new_op = random.choice([o for o in TS_OPS if o != target.op])
            target.op = new_op

        return tree

    def _mutate_constant(self, tree: Expr) -> Expr:
        """Perturb a ConstExpr value."""
        const_nodes = [n for n in collect_all_nodes(tree) if isinstance(n, ConstExpr)]
        if not const_nodes:
            return tree

        target = random.choice(const_nodes)
        # Perturb by ±20% on log scale or add small noise
        if random.random() < 0.5:
            target.value += random.uniform(-0.5, 0.5)
        else:
            target.value *= random.uniform(0.5, 1.5)
        target.value = round(target.value, 4)
        return tree

    def _mutate_unary(self, tree: Expr) -> Expr:
        """Wrap a random node in a unary operator, or remove a unary wrapper."""
        if random.random() < 0.5:
            # Wrap: insert a unary op above a random node
            nodes = collect_all_nodes(tree)
            if not nodes:
                return tree
            target = random.choice(nodes)
            new_unary = UnaryOp(random.choice(UNARY_OPS), target.clone())
            return tree.replace_child(target, new_unary)
        else:
            # Unwrap: remove a unary op, exposing its child
            unary_nodes = [n for n in collect_all_nodes(tree) if isinstance(n, UnaryOp)]
            if not unary_nodes:
                return tree
            target = random.choice(unary_nodes)
            return tree.replace_child(target, target.child.clone())

    # ── Generation Stats ───────────────────────────────────────────────────

    def _record_generation_stats(
        self, population: list[Individual], elapsed: float,
    ) -> GenerationStats:
        """Compute and record per-generation population statistics."""
        valid = [ind for ind in population if not np.isnan(ind.fitness) and ind.fitness > -900]
        if not valid:
            valid = population

        fitnesses = [ind.fitness for ind in valid]
        ics = [abs(ind.ic_mean) for ind in valid if not np.isnan(ind.ic_mean)]
        irs = [ind.ic_ir for ind in valid if not np.isnan(ind.ic_ir) and ind.ic_ir > -10]
        depths = [ind.depth for ind in valid]
        nodes = [ind.complexity for ind in valid]

        stats = GenerationStats(
            generation=self._generation,
            best_fitness=valid[0].fitness if valid else -999,
            mean_fitness=float(np.mean(fitnesses)) if fitnesses else -999,
            median_fitness=float(np.median(fitnesses)) if fitnesses else -999,
            worst_fitness=min(fitnesses) if fitnesses else -999,
            std_fitness=float(np.std(fitnesses)) if fitnesses else 0,
            best_ic=valid[0].ic_mean if valid else 0,
            mean_ic=float(np.mean(ics)) if ics else 0,
            best_ir=valid[0].ic_ir if valid else 0,
            mean_ir=float(np.mean(irs)) if irs else 0,
            best_depth=valid[0].depth if valid else 0,
            mean_depth=float(np.mean(depths)) if depths else 0,
            best_nodes=valid[0].complexity if valid else 0,
            mean_nodes=float(np.mean(nodes)) if nodes else 0,
            valid_count=len(valid),
            total_count=len(population),
            hall_of_fame_size=len(self._hall_of_fame),
            stall_count=self._stall_count,
            elapsed_seconds=elapsed,
        )
        self._history.append(stats)
        return stats

    @property
    def generation_history(self) -> list[GenerationStats]:
        return self._history

    # ── Hall of Fame ───────────────────────────────────────────────────────

    def _update_hall_of_fame(self, elites: list[Individual]) -> None:
        """Maintain a deduplicated hall of fame, rejecting NaN-fitness entries."""
        existing_names = {ind.factor_name for ind in self._hall_of_fame}
        for ind in elites:
            if ind.factor_name not in existing_names and not np.isnan(ind.fitness) and ind.fitness > -100:
                self._hall_of_fame.append(ind)
                existing_names.add(ind.factor_name)

        # Keep top 50
        self._hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        if len(self._hall_of_fame) > 50:
            self._hall_of_fame = self._hall_of_fame[:50]

    @property
    def generation(self) -> int:
        return self._generation

    def history_to_dict(self) -> list[dict]:
        """Serialize generation history for persistence in gp_factors.json."""
        import dataclasses
        return [dataclasses.asdict(s) for s in self._history]
