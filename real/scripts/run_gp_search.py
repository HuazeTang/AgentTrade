"""Standalone scaled-up GP factor search (~5-6 hour run).

Usage: .venv/bin/python scripts/run_gp_search.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Limit NumExpr threads to avoid oversubscription with GP worker threads
# Limit internal NumExpr threads to avoid nested-parallelism contention
# with the GP worker pool.  2 NumExpr threads × 3 GP workers = 6 threads
# on a 10-core machine, avoiding oversubscription.
for _env_var in ("NUMEXPR_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    if _env_var not in os.environ:
        os.environ[_env_var] = "2"
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from data.cache import read_daily
from discovery.expr import TERMINAL_FIELDS
from discovery.gp import GPEngine, GPConfig
from discovery.pure_factor import FactorMimickingPortfolio, walk_forward_validate
from discovery.validate import FactorValidator
from factor.engine import FactorEngine
from factor.registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gp_search")

FORWARD_PERIODS = 5
TRADING_DAYS_PER_YEAR = 244
GP_FACTORS_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "gp_factors.json"

# ── Scaled-up GP configuration (target: 5-6 hours) ────────────────────────
# 400 pop × 50 gens = 20k evaluations.  With 500 stocks / 700 dates and
# 2 workers, each eval ~1.0s, so ~20k / 1.9 × 1.0 ≈ 10,500s ≈ 3h.
# Plus per-gen top-10 pure factor eval and overhead → 5-6h total.
GP_CFG = GPConfig(
    population_size=400,
    max_generations=50,
    tournament_size=5,
    crossover_prob=0.7,
    mutation_prob=0.5,
    elite_count=8,
    max_depth=5,
    max_complexity=25,
    early_stop_generations=18,
    parsimony_penalty=0.0003,
    terminal_prob=0.3,
    ic_mean_weight=0.25,
    ic_ir_weight=0.35,
    stability_weight=0.25,
    hit_rate_weight=0.15,
    max_workers=2,
    pure_factor_top_n=0,
    pure_factor_blend_weight=0.0,
    refresh_stall_threshold=12,
    refresh_fraction=0.30,
    same_factor_stall_threshold=5,
    subtree_mutation_weight=0.20,
    window_mutation_weight=0.15,
    operator_mutation_weight=0.15,
    constant_mutation_weight=0.10,
    unary_mutation_weight=0.10,
    wrap_rolling_mutation_weight=0.30,
)


def load_data() -> pd.DataFrame:
    """Load daily data with derived fields (same setup as GP pipeline)."""
    df = read_daily(date(2023, 1, 1), date(2026, 5, 21))
    if df.empty:
        raise RuntimeError("No data loaded from cache")

    close = df["close"].unstack()
    volume = df["volume"].unstack()
    amount = df["amount"].unstack()
    high = df["high"].unstack()
    low = df["low"].unstack()

    derived = {
        "ret_5d": close.pct_change(5).stack(),
        "ret_20d": close.pct_change(20).stack(),
        "ret_60d": close.pct_change(60).stack(),
        "vol_20d": volume.rolling(20, min_periods=5).mean().stack(),
        "vol_60d": volume.rolling(60, min_periods=10).mean().stack(),
        "hl_ratio": ((high - low) / close.clip(lower=1e-8)).stack(),
        "vol_ratio": (volume / volume.shift(1).clip(lower=1e-8)).stack(),
        "amihud": (close.pct_change().abs() / amount.clip(lower=1e-8)).stack(),
    }
    for name, series in derived.items():
        if name not in df.columns:
            df[name] = series

    return df


def select_stock_pool(
    data: pd.DataFrame,
    train_start: date,
    train_end: date,
    target_n: int = 900,
    min_per_industry: int = 3,
) -> list[str]:
    """Select a diverse stock pool stratified by Shenwan L1 industry.

    Within each industry, stocks are ranked by average daily turnover amount
    during the training period. Top N stocks per industry are sampled
    proportionally to sqrt(industry_size) for diversity.
    """
    from collections import Counter

    from data.industry import build_industry_map

    all_symbols = sorted(data.index.get_level_values("symbol").unique().tolist())
    logger.info("Universe: %d symbols, building industry map...", len(all_symbols))

    industry_map = build_industry_map(all_symbols)
    by_industry: dict[str, list[tuple[str, float]]] = {}
    unclassified: list[str] = []

    for sym in all_symbols:
        ind = industry_map.get(sym, "综合")
        if ind not in by_industry:
            by_industry[ind] = []
        by_industry[ind].append(sym)

    # Compute average daily amount per stock during training period
    train_mask = (
        (data.index.get_level_values("trade_date") >= pd.Timestamp(train_start))
        & (data.index.get_level_values("trade_date") <= pd.Timestamp(train_end))
    )
    train_data = data.loc[train_mask]
    avg_amount = train_data.groupby("symbol")["amount"].mean()

    # Within each industry, rank by liquidity and select top N
    industry_counts = {ind: len(syms) for ind, syms in by_industry.items()}
    # Proportional to sqrt(size) for diversity, then normalize to target_n
    sqrt_sizes = {ind: max(min_per_industry, int(np.sqrt(cnt) * 2.5))
                  for ind, cnt in industry_counts.items()}
    total_alloc = sum(sqrt_sizes.values())
    scale = target_n / total_alloc
    allocations = {ind: max(min_per_industry, int(n * scale))
                   for ind, n in sqrt_sizes.items()}

    logger.info("Industry allocations (target=%d stocks across %d industries):",
                 target_n, len(allocations))
    for ind in sorted(allocations, key=allocations.get, reverse=True)[:10]:
        logger.info("  %s: %d/%d", ind, allocations[ind], industry_counts[ind])

    selected: list[str] = []
    for ind, syms in by_industry.items():
        n_select = allocations.get(ind, min_per_industry)
        # Rank by liquidity within industry
        syms_with_liq = [(s, avg_amount.get(s, 0) or 0) for s in syms]
        syms_with_liq.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in syms_with_liq[:n_select]]
        selected.extend(top)

    logger.info("Selected %d stocks across %d industries", len(selected), len(by_industry))
    return selected


def _create_pure_factor_callback(
    daily_cache: pd.DataFrame,
    forward_returns: pd.Series,
    existing_factor_values: pd.DataFrame,
    top_n: int = 10,
    blend_weight: float = 0.3,
):
    """Create a per-generation callback that blends pure factor Sharpe into fitness."""
    from discovery.pure_factor import FactorMimickingPortfolio

    portfolio = FactorMimickingPortfolio(
        total_leverage=1.0, rebalance_freq="daily", long_only=False, use_ranks=True,
    )

    def callback(sorted_population: list, generation: int) -> None:
        if top_n <= 0 or blend_weight <= 0:
            return
        for i, ind in enumerate(sorted_population):
            if i >= top_n:
                break
            if ind.factor_cls is None:
                continue
            try:
                fv = ind.factor_cls().compute(daily_cache)
                if fv.notna().sum() < 10:
                    continue
                pfm = portfolio.evaluate(fv, forward_returns)
                pure_sr = pfm.sharpe_ratio
                if np.isfinite(pure_sr):
                    blended = (1.0 - blend_weight) * ind.fitness + blend_weight * pure_sr
                    sorted_population[i] = ind._replace(fitness=blended)
            except Exception:
                pass

    return callback


def _create_llm_diversity_callback():
    """Create a callback that asks LLM to analyze stuck factors and propose variants.

    Called when the same factor dominates for same_factor_stall_threshold consecutive
    generations.  The LLM receives the stuck factor's expression tree (converted to
    a human-readable string), its metrics, and available operators, then proposes
    N variant expressions that are compiled into seed Individuals.

    Returns compiled Individuals, or None if LLM is unavailable or fails.
    """
    from agent.llm_client import LLMClient
    from discovery.llm_seed import LLMSeedGenerator
    from discovery.operators import operator_registry

    llm = LLMClient()
    if not llm.configured:
        logger.warning("LLM diversity injection disabled (no API key)")
        return lambda _ind, _gen: None

    seed_gen = LLMSeedGenerator()

    def _tree_to_shorthand(tree) -> str:
        """Convert an Expr tree to shorthand format parseable by _parse_hint()."""
        from discovery.expr import (VarExpr, ConstExpr, UnaryOp, BinaryOp,
                                     RollingOp, CrossSectionalOp, GroupedCrossSectionalOp,
                                     TimeSeriesOp, TernaryOp)
        if isinstance(tree, VarExpr):
            return tree.name
        if isinstance(tree, ConstExpr):
            return f"{tree.value}"
        if isinstance(tree, UnaryOp):
            return f"{tree.op}({_tree_to_shorthand(tree.child)})"
        if isinstance(tree, BinaryOp):
            return f"{tree.op}({_tree_to_shorthand(tree.left)}, {_tree_to_shorthand(tree.right)})"
        if isinstance(tree, RollingOp):
            inner = _tree_to_shorthand(tree.child)
            if tree.right is not None:
                return f"{tree.op}({inner}, {_tree_to_shorthand(tree.right)}, {tree.window})"
            if tree.quantile is not None:
                return f"{tree.op}({inner}, {tree.window}, {tree.quantile})"
            return f"{tree.op}({inner}, {tree.window})"
        if isinstance(tree, CrossSectionalOp):
            return f"{tree.op}({_tree_to_shorthand(tree.child)})"
        if isinstance(tree, GroupedCrossSectionalOp):
            return f"{tree.op}({_tree_to_shorthand(tree.child)}, {tree.group_field})"
        if isinstance(tree, TimeSeriesOp):
            return f"{tree.op}({_tree_to_shorthand(tree.child)}, {tree.periods})"
        if isinstance(tree, TernaryOp):
            return (f"if_then({_tree_to_shorthand(tree.cond)}, "
                    f"{_tree_to_shorthand(tree.then_branch)}, "
                    f"{_tree_to_shorthand(tree.else_branch)})")
        return repr(tree)

    def callback(stuck_ind, generation: int):
        tree_shorthand = _tree_to_shorthand(stuck_ind.tree)
        name = stuck_ind.factor_name
        ic_mean = stuck_ind.ic_mean
        ic_ir = stuck_ind.ic_ir

        prompt = f"""You are a quantitative researcher debugging a Genetic Programming factor discovery run.

## Problem
The GP has been stuck for {generation} generations — the same factor expression keeps dominating the population. The search needs fresh ideas to escape this local optimum.

## Stuck Factor
Expression: {tree_shorthand}
IC mean: {ic_mean:.4f}
IC IR: {ic_ir:.3f}
Name: {name}

## Available Operators
{operator_registry.to_llm_prompt()}

## Task
Propose exactly 5 variant factor expressions that:
1. **Modify or extend** the stuck factor — change operators, swap data fields, adjust windows, add/remove transformations
2. **Explore different directions** — if the factor uses rolling windows, try cross-sectional ops; if it's momentum, try mean-reversion or volume-based variants
3. **Have economic intuition** — each variant should make sense as a trading signal
4. **Keep complexity reasonable** — depth 3-5, similar to the stuck factor

Output a JSON array:
[{{"name": "snake_case_name", "intuition": "1 sentence why this differs from the stuck factor",
   "category": "momentum|value|quality|volatility|liquidity|volume_price|risk|composite",
   "expression_hint": "op(field_or_expr, param_or_expr)"}}]

## Expression Hint Syntax (CRITICAL)
Use ONLY this function-call format — the same format as the Stuck Factor above:
- Variables: close, high, low, volume, ret_5d, ret_20d, vol_20d, ret_60d, etc.
- Constants: 0.5, -1.0, 1.0385
- Unary: neg(x), abs(x), log(x), sqrt(x), sign(x)
- Binary: add(x,y), sub(x,y), mul(x,y), div(x,y), gt(x,y), lt(x,y), max(x,y), min(x,y)
- Rolling: ts_mean(x, w), ts_std(x, w), ts_min(x, w), ts_max(x, w), ts_quantile(x, w, q), ts_corr(x, y, w), ts_skew(x, w)
- Cross-sectional: cs_rank(x), cs_zscore(x)
- Time-series: pct_change(x, periods), delta(x, periods), ts_lag(x, periods)
- Ternary: if_then(cond, then_branch, else_branch)

Examples of valid expression_hint values:
  - lt(ts_mean(close, 20), 1.05)
  - cs_rank(lt(high, ret_20d))
  - ts_quantile(high, 30, 0.25)
  - if_then(gt(ret_5d, 0.02), close, neg(close))
  - mul(vol_20d, ts_std(ret_5d, 60))

IMPORTANT: Return ONLY the JSON array, no markdown fences, no explanation."""

        try:
            proposals = llm.chat_json(prompt, expected_keys=None)
        except Exception as e:
            logger.warning("LLM diversity call failed gen %d: %s", generation, e)
            return None

        if not isinstance(proposals, list) or len(proposals) == 0:
            logger.info("LLM diversity: no proposals returned")
            return None

        seeds = seed_gen.compile_seeds(proposals)
        logger.info(
            "LLM diversity gen %d: %d proposals → %d compiled seeds",
            generation, len(proposals), len(seeds),
        )
        return seeds if seeds else None

    return callback


def main():
    t_start = time.time()

    # ── 1. Load and prepare data ────────────────────────────────────────────
    logger.info("Loading market data...")
    data = load_data()
    logger.info("Loaded %d rows, %d dates, %d symbols",
                 len(data),
                 data.index.get_level_values("trade_date").nunique(),
                 data.index.get_level_values("symbol").nunique())

    # Use fixed training window matching TRAIN_PERIOD from run_agent_simulation.
    # Factor values are computed on ALL data (2022-2026) so rolling/ts operators
    # have sufficient lookback; IC is only computed within this training window.
    TRAIN_START = date(2024, 8, 30)
    TRAIN_END = date(2025, 10, 16)
    all_dates = sorted(data.index.get_level_values("trade_date").unique())
    train_dates = [d for d in all_dates if TRAIN_START <= d.date() <= TRAIN_END]
    logger.info("GP training: %s ~ %s (%d days)",
                 train_dates[0].date(), train_dates[-1].date(), len(train_dates))

    # ── 1b. Select industry-balanced stock pool ────────────────────────────
    pool = select_stock_pool(data, TRAIN_START, TRAIN_END, target_n=500)
    data = data[data.index.get_level_values("symbol").isin(pool)].copy()
    logger.info("After pool filter: %d rows, %d symbols",
                 len(data), data.index.get_level_values("symbol").nunique())

    train_mask = data.index.get_level_values("trade_date").isin(train_dates)
    gp_data = data.copy()

    # Forward returns
    close_all = data["close"].unstack()
    fwd_ret_all = close_all.pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()
    fwd_ret_all.name = "fwd_ret"
    fwd_ret_train = fwd_ret_all.loc[gp_data.loc[train_mask].index]

    # ── 2. Compute baseline factors ─────────────────────────────────────────
    logger.info("Computing baseline factors...")
    import factor.factors as _  # noqa: F401 — register all factors
    from factor.registry import registry as factor_registry

    all_registered = factor_registry.list_all()

    # Exclude factors that need columns with bad dtypes (e.g. market_cap is
    # all-NaN object dtype which breaks np.log). Also skip market_dd_beta_20d
    # whose compute() returns a malformed index (Timestamps in symbol level).
    SKIP_FACTORS = {"ln_market_cap", "market_dd_beta_20d"}

    to_compute = [f for f in all_registered if f not in SKIP_FACTORS]
    logger.info("%d registered factors, %d selected for computation",
                 len(all_registered), len(to_compute))

    engine = FactorEngine()
    existing_df = engine.compute(to_compute, gp_data)
    coverage = existing_df.notna().mean()
    usable = coverage[coverage >= 0.5].index.tolist()
    logger.info("%d/%d factors usable (>=50%% coverage)", len(usable), len(to_compute))

    if len(usable) < 5:
        logger.error("Too few usable baseline factors (%d), aborting", len(usable))
        sys.exit(1)

    existing_df = existing_df[usable].fillna(0.0)

    # Extend terminals with baseline factor columns
    raw_field_set = set(gp_data.columns) | set(TERMINAL_FIELDS)
    factor_terminals = [f for f in usable if f not in raw_field_set]
    if factor_terminals:
        joinable = [f for f in factor_terminals if f in existing_df.columns]
        if joinable:
            gp_data = gp_data.join(existing_df[joinable], how="left")
        logger.info("Injected %d factor columns as GP terminals", len(factor_terminals))

    extended_terminals = list(TERMINAL_FIELDS) + factor_terminals
    GP_CFG.terminals = extended_terminals

    common_idx = existing_df.index.intersection(fwd_ret_train.dropna().index)
    gp_fwd = fwd_ret_train.loc[common_idx]
    gp_existing = existing_df.loc[common_idx]

    if len(gp_fwd) < 100:
        logger.error("Too few GP training samples (%d), aborting", len(gp_fwd))
        sys.exit(1)

    # ── 3. Load prior GP factors for context ────────────────────────────────
    prior_gp_names: list[str] = []
    prior_data: dict = {}
    if GP_FACTORS_PATH.exists():
        with open(GP_FACTORS_PATH, "r", encoding="utf-8") as f:
            prior_data = json.load(f)
        prior_gp_names = [e["name"] for e in prior_data.get("gp_factors", [])
                          if e.get("accepted", True)]
        logger.info("Loaded %d prior accepted GP factors", len(prior_gp_names))

        # Re-compile prior GP factors so they're available in the FactorEngine registry.
        # gp_factors.json stores the expression tree as JSON; we need to deserialize
        # and compile it back to a Factor class on each run.
        from discovery.expr import Expr
        from discovery.compiler import compile_expr

        for entry in prior_data.get("gp_factors", []):
            name = entry.get("name", "")
            if name not in prior_gp_names:
                continue
            try:
                expr = Expr.from_dict(entry["expression"])
                compile_expr(expr, factor_name=name, register=True)
            except Exception:
                logger.debug("Prior GP factor '%s' already registered or failed to compile", name)
        logger.info("Re-compiled prior GP factors into registry")

    # ── 4. Run GP evolution ─────────────────────────────────────────────────
    logger.info("Starting GP evolution: pop=%d, gens=%d, workers=%d",
                 GP_CFG.population_size, GP_CFG.max_generations, GP_CFG.max_workers)
    logger.info("Estimated runtime: 5-6 hours")

    gp = GPEngine(config=GP_CFG)

    llm_diversity_cb = _create_llm_diversity_callback()

    best_individuals = gp.evolve(
        data=gp_data,
        forward_returns=gp_fwd,
        existing_factors=gp_existing,
        llm_diversity_callback=llm_diversity_cb,
    )
    elapsed = time.time() - t_start
    logger.info("GP evolution complete: %.0fs (%.1f hours), %d generations",
                 elapsed, elapsed / 3600, gp.generation)

    # ── 5. Validate top candidates ──────────────────────────────────────────
    validator = FactorValidator()
    max_new = 5
    new_factors: list[dict] = []
    validated_values: dict[str, pd.Series] = {}
    hall = sorted(best_individuals, key=lambda x: x.fitness, reverse=True)[:max_new * 2]

    for ind in hall:
        if ind.factor_cls is None or ind.fitness < -100:
            continue
        if len(new_factors) >= max_new:
            break

        try:
            factor_vals = ind.factor_cls().compute(gp_data)
            result = validator.validate(
                factor_values=factor_vals,
                forward_returns=fwd_ret_train,
                factor_name=ind.factor_name,
                existing_factors=existing_df,
            )
            if result.passed:
                new_factors.append({
                    "name": ind.factor_name,
                    "category": ind.factor_cls.meta.category,
                    "expression": ind.tree.to_dict(),
                    "generation": ind.generation,
                    "fitness": ind.fitness,
                    "ic_mean": ind.ic_mean,
                    "ic_ir": ind.ic_ir,
                    "ic_std": getattr(result, "ic_std", 0),
                    "hit_rate": ind.hit_rate,
                    "auto_corr": ind.auto_corr,
                    "max_corr_existing": getattr(result, "max_corr_existing", 0),
                    "complexity": ind.complexity,
                    "depth": ind.depth,
                    "validation_passed": True,
                    "wf_ic_mean": getattr(result, "wf_ic_mean", 0),
                    "factor_cls": ind.factor_cls,
                    "factor_vals": factor_vals,
                })
                existing_df[ind.factor_name] = factor_vals
                validated_values[ind.factor_name] = factor_vals
                logger.info("Validated: %s (gen=%d, fitness=%.4f, IC=%.4f, IR=%.3f)",
                             ind.factor_name, ind.generation, ind.fitness,
                             ind.ic_mean, ind.ic_ir)
            else:
                logger.info("Rejected: %s (gen=%d, %s)",
                             ind.factor_name, ind.generation,
                             ", ".join(result.failures[:2]))
        except Exception as e:
            logger.debug("Validation error for %s: %s", ind.factor_name, e)

    if not new_factors:
        logger.warning("No GP factors passed validation")
        _save_results(gp, [], prior_data, elapsed)
        return

    # ── 6. Walk-forward validation ──────────────────────────────────────────
    for fmeta in new_factors:
        fv = fmeta["factor_vals"]
        fname = fmeta["name"]
        try:
            wf_result = walk_forward_validate(
                fv, gp_fwd,
                window_size=min(252, len(train_dates) // 2),
                step_size=min(63, len(train_dates) // 6),
                min_windows=2,
            )
            fmeta["walk_forward"] = wf_result
            logger.info("WF %s: test_SR=%.3f (±%.3f) over %d windows",
                         fname, wf_result["mean_test_sharpe"],
                         wf_result["sharpe_std"], wf_result["n_windows"])
        except Exception as e:
            logger.warning("WF failed for %s: %s", fname, e)
            fmeta["walk_forward"] = {"error": str(e)}

    # ── 7. Pure factor evaluation ───────────────────────────────────────────
    portfolio = FactorMimickingPortfolio(
        total_leverage=1.0, rebalance_freq="daily", long_only=False, use_ranks=True,
    )
    fwd_ret = close_all.pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()

    # Solo evaluation
    for fmeta in new_factors:
        try:
            pfm = portfolio.evaluate(fmeta["factor_vals"], fwd_ret)
            fmeta["pure_solo"] = pfm.to_dict()
            logger.info("Pure solo %s: SR=%.3f, DD=%.1f%%, cumRet=%.1f%%",
                         fmeta["name"], pfm.sharpe_ratio,
                         pfm.max_drawdown * 100, pfm.cumulative_return * 100)
        except Exception as e:
            logger.warning("Pure solo failed for %s: %s", fmeta["name"], e)
            fmeta["pure_solo"] = {"error": str(e)}

    # Cumulative evaluation
    accepted_factors: list[dict] = []
    if prior_gp_names:
        prior_fv_df = engine.compute(prior_gp_names, gp_data)
        for entry in prior_data.get("gp_factors", []):
            n = entry["name"]
            if n in prior_gp_names and n in prior_fv_df.columns:
                accepted_factors.append({
                    "name": n,
                    "pure_cumulative": entry.get("pure_cumulative", {}),
                    "factor_vals": prior_fv_df[n],
                })
        if accepted_factors:
            logger.info("Cumulative eval: preloaded %d prior GP factors", len(accepted_factors))

    # Collect all factor values for weight calibration
    all_factor_vals = {}
    for fmeta in new_factors:
        all_factor_vals[fmeta["name"]] = fmeta["factor_vals"]
    for fm in accepted_factors:
        all_factor_vals[fm["name"]] = fm["factor_vals"]
    for col in gp_existing.columns:
        if col not in all_factor_vals:
            all_factor_vals[col] = gp_existing[col]

    # Simple equal-weight cumulative evaluation
    prior_cumul_sharpe = 0.0
    if accepted_factors:
        prior_names = [fm["name"] for fm in accepted_factors]
        prior_w = {n: 1.0 / len(prior_names) for n in prior_names}
        prior_vals = {n: all_factor_vals[n] for n in prior_names if n in all_factor_vals}
        prior_pfm = portfolio.evaluate_composite(prior_w, prior_vals, fwd_ret)
        prior_cumul_sharpe = prior_pfm.sharpe_ratio
        logger.info("Prior GP factors (%d) cumulative SR: %.3f", len(prior_names), prior_cumul_sharpe)

    for fmeta in new_factors:
        fname = fmeta["name"]
        test_names = [fm["name"] for fm in accepted_factors] + [fname]
        try:
            test_weights = {n: 1.0 / len(test_names) for n in test_names}
            test_values = {n: all_factor_vals[n] for n in test_names if n in all_factor_vals}
            pfm = portfolio.evaluate_composite(test_weights, test_values, fwd_ret)
            fmeta["pure_cumulative"] = pfm.to_dict()
            cur_sharpe = pfm.sharpe_ratio

            if len(accepted_factors) == 0:
                fmeta["accepted"] = bool(cur_sharpe >= 0.3)
            else:
                fmeta["accepted"] = bool(cur_sharpe >= prior_cumul_sharpe + 0.02)

            if fmeta["accepted"]:
                accepted_factors.append(fmeta)
                logger.info("Cumul +%s: SR=%.3f -> ACCEPTED", fname, cur_sharpe)
            else:
                logger.info("Cumul +%s: SR=%.3f -> REJECTED", fname, cur_sharpe)
        except Exception as e:
            logger.warning("Cumul eval failed for %s: %s", fname, e)
            fmeta["pure_cumulative"] = {"error": str(e)}
            fmeta["accepted"] = False

    # Register accepted factors
    for fmeta in new_factors:
        try:
            registry.register(fmeta["factor_cls"])
        except ValueError:
            pass

    # ── 8. Save results ─────────────────────────────────────────────────────
    _save_results(gp, new_factors, prior_data, elapsed)


def _save_results(gp: GPEngine, new_factors: list[dict], prior_data: dict, elapsed: float) -> None:
    """Merge with existing gp_factors.json and save."""

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if hasattr(o, "item"):
                return o.item()
            return super().default(o)

    # Build persistence entries (strip non-serializable fields)
    gp_meta = []
    for fmeta in new_factors:
        entry = {
            "name": fmeta["name"],
            "category": fmeta["category"],
            "expression": fmeta["expression"],
            "generation": fmeta["generation"],
            "fitness": fmeta["fitness"],
            "ic_mean": fmeta["ic_mean"],
            "ic_ir": fmeta["ic_ir"],
            "ic_std": fmeta.get("ic_std", 0),
            "hit_rate": fmeta["hit_rate"],
            "auto_corr": fmeta["auto_corr"],
            "max_corr_existing": fmeta.get("max_corr_existing", 0),
            "complexity": fmeta["complexity"],
            "depth": fmeta["depth"],
            "validation_passed": fmeta.get("validation_passed", True),
            "wf_ic_mean": fmeta.get("wf_ic_mean", 0),
            "walk_forward": fmeta.get("walk_forward", {}),
            "pure_solo": fmeta.get("pure_solo", {}),
            "pure_cumulative": fmeta.get("pure_cumulative", {}),
            "discovered_at": date.today().isoformat(),
            "accepted": fmeta.get("accepted", False),
        }
        gp_meta.append(entry)

    final_metrics = new_factors[-1].get("pure_cumulative", {}) if new_factors else {}
    persistence = {
        "gp_factors": gp_meta,
        "evolution_history": gp.history_to_dict(),
        "meta": {
            "discovered_at": datetime.now().isoformat(),
            "population_size": GP_CFG.population_size,
            "max_generations": GP_CFG.max_generations,
            "accepted_count": sum(1 for f in new_factors if f.get("accepted")),
            "total_candidates": len(new_factors),
            "final_pure_sharpe": final_metrics.get("sharpe_ratio", 0),
            "final_pure_drawdown": final_metrics.get("max_drawdown", 0),
            "elapsed_hours": round(elapsed / 3600, 2),
        },
    }

    GP_FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing
    existing_factors: dict[str, dict] = {}
    existing_history: list[dict] = []
    if GP_FACTORS_PATH.exists():
        with open(GP_FACTORS_PATH, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        for entry in old_data.get("gp_factors", []):
            existing_factors[entry["name"]] = entry
        existing_history = old_data.get("evolution_history", [])
        logger.info("Merging with %d existing GP factors", len(existing_factors))

    for entry in gp_meta:
        name = entry["name"]
        if name in existing_factors:
            old_fit = existing_factors[name].get("fitness", -999)
            if entry.get("fitness", -999) > old_fit:
                existing_factors[name] = entry
        else:
            existing_factors[name] = entry

    merged_factors = sorted(existing_factors.values(),
                            key=lambda e: e.get("fitness", -999), reverse=True)

    seen_gens = {h.get("generation") for h in existing_history}
    for h in persistence.get("evolution_history", []):
        if h.get("generation") not in seen_gens:
            existing_history.append(h)
            seen_gens.add(h.get("generation"))

    output = {
        "gp_factors": merged_factors,
        "evolution_history": existing_history,
        "meta": persistence["meta"],
    }

    with open(GP_FACTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info("Saved %d factors to %s", len(merged_factors), GP_FACTORS_PATH)

    # Print summary
    print(f"\n{'='*80}")
    print(f"GP SEARCH COMPLETE — {elapsed/3600:.1f} hours")
    print(f"{'='*80}")
    print(f"  Generations:         {gp.generation}")
    print(f"  Candidates found:    {len(new_factors)}")
    print(f"  Accepted:            {sum(1 for f in new_factors if f.get('accepted'))}")
    print(f"  Total factors saved: {len(merged_factors)}")
    print(f"  Output:              {GP_FACTORS_PATH}")
    print(f"{'='*80}")

    for fmeta in new_factors:
        solo = fmeta.get("pure_solo", {})
        print(f"\n  {fmeta['name']} {'[ACCEPTED]' if fmeta.get('accepted') else '[REJECTED]'}")
        print(f"    Category: {fmeta['category']}, Gen: {fmeta['generation']}, "
              f"Depth: {fmeta['depth']}, Complexity: {fmeta['complexity']}")
        print(f"    IC: {fmeta['ic_mean']:.4f}, IR: {fmeta['ic_ir']:.3f}, "
              f"Hit: {fmeta['hit_rate']:.2%}")
        if solo and "error" not in solo:
            print(f"    Solo:  SR={solo.get('sharpe_ratio', 0):.3f}, "
                  f"DD={solo.get('max_drawdown', 0)*100:.1f}%, "
                  f"Ret={solo.get('cumulative_return', 0)*100:.1f}%")
        cum = fmeta.get("pure_cumulative", {})
        if cum and "error" not in cum:
            print(f"    Cumul: SR={cum.get('sharpe_ratio', 0):.3f}, "
                  f"DD={cum.get('max_drawdown', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
