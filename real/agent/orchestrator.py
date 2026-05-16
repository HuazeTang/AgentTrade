"""Agent orchestrator — the central coordinator.

Implements the core research loop:
    diagnose → hypothesize → test → evaluate → learn

Coordinates GP engine, LLM client, causal strategies, monitor, and knowledge base.
Runs weekly research cycles (factor discovery, strategy tuning) and provides
daily execution hooks for the backtest engine.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from agent.decision import ExplorationPlanner, ExplorationPlan, ExplorationTarget
from agent.knowledge_base import (
    KnowledgeBase, FactorRecord, HypothesisRecord, FailureRecord,
    IterationRecord,
)
from agent.llm_client import LLMClient, create_default_client
from agent.monitor import Monitor, Diagnosis
from agent.report import ReportGenerator, IterationReport

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central agent coordinator.

    Usage:
        orch = Orchestrator(kb_path="data/kb.json")
        # Weekly research cycle
        report = orch.run_research_cycle(data, forward_returns)
        # Or: step through manually
        diagnosis = orch.diagnose(data, forward_returns)
        plan = orch.hypothesize(diagnosis)
        results = orch.test(plan, data, forward_returns)
        evaluation = orch.evaluate(results)
        orch.learn(evaluation)
    """

    def __init__(
        self,
        kb_path: str | None = None,
        llm_client: LLMClient | None = None,
        gp_population_size: int = 100,
        gp_generations: int = 20,
        max_new_factors_per_cycle: int = 10,
        verbose: bool = True,
    ):
        self.kb = KnowledgeBase(path=kb_path)
        self.llm = llm_client or create_default_client()
        self.monitor = Monitor()
        self.planner = ExplorationPlanner()
        self.reporter = ReportGenerator()

        # GP config (lightweight, can be overridden)
        from discovery.gp import GPEngine, GPConfig
        self.gp_config = GPConfig(
            population_size=gp_population_size,
            max_generations=gp_generations,
        )
        self.gp_engine = GPEngine(config=self.gp_config)

        self.max_new_factors = max_new_factors_per_cycle
        self.verbose = verbose

        # State
        self._iteration = self.kb.iteration_count
        self._last_diagnosis: Diagnosis | None = None
        self._last_plan: ExplorationPlan | None = None

    # ── Main research cycle ────────────────────────────────────────────────

    def run_research_cycle(
        self,
        data: pd.DataFrame,
        forward_returns: pd.Series,
        factor_values: pd.DataFrame | None = None,
        existing_factors: pd.DataFrame | None = None,
    ) -> IterationReport:
        """Execute one full diagnose→hypothesize→test→evaluate→learn cycle.

        Args:
            data: Multi-index OHLCV DataFrame.
            forward_returns: Same-index forward return Series.
            factor_values: Pre-computed factor values (computed if None).
            existing_factors: DataFrame of existing factor values for novelty check.

        Returns:
            IterationReport summarizing the cycle.
        """
        self._iteration += 1
        t0 = time.time()

        if self.verbose:
            logger.info("=" * 50)
            logger.info("Iteration %d: Starting research cycle", self._iteration)
            logger.info("=" * 50)

        # 1. DIAGNOSE
        diagnosis = self.diagnose(data, forward_returns, factor_values)
        if self.verbose:
            logger.info("Diagnosis: %s", diagnosis.summary)

        # 2. HYPOTHESIZE
        plan = self.hypothesize(diagnosis)
        if self.verbose:
            logger.info("Plan: %d targets, budget=%d", len(plan.targets), plan.budget)

        # 3. TEST
        test_results = self.test(plan, data, forward_returns, existing_factors)
        if self.verbose:
            logger.info("Test results: %d accepted, %d rejected",
                        len(test_results["accepted"]), len(test_results["rejected"]))

        # 4. EVALUATE
        evaluation = self.evaluate(test_results, diagnosis, plan)

        # 5. LEARN
        learnings = self.learn(test_results, evaluation, diagnosis)

        # Generate report
        report = self.reporter.make_iteration_report(
            iteration=self._iteration,
            diagnosis=diagnosis,
            plan=plan,
            test_results=test_results,
            accepted=test_results["accepted"],
            rejected=test_results["rejected"],
            learnings=learnings,
        )

        # Record iteration
        self.kb.add_iteration(IterationRecord(
            iteration=self._iteration,
            timestamp=datetime.now().isoformat(),
            phase="complete",
            summary=diagnosis.summary,
            metrics={
                "targets": len(plan.targets),
                "accepted": len(test_results["accepted"]),
                "rejected": len(test_results["rejected"]),
                "elapsed_s": time.time() - t0,
            },
            decisions=test_results["accepted"] + test_results["rejected"],
        ))
        self.kb.flush()

        if self.verbose:
            logger.info("Cycle complete in %.1fs", time.time() - t0)
            logger.info(report.format_text() if hasattr(report, 'format_text') else str(report))

        return report

    # ── Phase methods (callable individually) ──────────────────────────────

    def diagnose(
        self,
        data: pd.DataFrame,
        forward_returns: pd.Series,
        factor_values: pd.DataFrame | None = None,
    ) -> Diagnosis:
        """Phase 1: Diagnose factor health and market regime."""
        if factor_values is None:
            factor_values = self._compute_active_factors(data)

        diagnosis = self.monitor.diagnose(
            factor_values=factor_values,
            forward_returns=forward_returns,
            price_data=data,
        )
        self._last_diagnosis = diagnosis
        return diagnosis

    def hypothesize(self, diagnosis: Diagnosis | None = None) -> ExplorationPlan:
        """Phase 2: Generate hypotheses via exploration planner + optional LLM."""
        if diagnosis is None:
            diagnosis = self._last_diagnosis
        if diagnosis is None:
            return ExplorationPlan()

        kb_stats = self.kb.stats()
        active_factors = self.kb.get_active_factor_names()

        plan = self.planner.plan(diagnosis, kb_stats, active_factors)

        # Augment with LLM if available
        if self.llm.configured and plan.targets:
            try:
                llm_ideas = self.llm.generate_factor_ideas(
                    diagnosis=diagnosis.to_dict(),
                    existing_factors=active_factors,
                    n_ideas=3,
                )
                if llm_ideas:
                    # Add LLM-suggested ideas as additional context on top target
                    for idea in llm_ideas[:2]:
                        name = idea.get("name", "llm_idea")
                        intuition = idea.get("intuition", "")
                        if plan.top_priority:
                            plan.top_priority.suggested_actions.append(
                                f"LLM idea: {name} — {intuition}"
                            )
            except Exception as e:
                logger.debug("LLM hypothesis augmentation failed: %s", e)

        self._last_plan = plan
        return plan

    def test(
        self,
        plan: ExplorationPlan | None,
        data: pd.DataFrame,
        forward_returns: pd.Series,
        existing_factors: pd.DataFrame | None = None,
    ) -> dict:
        """Phase 3: Test hypotheses — run GP, validate discovered factors.

        Returns dict with keys: 'accepted', 'rejected', 'details'.
        """
        if plan is None:
            plan = self._last_plan
        if plan is None or not plan.targets:
            return {"accepted": [], "rejected": [], "details": {}}

        from discovery.validate import FactorValidator

        accepted: list[str] = []
        rejected: list[str] = []
        details: dict = {}

        validator = FactorValidator()

        # Run GP if we have high-urgency targets for factor discovery
        urgent_targets = [t for t in plan.targets if t.urgency > 0.3]
        if urgent_targets:
            try:
                self.gp_config.population_size = min(100, plan.budget * 10)
                self.gp_config.max_generations = max(10, plan.budget)

                best_individuals = self.gp_engine.evolve(
                    data=data,
                    forward_returns=forward_returns,
                    existing_factors=existing_factors,
                )

                details["gp_generations"] = self.gp_engine.generation
                details["gp_individuals"] = len(best_individuals)

                # Validate top individuals
                for ind in best_individuals[:plan.budget]:
                    if ind.factor_name.startswith("gp_") and ind.fitness > 0:
                        result = validator.validate(
                            factor_values=ind.factor_cls().compute(data) if ind.factor_cls else pd.Series(),
                            forward_returns=forward_returns,
                            factor_name=ind.factor_name,
                            existing_factors=existing_factors,
                        )
                        details[ind.factor_name] = {
                            "fitness": ind.fitness,
                            "ic_mean": result.ic_mean,
                            "ic_ir": result.ic_ir,
                            "passed": result.passed,
                        }
                        if result.passed:
                            accepted.append(ind.factor_name)
                        else:
                            rejected.append(ind.factor_name)
            except Exception as e:
                logger.error("GP evolution failed: %s", e)
                details["gp_error"] = str(e)

        # LLM factor generation for low-urgency exploration
        if self.llm.configured:
            explore_targets = [t for t in plan.targets if t.target_type == "random_explore"]
            if explore_targets and self.llm.configured:
                try:
                    llm_ideas = self.llm.generate_factor_ideas(
                        diagnosis=self._last_diagnosis.to_dict() if self._last_diagnosis else {},
                        existing_factors=self.kb.get_active_factor_names(),
                        n_ideas=2,
                    )
                    details["llm_ideas"] = len(llm_ideas) if isinstance(llm_ideas, list) else 0
                except Exception as e:
                    logger.debug("LLM factor generation failed: %s", e)

        return {"accepted": accepted, "rejected": rejected, "details": details}

    def evaluate(
        self,
        test_results: dict,
        diagnosis: Diagnosis | None = None,
        plan: ExplorationPlan | None = None,
    ) -> dict:
        """Phase 4: Evaluate test results against current state."""
        evaluation = {
            "new_factors_accepted": len(test_results.get("accepted", [])),
            "new_factors_rejected": len(test_results.get("rejected", [])),
            "acceptance_rate": 0.0,
            "action": "hold",
        }

        n_accepted = evaluation["new_factors_accepted"]
        n_total = n_accepted + evaluation["new_factors_rejected"]

        if n_total > 0:
            evaluation["acceptance_rate"] = n_accepted / n_total

        if n_accepted > 0:
            evaluation["action"] = "update"
            evaluation["message"] = f"Adopting {n_accepted} new factors"
        elif diagnosis and any(
            fh.status == "decaying" for fh in diagnosis.factors.values()
        ):
            evaluation["action"] = "retire"
            evaluation["message"] = "No viable new factors, retiring worst decaying factor"
        else:
            evaluation["action"] = "hold"
            evaluation["message"] = "No changes this cycle"

        return evaluation

    def learn(
        self,
        test_results: dict,
        evaluation: dict,
        diagnosis: Diagnosis | None = None,
    ) -> list[str]:
        """Phase 5: Update knowledge base with results."""
        learnings: list[str] = []

        # Accept new factors
        for name in test_results.get("accepted", []):
            detail = test_results.get("details", {}).get(name, {})
            self.kb.add_factor(FactorRecord(
                name=name,
                category="composite",
                description=f"GP-discovered factor (iteration {self._iteration})",
                source="gp",
                status="validated",
                ic_mean=detail.get("ic_mean", 0),
                ic_ir=detail.get("ic_ir", 0),
            ))
            learnings.append(f"Accepted new factor: {name}")

        # Record rejected factors as hypotheses (negative results)
        for name in test_results.get("rejected", []):
            detail = test_results.get("details", {}).get(name, {})
            hyp_id = f"gp_{self._iteration}_{name}"
            self.kb.add_hypothesis(HypothesisRecord(
                id=hyp_id,
                description=f"GP factor: {name}",
                source="gp",
                category="factor_discovery",
                outcome="rejected",
                evidence=detail,
            ))
            learnings.append(f"Rejected factor: {name} — recorded for avoidance")

        # Retire dead factors if needed
        if evaluation.get("action") == "retire" and diagnosis:
            for name, fh in diagnosis.factors.items():
                if fh.status == "dead":
                    self.kb.update_factor(name, status="retired")
                    learnings.append(f"Retired dead factor: {name}")
                    self.kb.add_failure(FailureRecord(
                        failure_type="factor_rejected",
                        description=f"Factor '{name}' died: IC={fh.ic_mean:.4f}, trend={fh.ic_trend:.4f}",
                        lesson=f"IC mean of {fh.ic_mean:.4f} with negative trend indicates regime dependency",
                    ))

        self.kb.flush()
        return learnings

    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_active_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all active factors from the knowledge base."""
        from factor.engine import FactorEngine
        from factor.registry import registry

        engine = FactorEngine()
        active_names = self.kb.get_active_factor_names()

        # Also include baseline factors if none active
        if not active_names:
            active_names = registry.list_all()[:5]  # first 5 baseline factors

        if not active_names:
            return pd.DataFrame()

        try:
            return engine.compute(active_names, data)
        except Exception as e:
            logger.warning("Factor computation failed: %s", e)
            return pd.DataFrame()

    def make_dashboard(self) -> str:
        """Generate a performance dashboard."""
        diagnosis = self._last_diagnosis
        if diagnosis is None:
            return "No diagnosis yet. Run diagnose() first."

        return self.reporter.make_performance_dashboard(
            kb_stats=self.kb.stats(),
            diagnosis=diagnosis,
        )

    @property
    def iteration(self) -> int:
        return self._iteration
