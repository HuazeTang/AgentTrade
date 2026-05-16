"""Exploration priority algorithm for the agent system.

Computes urgency-weighted priorities for what the agent should investigate
next: which factors to replace, which categories to explore, which strategies
to tune.

Algorithmic (no LLM dependency) — produces a ranked list consumed by the
orchestrator to decide where to allocate GP/LLM resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.monitor import Diagnosis, FactorHealth

# Factor categories in the system
ALL_CATEGORIES = [
    "momentum", "value", "quality", "volatility",
    "size", "liquidity", "growth", "composite",
]


@dataclass
class ExplorationTarget:
    """A suggested exploration target with urgency score."""
    target_type: str  # "replace_factor", "new_category", "tune_strategy", "random_explore"
    description: str
    urgency: float  # 0-1, higher = more urgent
    context: dict = field(default_factory=dict)
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class ExplorationPlan:
    """Ranked exploration priorities for the current cycle."""
    targets: list[ExplorationTarget] = field(default_factory=list)
    budget: int = 10  # max number of new factors to generate this cycle
    focus_categories: list[str] = field(default_factory=list)
    focus_terminals: list[str] = field(default_factory=list)

    @property
    def top_priority(self) -> ExplorationTarget | None:
        return self.targets[0] if self.targets else None


class ExplorationPlanner:
    """Computes exploration priorities from diagnosis and knowledge base state.

    Usage:
        planner = ExplorationPlanner()
        plan = planner.plan(diagnosis, knowledge_base_stats, active_factor_names)
    """

    def __init__(
        self,
        decay_urgency_weight: float = 0.8,
        uncovered_category_weight: float = 0.5,
        regime_change_weight: float = 0.7,
        strategy_fix_weight: float = 0.6,
        random_explore_weight: float = 0.2,
    ):
        self.decay_urgency_weight = decay_urgency_weight
        self.uncovered_category_weight = uncovered_category_weight
        self.regime_change_weight = regime_change_weight
        self.strategy_fix_weight = strategy_fix_weight
        self.random_explore_weight = random_explore_weight

    def plan(
        self,
        diagnosis: Diagnosis,
        kb_stats: dict,
        active_factor_names: list[str] | None = None,
        max_targets: int = 5,
    ) -> ExplorationPlan:
        """Generate ranked exploration priorities.

        Args:
            diagnosis: Current system diagnosis from Monitor.
            kb_stats: Knowledge base stats (from KnowledgeBase.stats()).
            active_factor_names: Currently active factors.
            max_targets: Max number of targets to return.

        Returns:
            ExplorationPlan with ranked targets.
        """
        targets: list[ExplorationTarget] = []
        active = active_factor_names or kb_stats.get("active_factor_names", [])

        # 1. Factor decay urgency
        decaying_factors = [
            (name, fh) for name, fh in diagnosis.factors.items()
            if fh.status in ("decaying", "dead")
        ]
        for name, fh in decaying_factors:
            # Urgency proportional to how much IC has dropped
            ic_drop = abs(fh.ic_mean - fh.recent_ic_mean) / max(abs(fh.ic_mean), 1e-6)
            urgency = min(self.decay_urgency_weight + ic_drop * 0.2, 0.99)

            targets.append(ExplorationTarget(
                target_type="replace_factor",
                description=f"Replace decaying factor '{name}' (IC: {fh.ic_mean:.4f}→{fh.recent_ic_mean:.4f})",
                urgency=urgency,
                context={
                    "factor_name": name,
                    "ic_mean": fh.ic_mean,
                    "recent_ic_mean": fh.recent_ic_mean,
                    "ic_trend": fh.ic_trend,
                },
                suggested_actions=[
                    f"Run GP with similar terminal set to {name}",
                    f"Ask LLM for alternatives to {name}",
                    f"Check if {name} works in specific sectors only",
                ],
            ))

        # 2. Uncovered factor categories
        covered_categories = self._infer_covered_categories(active, diagnosis)
        uncovered = set(ALL_CATEGORIES) - covered_categories
        if uncovered:
            urgency = self.uncovered_category_weight
            targets.append(ExplorationTarget(
                target_type="new_category",
                description=f"Explore uncovered categories: {', '.join(sorted(uncovered))}",
                urgency=urgency,
                context={"uncovered_categories": list(uncovered)},
                suggested_actions=[
                    f"Generate factors in: {', '.join(sorted(uncovered))}",
                    "Use LLM to suggest economic rationales for new categories",
                    "Run GP with terminals biased to uncovered category patterns",
                ],
            ))

        # 3. Regime change response
        if diagnosis.regime != "normal" and diagnosis.regime_confidence > 0.6:
            urgency = self.regime_change_weight * diagnosis.regime_confidence
            targets.append(ExplorationTarget(
                target_type="tune_strategy",
                description=f"Adapt to regime change: {diagnosis.regime} (confidence={diagnosis.regime_confidence:.2f})",
                urgency=urgency,
                context={
                    "regime": diagnosis.regime,
                    "confidence": diagnosis.regime_confidence,
                },
                suggested_actions=[
                    "Adjust factor weights for current regime",
                    "Run causal analysis to identify regime-specific factors",
                    "Consider regime-switching model for factor blending",
                ],
            ))

        # 4. Low-performing strategy repair
        if kb_stats.get("active_factors", 0) < 3:
            urgency = self.strategy_fix_weight
            targets.append(ExplorationTarget(
                target_type="tune_strategy",
                description=f"Too few active factors ({kb_stats.get('active_factors', 0)}), need more diversity",
                urgency=urgency,
                context={"active_count": kb_stats.get("active_factors", 0)},
                suggested_actions=[
                    "Run GP with broader terminal set",
                    "Lower acceptance threshold temporarily",
                    "Import baseline factors from knowledge base",
                ],
            ))

        # 5. Random exploration (always present, low weight)
        targets.append(ExplorationTarget(
            target_type="random_explore",
            description="Random exploration of novel factor structures",
            urgency=self.random_explore_weight,
            context={},
            suggested_actions=[
                "Generate random expression trees with no bias",
                "Try unusual operator combinations",
                "Explore non-standard window sizes",
            ],
        ))

        # Sort by urgency descending
        targets.sort(key=lambda t: t.urgency, reverse=True)

        # Compute focus categories and terminals
        focus_categories = list(set(
            cat for t in targets if t.target_type == "new_category"
            for cat in t.context.get("uncovered_categories", [])
        ))
        focus_terminals = self._suggest_terminals(targets, diagnosis)

        return ExplorationPlan(
            targets=targets[:max_targets],
            budget=self._compute_budget(diagnosis, targets),
            focus_categories=focus_categories,
            focus_terminals=focus_terminals,
        )

    def _infer_covered_categories(
        self,
        active_factors: list[str],
        diagnosis: Diagnosis,
    ) -> set[str]:
        """Infer which categories are covered by active and healthy factors."""
        covered = set()
        for name in active_factors:
            # Try to infer category from factor name
            for cat in ALL_CATEGORIES:
                if cat in name.lower():
                    covered.add(cat)
                    break

        # Also include categories of healthy factors
        for name, fh in diagnosis.factors.items():
            if fh.status == "healthy":
                for cat in ALL_CATEGORIES:
                    if cat in name.lower():
                        covered.add(cat)
        return covered

    def _suggest_terminals(
        self,
        targets: list[ExplorationTarget],
        diagnosis: Diagnosis,
    ) -> list[str]:
        """Suggest terminal fields based on exploration targets."""
        terminals = set(["close", "volume"])  # always include basics

        for t in targets:
            if t.target_type == "new_category":
                for cat in t.context.get("uncovered_categories", []):
                    if cat == "value":
                        terminals.update(["close", "amount"])
                    elif cat == "quality":
                        terminals.update(["close", "amount", "turnover"])
                    elif cat == "liquidity":
                        terminals.update(["volume", "amount", "turnover"])
                    elif cat == "growth":
                        terminals.update(["close", "amount"])
                    elif cat == "volatility":
                        terminals.update(["high", "low", "close"])

        return sorted(terminals)

    def _compute_budget(
        self,
        diagnosis: Diagnosis,
        targets: list[ExplorationTarget],
    ) -> int:
        """How many new factors to generate this cycle."""
        decaying_count = sum(
            1 for fh in diagnosis.factors.values() if fh.status == "decaying"
        )
        dead_count = sum(
            1 for fh in diagnosis.factors.values() if fh.status == "dead"
        )

        # More decay → more budget
        base = 5
        base += decaying_count * 2
        base += dead_count * 1
        return min(base, 30)
