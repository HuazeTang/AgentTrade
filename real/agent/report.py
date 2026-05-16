"""Iteration reports for the agent system.

Generates text reports summarizing each diagnose→hypothesize→test→evaluate→learn
cycle. Used for logging, debugging, and human review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.decision import ExplorationPlan, ExplorationTarget
from agent.monitor import Diagnosis, FactorHealth


@dataclass
class IterationReport:
    """Report for one complete agent iteration."""
    iteration: int
    timestamp: str = ""
    phase_results: dict[str, str] = field(default_factory=dict)

    # Phase outputs
    diagnosis_summary: str = ""
    hypotheses: list[str] = field(default_factory=list)
    test_results: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    learnings: list[str] = field(default_factory=list)

    # Status
    new_factors_accepted: list[str] = field(default_factory=list)
    new_factors_rejected: list[str] = field(default_factory=list)
    factors_retired: list[str] = field(default_factory=list)
    strategies_adjusted: list[str] = field(default_factory=list)

    performance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "diagnosis_summary": self.diagnosis_summary,
            "hypotheses": self.hypotheses,
            "test_results": self.test_results,
            "evaluation": self.evaluation,
            "learnings": self.learnings,
            "new_factors_accepted": self.new_factors_accepted,
            "new_factors_rejected": self.new_factors_rejected,
            "factors_retired": self.factors_retired,
            "strategies_adjusted": self.strategies_adjusted,
            "performance": self.performance,
        }

    def format_text(self) -> str:
        """Format as human-readable text report."""
        sep = "=" * 60
        lines = [
            sep,
            f"AGENT ITERATION #{self.iteration} — {self.timestamp}",
            sep,
            "",
            "── DIAGNOSIS ──",
            self.diagnosis_summary or "(none)",
            "",
            "── HYPOTHESES ──",
        ]
        for i, h in enumerate(self.hypotheses, 1):
            lines.append(f"  {i}. {h}")
        if not self.hypotheses:
            lines.append("  (none)")

        lines.extend([
            "",
            "── TEST RESULTS ──",
        ])
        for k, v in self.test_results.items():
            lines.append(f"  {k}: {v}")
        if not self.test_results:
            lines.append("  (none)")

        lines.extend([
            "",
            "── EVALUATION ──",
        ])
        for k, v in self.evaluation.items():
            lines.append(f"  {k}: {v}")

        lines.extend([
            "",
            "── LEARNINGS ──",
        ])
        for l in self.learnings:
            lines.append(f"  • {l}")

        lines.extend([
            "",
            "── DECISIONS ──",
            f"  Accepted factors: {', '.join(self.new_factors_accepted) if self.new_factors_accepted else '(none)'}",
            f"  Rejected factors: {', '.join(self.new_factors_rejected) if self.new_factors_rejected else '(none)'}",
            f"  Retired factors: {', '.join(self.factors_retired) if self.factors_retired else '(none)'}",
            f"  Adjusted strategies: {', '.join(self.strategies_adjusted) if self.strategies_adjusted else '(none)'}",
            "",
            sep,
        ])

        return "\n".join(lines)


class ReportGenerator:
    """Generates iteration, factor discovery, and performance reports."""

    def make_iteration_report(
        self,
        iteration: int,
        diagnosis: Diagnosis,
        plan: ExplorationPlan,
        test_results: dict,
        accepted: list[str],
        rejected: list[str],
        learnings: list[str],
    ) -> IterationReport:
        """Create a full iteration report."""
        report = IterationReport(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            diagnosis_summary=diagnosis.summary,
            hypotheses=[t.description for t in plan.targets],
            test_results=test_results,
            evaluation={
                "targets_planned": len(plan.targets),
                "factors_tested": len(accepted) + len(rejected),
                "acceptance_rate": len(accepted) / max(len(accepted) + len(rejected), 1),
                "budget_used": plan.budget,
            },
            learnings=learnings,
            new_factors_accepted=accepted,
            new_factors_rejected=rejected,
        )
        return report

    def make_factor_report(
        self,
        factor_name: str,
        ic_mean: float,
        ic_ir: float,
        hit_rate: float,
        auto_corr: float,
        expression: str,
        source: str,
        status: str,
    ) -> str:
        """Generate a single-factor discovery report."""
        return f"""Factor Discovery Report: {factor_name}
{'─' * 40}
Source: {source}
Status: {status}

Performance:
  IC Mean:  {ic_mean:.6f}
  IC IR:    {ic_ir:.4f}
  Hit Rate: {hit_rate:.3f}
  AutoCorr: {auto_corr:.4f}

Expression:
  {expression}
"""

    def make_performance_dashboard(
        self,
        kb_stats: dict,
        diagnosis: Diagnosis,
    ) -> str:
        """Generate a text-based performance dashboard."""
        sep = "─" * 50
        lines = [
            f"PERFORMANCE DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            sep,
            "",
            "Factor Health:",
        ]

        for name, fh in diagnosis.factors.items():
            status_icon = {
                "healthy": "✓",
                "decaying": "⚠",
                "dead": "✗",
                "weak": "○",
                "unknown": "?",
            }.get(fh.status, "?")
            lines.append(
                f"  {status_icon} {name:20s} IC={fh.ic_mean:+.4f}  "
                f"IR={fh.ic_ir:.3f}  trend={fh.ic_trend:+.4f}  [{fh.status}]"
            )

        lines.extend([
            "",
            f"Regime: {diagnosis.regime} (conf={diagnosis.regime_confidence:.2f})",
            "",
            "Knowledge Base:",
            f"  Total factors: {kb_stats.get('total_factors', 0)}",
            f"  Active: {kb_stats.get('active_factors', 0)}",
            f"  Decaying: {kb_stats.get('decaying_factors', 0)}",
            f"  Retired: {kb_stats.get('retired_factors', 0)}",
            f"  Hypotheses: {kb_stats.get('total_hypotheses', 0)} "
            f"(acceptance: {kb_stats.get('acceptance_rate', 0):.0%})",
            f"  Iterations: {kb_stats.get('total_iterations', 0)}",
            "",
        ])

        if diagnosis.anomalies:
            lines.extend(["Anomalies:", sep])
            for a in diagnosis.anomalies:
                lines.append(f"  • {a}")
            lines.append("")

        if diagnosis.actionable:
            lines.extend(["Recommended Actions:", sep])
            for a in diagnosis.actionable:
                lines.append(f"  → {a}")

        return "\n".join(lines)
