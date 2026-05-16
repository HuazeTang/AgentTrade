"""Persistent knowledge base for the agent system.

Stores factor discoveries, strategy performance, hypotheses, and failure records.
JSON-file backed with atomic writes. Supports the orchestrator's
"diagnose → hypothesize → test → evaluate → learn" loop.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Data records
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FactorRecord:
    """A discovered (or tracked) factor in the knowledge base.

    Status lifecycle: candidate → validated → active → decaying → retired
    """
    name: str
    category: str
    description: str = ""
    expression_repr: str = ""  # repr of the Expr tree
    source: str = "gp"  # "gp", "llm", "manual", "baseline"
    status: str = "candidate"  # candidate, validated, active, decaying, retired
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    hit_rate: float = 0.0
    auto_corr: float = 0.0
    max_corr_existing: float = 0.0
    created_at: str = ""
    validated_at: str = ""
    retired_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FactorRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyRecord:
    """A strategy configuration and its performance snapshot."""
    name: str
    description: str = ""
    factor_weights: dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    status: str = "active"  # active, paused, retired
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StrategyRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HypothesisRecord:
    """A hypothesis generated during the research loop and its outcome."""
    id: str
    description: str
    source: str = ""  # "llm", "gp", "manual"
    category: str = ""  # "factor_discovery", "parameter_tuning", "regime_adaptation"
    proposed_action: str = ""
    outcome: str = "pending"  # pending, accepted, rejected, inconclusive
    evidence: dict = field(default_factory=dict)
    created_at: str = ""
    resolved_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HypothesisRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FailureRecord:
    """Record of a failed approach to avoid repeating."""
    failure_type: str  # "factor_rejected", "strategy_failed", "parameter_bad"
    description: str
    context: dict = field(default_factory=dict)
    lesson: str = ""
    recorded_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FailureRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class IterationRecord:
    """One complete iteration of the agent loop."""
    iteration: int
    timestamp: str
    phase: str  # "diagnose", "hypothesize", "test", "evaluate", "learn"
    summary: str = ""
    metrics: dict = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> IterationRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """Persistent store for all agent knowledge.

    Loads from / saves to a JSON file. The in-memory cache is the source of truth
    during operation; flush() persists to disk.
    """

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else DEFAULT_KB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self.factors: dict[str, FactorRecord] = {}
        self.strategies: dict[str, StrategyRecord] = {}
        self.hypotheses: dict[str, HypothesisRecord] = {}
        self.failures: list[FailureRecord] = []
        self.iterations: list[IterationRecord] = []

        if self._path.exists():
            self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write current state to disk atomically."""
        data = {
            "factors": {k: v.to_dict() for k, v in self.factors.items()},
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            "failures": [f.to_dict() for f in self.failures],
            "iterations": [i.to_dict() for i in self.iterations],
        }
        # Atomic write
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=self._path.parent, prefix="kb_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, self._path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _load(self) -> None:
        """Load state from disk."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load knowledge base, starting fresh")
            return

        self.factors = {
            k: FactorRecord.from_dict(v) for k, v in data.get("factors", {}).items()
        }
        self.strategies = {
            k: StrategyRecord.from_dict(v) for k, v in data.get("strategies", {}).items()
        }
        self.hypotheses = {
            k: HypothesisRecord.from_dict(v) for k, v in data.get("hypotheses", {}).items()
        }
        self.failures = [
            FailureRecord.from_dict(f) for f in data.get("failures", [])
        ]
        self.iterations = [
            IterationRecord.from_dict(i) for i in data.get("iterations", [])
        ]

    # ── Factors ─────────────────────────────────────────────────────────────

    def add_factor(self, record: FactorRecord) -> None:
        record.created_at = record.created_at or _now()
        self.factors[record.name] = record

    def update_factor(self, name: str, **kwargs) -> None:
        if name in self.factors:
            for k, v in kwargs.items():
                if hasattr(self.factors[name], k):
                    setattr(self.factors[name], k, v)

    def get_factor(self, name: str) -> FactorRecord | None:
        return self.factors.get(name)

    def list_factors(self, status: str | None = None, category: str | None = None) -> list[FactorRecord]:
        result = list(self.factors.values())
        if status:
            result = [r for r in result if r.status == status]
        if category:
            result = [r for r in result if r.category == category]
        return result

    def get_active_factor_names(self) -> list[str]:
        return [r.name for r in self.factors.values() if r.status == "active"]

    def get_decaying_factors(self) -> list[FactorRecord]:
        return [r for r in self.factors.values() if r.status == "decaying"]

    # ── Strategies ──────────────────────────────────────────────────────────

    def add_strategy(self, record: StrategyRecord) -> None:
        record.updated_at = record.updated_at or _now()
        self.strategies[record.name] = record

    def update_strategy(self, name: str, **kwargs) -> None:
        if name in self.strategies:
            for k, v in kwargs.items():
                if hasattr(self.strategies[name], k):
                    setattr(self.strategies[name], k, v)
            self.strategies[name].updated_at = _now()

    def get_strategy(self, name: str) -> StrategyRecord | None:
        return self.strategies.get(name)

    def list_strategies(self, status: str | None = None) -> list[StrategyRecord]:
        result = list(self.strategies.values())
        if status:
            result = [r for r in result if r.status == status]
        return result

    # ── Hypotheses ──────────────────────────────────────────────────────────

    def add_hypothesis(self, record: HypothesisRecord) -> None:
        record.created_at = record.created_at or _now()
        self.hypotheses[record.id] = record

    def resolve_hypothesis(self, hyp_id: str, outcome: str, evidence: dict | None = None) -> None:
        if hyp_id in self.hypotheses:
            self.hypotheses[hyp_id].outcome = outcome
            self.hypotheses[hyp_id].resolved_at = _now()
            if evidence:
                self.hypotheses[hyp_id].evidence.update(evidence)

    def list_hypotheses(
        self, outcome: str | None = None, category: str | None = None
    ) -> list[HypothesisRecord]:
        result = list(self.hypotheses.values())
        if outcome:
            result = [r for r in result if r.outcome == outcome]
        if category:
            result = [r for r in result if r.category == category]
        return result

    def get_pending_hypotheses(self) -> list[HypothesisRecord]:
        return self.list_hypotheses(outcome="pending")

    # ── Failures ────────────────────────────────────────────────────────────

    def add_failure(self, record: FailureRecord) -> None:
        record.recorded_at = record.recorded_at or _now()
        self.failures.append(record)

    def list_failures(self, failure_type: str | None = None) -> list[FailureRecord]:
        if failure_type:
            return [f for f in self.failures if f.failure_type == failure_type]
        return list(self.failures)

    def is_known_failure(self, description: str, threshold: float = 0.7) -> bool:
        """Check if a similar failure has been recorded (simple substring match)."""
        desc_lower = description.lower()
        for f in self.failures:
            if desc_lower in f.description.lower() or f.description.lower() in desc_lower:
                return True
        return False

    # ── Iterations ──────────────────────────────────────────────────────────

    def add_iteration(self, record: IterationRecord) -> None:
        self.iterations.append(record)
        # Keep last 200 iterations max
        if len(self.iterations) > 200:
            self.iterations = self.iterations[-200:]

    def get_latest_iteration(self) -> IterationRecord | None:
        return self.iterations[-1] if self.iterations else None

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    # ── Statistics ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary statistics for reporting."""
        active_factors = self.list_factors(status="active")
        decaying_factors = self.list_factors(status="decaying")
        accepted_hypotheses = self.list_hypotheses(outcome="accepted")
        rejected_hypotheses = self.list_hypotheses(outcome="rejected")

        return {
            "total_factors": len(self.factors),
            "active_factors": len(active_factors),
            "decaying_factors": len(decaying_factors),
            "retired_factors": len(self.list_factors(status="retired")),
            "total_hypotheses": len(self.hypotheses),
            "accepted_hypotheses": len(accepted_hypotheses),
            "rejected_hypotheses": len(rejected_hypotheses),
            "pending_hypotheses": len(self.get_pending_hypotheses()),
            "total_failures": len(self.failures),
            "total_iterations": len(self.iterations),
            "acceptance_rate": (
                len(accepted_hypotheses) / max(len(self.hypotheses), 1)
            ),
            "active_factor_names": [r.name for r in active_factors],
            "decaying_factor_names": [r.name for r in decaying_factors],
        }

    # ── Export for LLM context ──────────────────────────────────────────────

    def context_for_llm(self) -> dict:
        """Generate a compact summary for LLM prompts."""
        return {
            "active_factors": [
                {"name": r.name, "category": r.category, "ic_ir": r.ic_ir, "status": r.status}
                for r in self.factors.values() if r.status in ("active", "validated")
            ],
            "decaying_factors": [
                {"name": r.name, "ic_ir": r.ic_ir}
                for r in self.get_decaying_factors()
            ],
            "recent_hypotheses": [
                {"id": h.id, "description": h.description, "outcome": h.outcome}
                for h in sorted(
                    self.hypotheses.values(),
                    key=lambda x: x.created_at,
                    reverse=True,
                )[:10]
            ],
            "recent_failures": [
                {"type": f.failure_type, "description": f.description, "lesson": f.lesson}
                for f in self.failures[-10:]
            ],
            "iteration_count": len(self.iterations),
        }


def _now() -> str:
    return datetime.now().isoformat()
