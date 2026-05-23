"""Parameter definitions and search configuration."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParameterSpec:
    """Definition of one tunable parameter."""
    name: str
    type: str = "float"            # "float" | "int" | "choice"
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    default: Any = None
    log_scale: bool = False        # sample in log-space for random search

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw a random value within the parameter's range."""
        if self.type == "choice":
            return rng.choice(self.choices)
        if self.type == "int":
            return int(rng.integers(self.low, self.high + 1))
        if self.log_scale and self.low > 0:
            lo, hi = np.log(self.low), np.log(self.high)
            return float(np.exp(rng.uniform(lo, hi)))
        return float(rng.uniform(self.low, self.high))

    def grid_values(self, n_points: int = 5) -> list[Any]:
        """Generate evenly-spaced grid points."""
        if self.type == "choice":
            return list(self.choices)
        if self.type == "int":
            step = max(1, (self.high - self.low) // (n_points - 1))
            return list(range(self.low, self.high + 1, step))
        if self.log_scale and self.low > 0:
            pts = np.linspace(np.log(self.low), np.log(self.high), n_points)
            return [round(float(np.exp(p)), 4) for p in pts]
        return [round(float(v), 4) for v in np.linspace(self.low, self.high, n_points)]


@dataclass
class SearchConfig:
    """Complete parameter search configuration."""
    parameters: list[ParameterSpec]
    strategy: str = "random"        # "random" | "grid" | "sequential"
    n_iterations: int = 500
    n_workers: int = 4
    random_seed: int = 42
    metric: str = "sharpe_ratio"    # primary optimization metric
    output_path: Path = field(
        default_factory=lambda: Path("data/results/param_search.json")
    )
    trading_start: date | None = None
    trading_end: date | None = None
    initial_cash: float | None = None
    resume: bool = False


# ── Full parameter catalog with wide search ranges ──

ALL_TUNABLE_PARAMS: list[ParameterSpec] = [
    # Exit rules
    ParameterSpec("TAKE_PROFIT_PCT",       "float",  0.10, 0.50,  default=0.25),
    ParameterSpec("STOP_LOSS_VOL_MULT",    "float",  0.5,  4.0,  default=2.0),
    ParameterSpec("TRAIL_STOP_FROM_PEAK",  "float",  0.05, 0.35, default=0.15),
    ParameterSpec("MIN_STOP_LOSS_PCT",     "float",  0.01, 0.12, default=0.05),
    ParameterSpec("MAX_STOP_LOSS_PCT",     "float",  0.10, 0.30, default=0.15),
    ParameterSpec("CONSECUTIVE_DOWN_EXIT", "int",    2,    6,     default=3),

    # Position sizing (major structural impact)
    ParameterSpec("MAX_POSITIONS",         "int",    1,    5,     default=1),
    ParameterSpec("SELL_RANK_LIMIT",       "int",    2,    12,    default=5),
    ParameterSpec("MAX_POSITION_PCT",      "float",  0.5,  1.0,   default=0.95),

    # Market regime
    ParameterSpec("REGIME_DECLINE_PCT",    "float",  -0.10, -0.01, default=-0.03),
    ParameterSpec("REGIME_CHOPPY_PCT",     "float",  0.01,  0.10,  default=0.03),

    # Entry quality gates
    ParameterSpec("MIN_COMPOSITE_SCORE",    "float",  0.0,   0.5,   default=0.0),
    ParameterSpec("REQUIRE_TREND_ABOVE_MA", "choice", choices=[True, False], default=False),
    ParameterSpec("MIN_VOLUME_RATIO",       "float",  0.0,   1.5,   default=0.0),
    ParameterSpec("RECOVERY_DAYS",          "int",    0,     5,     default=0),
]

DEFAULT_PARAMS: dict[str, Any] = {p.name: p.default for p in ALL_TUNABLE_PARAMS}
