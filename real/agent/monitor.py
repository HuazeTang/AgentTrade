"""Performance monitor for the agent system.

Tracks factor health (IC decay, autocorrelation drift), detects regime changes,
and flags anomalies. Produces a diagnosis dict consumed by the orchestrator
and LLM for hypothesis generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from factor.validation import compute_rank_ic, ic_summary, factor_auto_correlation

logger = logging.getLogger(__name__)


@dataclass
class FactorHealth:
    """Health snapshot of a single factor."""
    name: str
    ic_mean: float = np.nan
    ic_std: float = np.nan
    ic_ir: float = np.nan
    hit_rate: float = np.nan
    auto_corr: float = np.nan
    recent_ic_mean: float = np.nan  # last 60 days
    ic_trend: float = np.nan  # slope of IC over time
    status: str = "unknown"  # healthy, decaying, dead, improving
    warnings: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    """Full system diagnosis produced by the monitor."""
    timestamp: str
    factors: dict[str, FactorHealth] = field(default_factory=dict)
    regime: str = "normal"  # normal, high_vol, low_vol, rotating, trending
    regime_confidence: float = 0.0
    anomalies: list[str] = field(default_factory=list)
    summary: str = ""
    actionable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "anomalies": self.anomalies,
            "summary": self.summary,
            "actionable": self.actionable,
            "factors": {
                name: {
                    "ic_mean": fh.ic_mean,
                    "ic_ir": fh.ic_ir,
                    "hit_rate": fh.hit_rate,
                    "auto_corr": fh.auto_corr,
                    "recent_ic_mean": fh.recent_ic_mean,
                    "ic_trend": fh.ic_trend,
                    "status": fh.status,
                    "warnings": fh.warnings,
                }
                for name, fh in self.factors.items()
            },
        }


class Monitor:
    """Monitors factor performance, detects regime shifts and anomalies.

    Usage:
        monitor = Monitor(lookback_recent=60, decay_threshold=0.3)
        diagnosis = monitor.diagnose(
            factor_values=df_of_factors,
            forward_returns=fwd_ret_series,
            price_data=ohlcv_data,
        )
    """

    def __init__(
        self,
        lookback_recent: int = 60,
        decay_threshold: float = 0.3,
        auto_corr_warn: float = 0.95,
        ic_min_abs: float = 0.01,
        vol_regime_threshold: float = 1.5,  # multiplier vs historical vol
    ):
        self.lookback_recent = lookback_recent
        self.decay_threshold = decay_threshold
        self.auto_corr_warn = auto_corr_warn
        self.ic_min_abs = ic_min_abs
        self.vol_regime_threshold = vol_regime_threshold

        # Track history for trend detection
        self._ic_history: dict[str, pd.Series] = {}
        self._vol_history: pd.Series | None = None

    def diagnose(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.Series,
        price_data: pd.DataFrame | None = None,
    ) -> Diagnosis:
        """Run full diagnosis on current factor and market state.

        Args:
            factor_values: DataFrame with factor columns, multi-index (trade_date, symbol).
            forward_returns: Series with same index, forward returns.
            price_data: Optional OHLCV for volatility regime detection.

        Returns:
            Diagnosis with factor health, regime, and actionable items.
        """
        from datetime import datetime

        diagnosis = Diagnosis(timestamp=datetime.now().isoformat())
        anomalies: list[str] = []

        # 1. Factor health checks
        for col in factor_values.columns:
            health = self._check_factor(col, factor_values[col], forward_returns)
            diagnosis.factors[col] = health
            anomalies.extend(health.warnings)

        # 2. Regime detection
        if price_data is not None and "close" in price_data.columns:
            diagnosis.regime, diagnosis.regime_confidence = self._detect_regime(price_data)

        # 3. Anomaly synthesis
        diagnosis.anomalies = list(set(anomalies))

        # 4. Summary and actionable items
        diagnosis.summary, diagnosis.actionable = self._summarize(diagnosis)

        return diagnosis

    def _check_factor(
        self,
        name: str,
        factor_series: pd.Series,
        forward_returns: pd.Series,
    ) -> FactorHealth:
        """Evaluate health of a single factor."""
        health = FactorHealth(name=name)
        warnings: list[str] = []

        ic = compute_rank_ic(factor_series, forward_returns)
        if ic.empty:
            health.status = "unknown"
            health.warnings = ["No IC data"]
            return health

        summary = ic_summary(ic)
        health.ic_mean = summary["mean"]
        health.ic_std = summary["std"]
        health.ic_ir = summary["ir"]
        health.hit_rate = summary["hit_rate"]

        # Recent IC (last N days)
        recent_cutoff = ic.index[-1] - pd.Timedelta(days=self.lookback_recent * 2)
        recent_ic = ic[ic.index >= recent_cutoff]
        if len(recent_ic) > 0:
            health.recent_ic_mean = float(recent_ic.mean())

        # IC trend
        if len(ic) >= 20:
            x = np.arange(len(ic))
            y = ic.values
            mask = np.isfinite(y)
            if mask.sum() >= 10:
                slope = np.polyfit(x[mask], y[mask], 1)[0]
                health.ic_trend = float(slope)

        # Autocorrelation
        health.auto_corr = factor_auto_correlation(factor_series)

        # Status classification
        if abs(health.ic_mean) < self.ic_min_abs:
            health.status = "dead"
            warnings.append(f"IC too low: |{health.ic_mean:.4f}| < {self.ic_min_abs}")
        elif (abs(health.recent_ic_mean) < self.ic_min_abs and
              abs(health.ic_mean) > self.ic_min_abs):
            health.status = "decaying"
            decay_pct = (abs(health.recent_ic_mean) - abs(health.ic_mean)) / max(abs(health.ic_mean), 1e-6)
            warnings.append(f"IC decay: recent IC={health.recent_ic_mean:.4f} vs overall={health.ic_mean:.4f}")
        elif health.ic_trend < -0.0001:
            health.status = "decaying"
            warnings.append(f"Negative IC trend: slope={health.ic_trend:.6f}")
        elif abs(health.ic_ir) > 0.3 and abs(health.recent_ic_mean) > self.ic_min_abs:
            health.status = "healthy"
        elif abs(health.ic_mean) > self.ic_min_abs:
            health.status = "healthy"
        else:
            health.status = "weak"

        if not np.isnan(health.auto_corr) and health.auto_corr > self.auto_corr_warn:
            warnings.append(f"High autocorrelation: {health.auto_corr:.3f} > {self.auto_corr_warn}")
            if health.status == "healthy":
                health.status = "decaying"

        health.warnings = warnings
        return health

    def _detect_regime(
        self,
        price_data: pd.DataFrame,
    ) -> tuple[str, float]:
        """Classify current market regime from price data."""
        if "close" not in price_data.columns:
            return "normal", 0.0

        close = price_data["close"]
        if isinstance(close.index, pd.MultiIndex):
            if "trade_date" in close.index.names:
                # Use cross-sectional mean return for market proxy
                ret = close.unstack().pct_change().mean(axis=1)
            else:
                return "normal", 0.0
        else:
            ret = close.pct_change()

        ret = ret.dropna()
        if len(ret) < 20:
            return "normal", 0.0

        recent_vol = ret.iloc[-20:].std()
        hist_vol = ret.iloc[:-20].std() if len(ret) > 40 else recent_vol

        vol_ratio = recent_vol / max(hist_vol, 1e-6)

        # Trend detection
        recent_trend = ret.iloc[-20:].mean()
        hist_trend = ret.iloc[:-20].mean() if len(ret) > 40 else 0

        if vol_ratio > self.vol_regime_threshold:
            return "high_vol", min(vol_ratio / 3, 1.0)
        elif vol_ratio < (1 / self.vol_regime_threshold):
            return "low_vol", min(3 / max(vol_ratio, 0.1), 1.0)
        elif abs(recent_trend) > 2 * abs(hist_trend):
            return "trending", 0.7

        return "normal", 0.5

    def _summarize(self, diagnosis: Diagnosis) -> tuple[str, list[str]]:
        """Generate human-readable summary and actionable items."""
        parts = []
        actions = []

        decaying = [n for n, h in diagnosis.factors.items() if h.status == "decaying"]
        dead = [n for n, h in diagnosis.factors.items() if h.status == "dead"]
        healthy = [n for n, h in diagnosis.factors.items() if h.status == "healthy"]

        parts.append(f"Regime: {diagnosis.regime}")
        parts.append(f"Factors: {len(healthy)} healthy, {len(decaying)} decaying, {len(dead)} dead")

        if decaying:
            parts.append(f"Decaying: {', '.join(decaying)}")
            actions.append(f"Replace or retrain decaying factors: {decaying}")

        if dead:
            parts.append(f"Dead: {', '.join(dead)}")
            actions.append(f"Retire dead factors: {dead}")

        if diagnosis.regime == "high_vol":
            actions.append("Reduce position sizes in high volatility regime")
            actions.append("Widen stop-loss thresholds")
        elif diagnosis.regime == "trending":
            actions.append("Increase momentum factor weights in trending market")
            actions.append("Reduce mean-reversion exposure")

        if diagnosis.anomalies:
            parts.append(f"Anomalies: {len(diagnosis.anomalies)} detected")
            actions.append(f"Investigate anomalies: {diagnosis.anomalies[:3]}")

        if not actions:
            actions.append("No urgent actions — maintain current allocation")

        return "\n".join(parts), actions


def compute_ic_history(
    factor_values: pd.DataFrame,
    forward_returns: pd.Series,
) -> dict[str, pd.Series]:
    """Compute IC history for all factors at once."""
    history = {}
    for col in factor_values.columns:
        ic = compute_rank_ic(factor_values[col], forward_returns)
        if not ic.empty:
            history[col] = ic
    return history
