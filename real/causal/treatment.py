"""Treatment definitions for causal inference in factor investing.

A Treatment identifies which stocks are "treated" at each point in time.
This is the T in the potential outcomes framework Y(T).

Three treatment types:
- DiscreteTreatment: binary event (index inclusion, policy change)
- ContinuousTreatment: dose-response (institutional flow magnitude)
- ThresholdTreatment: derived from data (factor > threshold = treated)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Treatment(ABC):
    """Abstract base for treatment definitions.

    A Treatment maps a MultiIndex DataFrame (trade_date, symbol) to
    treatment assignments T ∈ {0,1} and optionally propensity scores p(T=1|X).
    """

    @abstractmethod
    def compute_treatment(
        self, data: pd.DataFrame
    ) -> pd.Series:
        """Compute treatment assignment T ∈ {0,1}.

        Args:
            data: Multi-indexed (trade_date, symbol) DataFrame.

        Returns:
            Series with same index, values 0 or 1.
        """
        ...

    def compute_propensity(
        self, data: pd.DataFrame
    ) -> pd.Series | None:
        """Compute propensity score p(T=1|X). Default: uniform.

        Override when treatment probability is known (e.g., from a model).
        """
        t = self.compute_treatment(data)
        return pd.Series(0.5, index=t.index, name="propensity")

    @abstractmethod
    def required_fields(self) -> list[str]:
        """Raw data fields needed to compute treatment."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this treatment."""
        ...

    @property
    def is_binary(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Discrete treatment (events)
# ═══════════════════════════════════════════════════════════════════════════════

class DiscreteTreatment(Treatment):
    """Treatment defined by a pre-computed event indicator column.

    The data must contain a column `event_col` with 1 for treated, 0 for control.
    Common use: index inclusion/exclusion, policy announcements, ST/*ST transitions.
    """

    def __init__(self, name: str, event_col: str, treatment_window: int = 1):
        self._name = name
        self.event_col = event_col
        self.treatment_window = treatment_window

    @property
    def name(self) -> str:
        return self._name

    def compute_treatment(self, data: pd.DataFrame) -> pd.Series:
        if self.event_col not in data.columns:
            return pd.Series(0, index=data.index, name="T")

        event = data[self.event_col].fillna(0)
        if self.treatment_window <= 1:
            return (event > 0).astype(int).rename("T")

        # Extend treatment over window using forward fill within each symbol
        t = (event > 0).astype(int).unstack()
        for _ in range(self.treatment_window - 1):
            t = t | t.shift(-1).fillna(0).astype(int)
        return t.stack().rename("T")

    def required_fields(self) -> list[str]:
        return [self.event_col]


# ═══════════════════════════════════════════════════════════════════════════════
# Continuous treatment (dose-response)
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousTreatment(Treatment):
    """Treatment defined by a continuous exposure variable.

    Useful for: institutional flow magnitude, earnings surprise size,
    factor loading intensity.

    The treatment is binarized at the cross-sectional median (or given quantile)
    for binary causal methods, or used directly for dose-response (DML with
    continuous T).
    """

    def __init__(
        self,
        name: str,
        exposure_col: str,
        threshold_quantile: float = 0.5,
    ):
        self._name = name
        self.exposure_col = exposure_col
        self.threshold_quantile = threshold_quantile

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_binary(self) -> bool:
        return False

    def compute_treatment(self, data: pd.DataFrame) -> pd.Series:
        """Binarize at cross-sectional quantile."""
        if self.exposure_col not in data.columns:
            return pd.Series(0, index=data.index, name="T")

        exposure = data[self.exposure_col]
        threshold = exposure.groupby("trade_date").transform(
            lambda x: x.quantile(self.threshold_quantile)
        )
        return (exposure > threshold).astype(int).rename("T")

    def compute_dose(self, data: pd.DataFrame) -> pd.Series:
        """Return continuous exposure (for dose-response models)."""
        if self.exposure_col not in data.columns:
            return pd.Series(0.0, index=data.index, name="dose")
        return data[self.exposure_col].rename("dose")

    def required_fields(self) -> list[str]:
        return [self.exposure_col]


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold treatment (derived from data)
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdTreatment(Treatment):
    """Treatment triggered when a variable crosses a fixed threshold.

    Example: RSI < 30 = treated (oversold), volume > 3x avg = treated (surge).

    Can be asymmetric: above_threshold for long-side treatment,
    below_threshold for short-side treatment.
    """

    def __init__(
        self,
        name: str,
        field: str,
        threshold: float,
        direction: str = "above",
        lookback: int = 1,
    ):
        self._name = name
        self.field = field
        self.threshold = threshold
        self.direction = direction  # "above" or "below"
        self.lookback = lookback

    @property
    def name(self) -> str:
        return self._name

    def compute_treatment(self, data: pd.DataFrame) -> pd.Series:
        if self.field not in data.columns:
            return pd.Series(0, index=data.index, name="T")

        if self.lookback > 1:
            # Compare to rolling mean
            val = data[self.field].unstack()
            baseline = val.rolling(self.lookback, min_periods=max(1, self.lookback // 2)).mean()
            diff = (val - baseline).stack()
        else:
            diff = data[self.field]

        if self.direction == "above":
            return (diff > self.threshold).astype(int).rename("T")
        else:
            return (diff < self.threshold).astype(int).rename("T")

    def required_fields(self) -> list[str]:
        return [self.field]


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def treatment_balance_check(
    treatment: pd.Series,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Check covariate balance between treated and control groups.

    Returns DataFrame with columns: covariate, treated_mean, control_mean,
    std_diff (standardized difference), p_value (t-test).
    """
    from scipy import stats

    t = treatment.reindex(covariates.index)
    treated = covariates[t == 1]
    control = covariates[t == 0]

    results = []
    for col in covariates.columns:
        t_mean = treated[col].mean()
        c_mean = control[col].mean()
        t_std = treated[col].std()
        c_std = control[col].std()

        pooled_std = np.sqrt((t_std ** 2 + c_std ** 2) / 2)
        std_diff = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0

        try:
            _, p_val = stats.ttest_ind(
                treated[col].dropna(), control[col].dropna()
            )
        except Exception:
            p_val = np.nan

        results.append({
            "covariate": col,
            "treated_mean": t_mean,
            "control_mean": c_mean,
            "std_diff": std_diff,
            "p_value": p_val,
        })

    return pd.DataFrame(results).set_index("covariate")
