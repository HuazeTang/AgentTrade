"""Price momentum factors."""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class Momentum1M(Factor):
    meta = FactorMeta(
        name="momentum_1m",
        category="momentum",
        description="21-day price momentum",
        lookback_days=21,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        momentum = close.pct_change(periods=21)
        return momentum.stack().rename("momentum_1m").sort_index()


@register_factor
class Momentum3M(Factor):
    meta = FactorMeta(
        name="momentum_3m",
        category="momentum",
        description="63-day price momentum",
        lookback_days=63,
    )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        momentum = close.pct_change(periods=63)
        return momentum.stack().sort_index()


@register_factor
class Momentum6M(Factor):
    meta = FactorMeta(
        name="momentum_6m",
        category="momentum",
        description="126-day price momentum",
        lookback_days=126,
    )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        momentum = close.pct_change(periods=126)
        return momentum.stack().sort_index()


@register_factor
class Momentum12M1M(Factor):
    meta = FactorMeta(
        name="momentum_12m1m",
        category="momentum",
        description="12-month minus 1-month momentum (skip recent month)",
        lookback_days=252,
    )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        mom_12 = close.pct_change(periods=252)
        mom_1 = close.pct_change(periods=21)
        result = mom_12 - mom_1
        return result.stack().sort_index()


@register_factor
class MomentumAccel20D(Factor):
    """Price acceleration — second derivative of price trend.

    Computes momentum of momentum: how fast is the 20-day return changing
    vs 20 days ago. Positive = trend accelerating, negative = decelerating.
    Captures inflection points before they show in price level.
    """

    meta = FactorMeta(
        name="momentum_accel_20d",
        category="momentum",
        description="20-day momentum acceleration (2nd derivative of price)",
        lookback_days=40,
        version="1.0.0",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        mom_20 = close.pct_change(periods=20)
        result = mom_20 - mom_20.shift(20)
        return result.stack().sort_index()
