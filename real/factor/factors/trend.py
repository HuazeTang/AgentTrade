"""Trend-following factors for identifying major uptrends (主升浪)."""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class TrendEfficiency20D(Factor):
    """Kaufman-style Efficiency Ratio: price direction / path length.

    High values = smooth trending move (主升浪 candidate).
    Low values = choppy/sideways.
    """

    meta = FactorMeta(
        name="trend_efficiency_20d",
        category="trend",
        description="20-day price efficiency ratio (|net change| / total path)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret_1d = close.pct_change()
        ret_20d = close.pct_change(periods=20)
        path = ret_1d.abs().rolling(window=20).sum()
        efficiency = ret_20d.abs() / (path + 1e-10)
        return efficiency.stack().sort_index()


@register_factor
class MATrend5x20(Factor):
    """Price position relative to 20-day MA — core trend-following signal."""

    meta = FactorMeta(
        name="ma_trend_5_20",
        category="trend",
        description="(Close - MA20) / MA20, positive = above average = uptrend",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ma20 = close.rolling(window=20).mean()
        result = (close - ma20) / (ma20 + 1e-10)
        return result.stack().sort_index()


@register_factor
class DonchianPct20D(Factor):
    """Position within 20-day Donchian channel: (close - low) / (high - low).

    1.0 = at new 20-day high (breakout), 0.0 = at 20-day low.
    """

    meta = FactorMeta(
        name="donchian_pct_20d",
        category="trend",
        description="Position in 20-day Donchian channel (1.0 = new high breakout)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "low", "high"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"].unstack()
        low = data["low"].unstack()
        close = data["close"].unstack()
        hh20 = high.rolling(window=20).max()
        ll20 = low.rolling(window=20).min()
        result = (close - ll20) / (hh20 - ll20 + 1e-10)
        return result.stack().sort_index()


@register_factor
class UpDaysRatio20D(Factor):
    """Ratio of up-close days in the last 20 trading days.

    Measures persistent buying pressure — hallmark of 主升浪.
    """

    meta = FactorMeta(
        name="up_days_ratio_20d",
        category="trend",
        description="Proportion of days with positive close change in last 20 days",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        up = (close.pct_change() > 0).astype(float)
        result = up.rolling(window=20).mean()
        return result.stack().sort_index()


@register_factor
class MACrossover5x20(Factor):
    """5-day vs 20-day MA crossover signal.

    Short-term trend alignment with medium-term: positive = golden cross zone.
    """

    meta = FactorMeta(
        name="ma_cross_5_20",
        category="trend",
        description="MA5 / MA20 - 1, short-term vs medium-term trend alignment",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ma5 = close.rolling(window=5).mean()
        ma20 = close.rolling(window=20).mean()
        result = ma5 / (ma20 + 1e-10) - 1.0
        return result.stack().sort_index()
