"""Volatility and beta factors."""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class HistoricalVolatility20D(Factor):
    meta = FactorMeta(
        name="volatility_20d",
        category="volatility",
        description="20-day annualized historical volatility",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(window=20).std() * np.sqrt(252)
        return vol.stack().sort_index()


@register_factor
class HistoricalVolatility60D(Factor):
    meta = FactorMeta(
        name="volatility_60d",
        category="volatility",
        description="60-day annualized historical volatility",
        lookback_days=60,
    )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(window=60).std() * np.sqrt(252)
        return vol.stack().sort_index()


@register_factor
class Beta60D(Factor):
    meta = FactorMeta(
        name="beta_60d",
        category="volatility",
        description="60-day market beta (relative to equal-weighted average return)",
        lookback_days=60,
    )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret = close.pct_change()
        mkt_ret = ret.mean(axis=1)
        cov = ret.rolling(window=60).cov(mkt_ret)
        var = mkt_ret.rolling(window=60).var()
        beta = cov.div(var, axis=0)
        return beta.stack().sort_index()
