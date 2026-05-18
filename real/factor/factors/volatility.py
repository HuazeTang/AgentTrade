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


@register_factor
class VolRatio20x60(Factor):
    """Short-term vs mid-term volatility ratio — regime change detector.

    > 1.0 = short-term vol exceeding mid-term (breakout/panic starting).
    < 1.0 = vol contracting (consolidation / calm market).
    A-shares often experience explosive vol expansion at trend starts.
    """

    meta = FactorMeta(
        name="vol_ratio_20_60",
        category="volatility",
        description="20-day vol / 60-day vol — volatility regime change",
        lookback_days=60,
        version="1.0.0",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret = close.pct_change()
        vol_20 = ret.rolling(window=20).std()
        vol_60 = ret.rolling(window=60).std()
        result = vol_20 / (vol_60 + 1e-10)
        return result.stack().sort_index()


@register_factor
class DailyAmplitude20D(Factor):
    """Average intraday amplitude (high-low range) over 20 days.

    A-shares have higher intraday volatility than US markets.
    Stocks with consistently wide ranges tend to attract speculative capital.
    Normalized by close to make cross-sectionally comparable.
    """

    meta = FactorMeta(
        name="daily_amplitude_20d",
        category="volatility",
        description="20-day average (high-low)/close — intraday range intensity",
        lookback_days=20,
        version="1.0.0",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["high", "low", "close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"].unstack()
        low = data["low"].unstack()
        close = data["close"].unstack()
        amplitude = (high - low) / close
        result = amplitude.rolling(window=20).mean()
        return result.stack().sort_index()
