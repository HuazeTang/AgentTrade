"""Downside risk and drawdown-control factors.

These capture the asymmetry of risk: downside volatility hurts more than
upside volatility helps. Used defensively to avoid stocks in freefall
and identify those with controlled, steady uptrends.
"""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class DownsideVolatility20D(Factor):
    """Volatility computed only on down-close days (semi-deviation).

    Captures pure downside risk, unlike total volatility which includes
    upside. Stocks with high downside vol tend to have deeper drawdowns.
    """

    meta = FactorMeta(
        name="downside_vol_20d",
        category="risk",
        description="20-day standard deviation of negative daily returns only",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret = close.pct_change()
        downside = ret.clip(upper=0)  # keep only negative returns
        vol = downside.rolling(window=20).std() * np.sqrt(252)
        return vol.stack().sort_index()


@register_factor
class MaxDrawdown20D(Factor):
    """Current drawdown from 20-day peak — how far below the recent high.

    Direct drawdown measure. -0.15 means 15% below 20-day peak (freefall warning).
    Values near 0 mean at or near recent highs.
    """

    meta = FactorMeta(
        name="max_dd_20d",
        category="risk",
        description="Current drawdown from 20-day peak (0 = at peak, negative = declining)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        rolling_peak = close.rolling(window=20, min_periods=10).max()
        result = (close - rolling_peak) / rolling_peak
        return result.stack().sort_index()


@register_factor
class RiskAdjustedMomentum20D(Factor):
    """20-day momentum divided by 20-day volatility.

    Classic risk-adjusted trend: steady uptrend = high, volatile jump = low.
    Equivalent to a rolling 20-day Sharpe ratio of the stock.
    """

    meta = FactorMeta(
        name="risk_adj_mom_20d",
        category="risk",
        description="20-day return / 20-day daily return volatility (rolling Sharpe-like)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret = close.pct_change()
        mom = close.pct_change(periods=20)
        vol = ret.rolling(window=20).std()
        result = mom / (vol + 1e-10)
        return result.stack().sort_index()


@register_factor
class DrawdownRecovery5D(Factor):
    """5-day return from the 20-day low — bounce strength.

    High positive = strong V-shaped recovery. Negative = still near lows.
    Helps distinguish "catching a falling knife" from genuine reversal.
    """

    meta = FactorMeta(
        name="dd_recovery_5d",
        category="risk",
        description="5-day return from 20-day low (bounce detection)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "low"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        low = data["low"].unstack()
        low_20 = low.rolling(window=20).min()
        result = (close - low_20) / (low_20 + 1e-10)
        return result.stack().sort_index()


@register_factor
class MarketDrawdownBeta20D(Factor):
    """Market drawdown × individual stock beta.

    When the market (equal-weighted average) is in a drawdown, high-beta
    stocks get penalized much harder. Low-beta stocks survive the drawdown.
    This is cross-sectionally meaningful and directly controls market risk.
    """

    meta = FactorMeta(
        name="market_dd_beta_20d",
        category="risk",
        description="Market 20-day drawdown × stock beta (market risk exposure)",
        lookback_days=60,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret = close.pct_change()

        # Market = equal-weighted average return
        mkt_ret = ret.mean(axis=1)

        # Market 20-day drawdown (negative = market falling)
        mkt_peak = (1 + mkt_ret).cumprod().rolling(window=20).max()
        mkt_cum = (1 + mkt_ret).cumprod()
        mkt_dd = mkt_cum / mkt_peak - 1  # 0 to negative

        # Individual stock beta (60-day rolling vs market)
        cov = ret.rolling(window=60).cov(mkt_ret)
        var = mkt_ret.rolling(window=60).var()
        beta = cov.div(var + 1e-10, axis=0)

        # Market drawdown × beta: high-beta stocks suffer more when market falls
        # Use negative convention: negative = risky (market down × high beta)
        result = mkt_dd * beta.clip(lower=0, upper=3)
        return result.stack().sort_index()
