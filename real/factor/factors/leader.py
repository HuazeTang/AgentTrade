"""Leader (龙头) stock identification factors.

A-share specific: captures stocks with frequent limit-up (涨停), relative strength
vs sector, volume confirmation, and strong intraday closes — hallmarks of leading
stocks in a rally.
"""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class LimitUpFreq20D(Factor):
    """Count of 涨停 (limit-up) days in the last 20 trading days.

    A-share main board: close >= pre_close * 1.098 or close == high.
    STAR/Chinext (300/688): 20% limit.
    """

    meta = FactorMeta(
        name="limit_up_freq_20d",
        category="leader",
        description="Number of limit-up (涨停) days in last 20 trading days",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "pre_close", "high", "price_limit_frac"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        high = data["high"].unstack()
        pre_close = data["pre_close"].unstack()
        # price_limit_frac varies by board (0.10 main, 0.20 STAR/Chinext)
        pl_frac = data["price_limit_frac"].unstack().fillna(0.10)

        daily_ret = close / pre_close - 1.0
        # A stock is limit-up if close == high AND return >= limit - small buffer
        limit_up = (close == high) & (daily_ret >= (pl_frac - 0.005))

        result = limit_up.astype(float).rolling(window=20).sum()
        return result.stack().sort_index()


@register_factor
class RelativeStrength10D(Factor):
    """10-day excess return vs sector peer average.

    True leaders (龙头) outperform their own sector — this separates sector
    tailwinds from genuine leadership.
    """

    meta = FactorMeta(
        name="relative_strength_10d",
        category="leader",
        description="10-day return minus sector average return (leader vs followers)",
        lookback_days=10,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "board"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        board = data["board"].unstack().iloc[-1]  # board is time-invariant per symbol

        ret_10d = close.pct_change(periods=10)

        # Sector-average return per day
        sector_ret = pd.DataFrame(index=ret_10d.index)
        for b in board.dropna().unique():
            syms_in_board = board[board == b].index.intersection(ret_10d.columns)
            if len(syms_in_board) > 0:
                sector_ret[b] = ret_10d[syms_in_board].mean(axis=1)

        avg_sector_ret = pd.Series(index=ret_10d.index, dtype=float)
        for sym in ret_10d.columns:
            b = board.get(sym)
            if b and b in sector_ret.columns:
                avg_sector = sector_ret[b]
            else:
                avg_sector = sector_ret.mean(axis=1)
            ret_10d[sym] = ret_10d[sym] - avg_sector

        return ret_10d.stack().sort_index()


@register_factor
class VolumeSurge5D(Factor):
    """5-day avg volume / 20-day avg volume ratio.

    A breakout without volume is suspect. This confirms that buying is real.
    """

    meta = FactorMeta(
        name="volume_surge_5d",
        category="leader",
        description="Ratio of 5-day avg volume to 20-day avg volume (surge confirmation)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["volume"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        volume = data["volume"].unstack()
        vol_5 = volume.rolling(window=5).mean()
        vol_20 = volume.rolling(window=20).mean()
        result = vol_5 / (vol_20 + 1e-10)
        return result.stack().sort_index()


@register_factor
class ClosePosition5D(Factor):
    """5-day average of (close - low) / (high - low).

    Stocks that consistently close near the day's high show persistent
    intraday buying pressure — characteristic of 龙头 stocks.
    Returns 0.5 where high == low (no range).
    """

    meta = FactorMeta(
        name="close_position_5d",
        category="leader",
        description="5-day avg intraday close position (close near high = bullish)",
        lookback_days=5,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "high", "low"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        high = data["high"].unstack()
        low = data["low"].unstack()
        hilo_range = high - low
        position = (close - low) / hilo_range.replace(0, np.nan)
        position = position.fillna(0.5)  # no range day = neutral
        result = position.rolling(window=5).mean()
        return result.stack().sort_index()
