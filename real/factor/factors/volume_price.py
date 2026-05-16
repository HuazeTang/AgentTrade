"""Volume-price integrated factors: combining price direction with volume confirmation.

These capture the classic technical principle that price moves on high volume
are more reliable than moves on low volume.
"""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class VolWeightedMomentum5D(Factor):
    """5-day price momentum weighted by volume surge ratio.

    Strong breakout on high volume → more reliable continuation signal.
    """

    meta = FactorMeta(
        name="vol_weighted_mom_5d",
        category="volume_price",
        description="5-day price change × relative volume (vol_5 / vol_20)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "volume"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        volume = data["volume"].unstack()
        mom = close.pct_change(periods=5)
        vol_5 = volume.rolling(window=5).mean()
        vol_20 = volume.rolling(window=20).mean()
        vol_ratio = vol_5 / (vol_20 + 1e-10)
        result = mom * vol_ratio
        return result.stack().sort_index()


@register_factor
class MoneyFlowRatio20D(Factor):
    """Direction of money flow over 20 days.

    Money flow = close × volume. Separates into positive (up days) and
    negative (down days). Ratio = sum(positive MF) / sum(total MF).
    Values > 0.5 = net buying, < 0.5 = net selling.
    """

    meta = FactorMeta(
        name="money_flow_ratio_20d",
        category="volume_price",
        description="Ratio of positive money flow on up days to total money flow over 20d",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "volume", "pre_close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        volume = data["volume"].unstack()
        pre_close = data["pre_close"].unstack()

        mf = close * volume
        is_up = close > pre_close
        pos_mf = mf.where(is_up, 0.0)
        neg_mf = mf.where(~is_up, 0.0)

        pos_sum = pos_mf.rolling(window=20).sum()
        neg_sum = neg_mf.rolling(window=20).sum()
        total_sum = pos_sum + neg_sum

        # Ratio of positive to total, with prior: 0.5 when no flow
        result = pos_sum / (total_sum + 1e-10)
        return result.stack().sort_index()


@register_factor
class VWAPDelta5D(Factor):
    """Deviation of close from 5-day volume-weighted average price.

    Close above VWAP = buyers in control (paying above average price).
    Close below VWAP = sellers in control.
    """

    meta = FactorMeta(
        name="vwap_delta_5d",
        category="volume_price",
        description="(Close - VWAP_5d) / VWAP_5d, volume-weighted price anchor",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "volume"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        volume = data["volume"].unstack()

        pv = close * volume
        vwap = pv.rolling(window=5).sum() / (volume.rolling(window=5).sum() + 1e-10)
        result = (close - vwap) / (vwap + 1e-10)
        return result.stack().sort_index()


@register_factor
class VolPriceDivergence5D(Factor):
    """Divergence between price momentum and volume momentum.

    Price up but volume down = weakening trend (potential reversal).
    Price down but volume up = accumulation (potential bounce).
    Captures the classic "volume leads price" divergence pattern.
    """

    meta = FactorMeta(
        name="vol_price_div_5d",
        category="volume_price",
        description="5-day price change minus 5-day volume change (divergence detector)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "volume"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        volume = data["volume"].unstack()

        price_chg = close.pct_change(periods=5)
        vol_chg = volume.pct_change(periods=5)

        # Normalize to comparable scales via z-score
        price_z = (price_chg - price_chg.mean()) / (price_chg.std() + 1e-10)
        vol_z = (vol_chg - vol_chg.mean()) / (vol_chg.std() + 1e-10)

        # Divergence = price momentum minus volume momentum
        # Positive = price leading volume up (potential continuation)
        # Negative = price lagging volume (potential reversal)
        result = price_z - vol_z
        return result.stack().sort_index()
