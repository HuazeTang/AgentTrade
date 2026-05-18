"""Overnight gap factors — A-share specific signals.

A-share market has a 9:15-9:25 call auction that sets the opening price.
Overnight gaps (open vs previous close) reflect information accumulated
overnight and are a powerful signal for next-day direction.
"""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class OvernightGap5D(Factor):
    """Average overnight gap return over the last 5 trading days.

    Overnight gap = (open - prev_close) / prev_close.
    High positive = persistent gap-up openings (strong buying pressure).
    High negative = persistent gap-downs (selling pressure / panic).
    Computed as signed average — direction matters more than magnitude.
    """

    meta = FactorMeta(
        name="overnight_gap_5d",
        category="overnight",
        description="5-day average overnight gap return (open vs prev close)",
        lookback_days=6,
        version="1.0.0",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["open", "pre_close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        o = data["open"].unstack()
        pc = data["pre_close"].unstack()
        gap = (o - pc) / pc
        result = gap.rolling(window=5).mean()
        return result.stack().sort_index()


@register_factor
class GapStrength5D(Factor):
    """Ratio of positive-gap days to total gap days over the last 5 days.

    1.0 = all 5 days opened higher (strong upward bias).
    0.0 = all 5 days opened lower (persistent selling into open).
    Unlike raw gap magnitude, this captures consistency of direction.
    """

    meta = FactorMeta(
        name="gap_strength_5d",
        category="overnight",
        description="5-day ratio of positive-gap days (consistency of upward openings)",
        lookback_days=6,
        version="1.0.0",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["open", "pre_close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        o = data["open"].unstack()
        pc = data["pre_close"].unstack()
        gap = (o - pc) / pc
        pos_days = (gap > 0).astype(float)
        result = pos_days.rolling(window=5).mean()
        return result.stack().sort_index()
