"""Short-term reversal factors."""

import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class ShortTermReversal5D(Factor):
    meta = FactorMeta(
        name="reversal_5d",
        category="momentum",
        description="5-day price reversal (negative of 5-day return)",
        lookback_days=5,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret_5d = close.pct_change(periods=5)
        reversal = -ret_5d
        return reversal.stack().sort_index()


@register_factor
class ShortTermReversal10D(Factor):
    meta = FactorMeta(
        name="reversal_10d",
        category="momentum",
        description="10-day price reversal (negative of 10-day return)",
        lookback_days=10,
    )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        ret_10d = close.pct_change(periods=10)
        reversal = -ret_10d
        return reversal.stack().sort_index()
