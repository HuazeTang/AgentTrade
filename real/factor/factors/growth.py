"""Growth factors.

These require financial statement data (revenue growth, earnings growth)
which are not in daily OHLCV. They return NaN if required fields are absent.
"""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class RevenueGrowthYoY(Factor):
    meta = FactorMeta(
        name="revenue_growth_yoy",
        category="growth",
        description="Year-over-year revenue growth rate",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["revenue"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "revenue" not in data.columns:
            return pd.Series(0.0, index=data.index)
        rev = data["revenue"].unstack()
        growth = rev.pct_change(periods=4)  # quarterly: 4 quarters = 1 year
        return growth.stack().sort_index()


@register_factor
class EarningsGrowthYoY(Factor):
    meta = FactorMeta(
        name="earnings_growth_yoy",
        category="growth",
        description="Year-over-year earnings growth rate",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["earnings"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "earnings" not in data.columns:
            return pd.Series(0.0, index=data.index)
        earn = data["earnings"].unstack()
        growth = earn.pct_change(periods=4)
        return growth.stack().sort_index()
