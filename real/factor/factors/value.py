"""Value factors: EP, BP, SP.

Note: These factors require financial statement data (earnings, book value, revenue)
which are not in daily OHLCV. They return NaN if the required fields are absent.
"""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide a / b, returning 0 where b == 0 or b is NaN."""
    result = a / b.replace(0, np.nan)
    return result.fillna(0.0)


@register_factor
class EP(Factor):
    meta = FactorMeta(
        name="ep",
        category="value",
        description="Earnings-to-price ratio (1/PE). Requires earnings column.",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "earnings"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "earnings" not in data.columns:
            return pd.Series(0.0, index=data.index)
        close = data["close"]
        earnings = data["earnings"]
        return _safe_div(earnings, close).rename("ep")


@register_factor
class BP(Factor):
    meta = FactorMeta(
        name="bp",
        category="value",
        description="Book-to-price ratio (1/PB). Requires book_value column.",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "book_value"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "book_value" not in data.columns:
            return pd.Series(0.0, index=data.index)
        return _safe_div(data["book_value"], data["close"]).rename("bp")


@register_factor
class SP(Factor):
    meta = FactorMeta(
        name="sp",
        category="value",
        description="Sales-to-price ratio (1/PS). Requires revenue column.",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "revenue"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "revenue" not in data.columns:
            return pd.Series(0.0, index=data.index)
        return _safe_div(data["revenue"], data["close"]).rename("sp")
