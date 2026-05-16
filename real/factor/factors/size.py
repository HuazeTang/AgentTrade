"""Size factor -- natural log of market capitalization."""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class LnMarketCap(Factor):
    meta = FactorMeta(
        name="ln_market_cap",
        category="size",
        description="Natural log of total market capitalization",
    )

    @property
    def required_fields(self) -> list[str]:
        return ["market_cap"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "market_cap" not in data.columns:
            return pd.Series(0.0, index=data.index)
        return np.log(data["market_cap"].clip(lower=1.0)).rename("ln_market_cap")
