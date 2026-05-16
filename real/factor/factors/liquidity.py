"""Liquidity factors."""

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor


@register_factor
class Turnover20D(Factor):
    meta = FactorMeta(
        name="turnover_20d",
        category="liquidity",
        description="20-day average daily turnover rate",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["turnover"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "turnover" not in data.columns:
            return pd.Series(0.0, index=data.index)
        turnover = data["turnover"].unstack()
        avg_turnover = turnover.rolling(window=20).mean()
        return avg_turnover.stack().sort_index()


@register_factor
class AmihudIlliquidity20D(Factor):
    meta = FactorMeta(
        name="amihud_20d",
        category="liquidity",
        description="20-day Amihud illiquidity: avg(|return| / amount)",
        lookback_days=20,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "amount"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack()
        amount = data["amount"].unstack()
        ret = close.pct_change().abs()
        illiq = (ret / amount.replace(0, np.nan)).rolling(window=20).mean()
        result = illiq.stack().sort_index()
        # Clip extreme values
        return result.replace([np.inf, -np.inf], np.nan).clip(
            lower=result.quantile(0.01), upper=result.quantile(0.99)
        )
