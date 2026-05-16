"""Cross-sectional long-short strategy: rank by alpha, long top, short bottom."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy
from strategy.portfolio import equal_weight_distribution
from strategy.risk import industry_neutralize, market_cap_neutralize


class CrossSectionalStrategy(Strategy):
    """Rank stocks by composite signal or model alpha, long top decile.

    Supports long-only or long-short, with optional sector and size neutralization.
    """

    name = "cross_sectional"

    def __init__(
        self,
        signal_col: str = "alpha",
        top_quantile: float = 0.2,
        bottom_quantile: float = 0.0,
        long_only: bool = True,
        n_positions: int = 50,
        neutralize_industry: bool = False,
        neutralize_market_cap: bool = False,
    ):
        self.signal_col = signal_col
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.long_only = long_only
        self.n_positions = n_positions
        self.neutralize_industry = neutralize_industry
        self.neutralize_market_cap = neutralize_market_cap

    def generate_weights(
        self,
        date: pd.Timestamp,
        universe: list[str],
        data: pd.DataFrame,
        prices: pd.Series,
        current_positions: dict[str, float],
        cash: float,
    ) -> pd.Series:
        signal = data[self.signal_col].dropna()
        if signal.empty:
            return pd.Series(dtype=float)

        # Filter to universe
        signal = signal[signal.index.isin(universe)]

        # Neutralize
        if self.neutralize_industry and "industry" in data.columns:
            signal = industry_neutralize(signal, data["industry"])
        if self.neutralize_market_cap and "ln_market_cap" in data.columns:
            signal = market_cap_neutralize(signal, data["ln_market_cap"])

        long_n = max(1, int(len(signal) * self.top_quantile))
        short_n = 0 if self.long_only else max(1, int(len(signal) * self.bottom_quantile))

        ranked = signal.sort_values(ascending=False)

        long_syms = ranked.head(long_n).index.tolist()[:self.n_positions]
        short_syms = [] if self.long_only else ranked.tail(short_n).index.tolist()[:self.n_positions]

        weights = pd.Series(0.0, index=ranked.index)
        if long_syms:
            weights.loc[long_syms] = 1.0 / len(long_syms)
        if short_syms:
            weights.loc[short_syms] = -1.0 / len(short_syms)

        # Normalize so sum(|weights|) <= 1.0
        total_abs = weights.abs().sum()
        if total_abs > 0:
            weights = weights / total_abs

        return weights[weights != 0]
