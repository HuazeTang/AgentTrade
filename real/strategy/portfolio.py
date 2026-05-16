"""Portfolio construction methods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(signals: pd.Series, top_n: int = 50) -> pd.Series:
    """Long top_n stocks equally weighted."""
    ranked = signals.sort_values(ascending=False)
    selected = ranked.head(top_n).index
    weights = pd.Series(0.0, index=signals.index)
    weights.loc[selected] = 1.0 / len(selected)
    return weights


def value_weight(
    signals: pd.Series, market_caps: pd.Series, top_n: int = 50
) -> pd.Series:
    """Long top_n stocks weighted by market cap."""
    ranked = signals.sort_values(ascending=False)
    selected = ranked.head(top_n).index
    caps = market_caps.loc[market_caps.index.isin(selected)]
    weights = pd.Series(0.0, index=signals.index)
    if caps.sum() > 0:
        weights.loc[caps.index] = caps / caps.sum()
    return weights


def signal_weight(signals: pd.Series, top_n: int = 50) -> pd.Series:
    """Long top_n stocks weighted proportionally to signal strength."""
    ranked = signals.sort_values(ascending=False)
    selected = ranked.head(top_n)
    weights = pd.Series(0.0, index=signals.index)
    abs_sum = selected.abs().sum()
    if abs_sum > 0:
        weights.loc[selected.index] = selected / abs_sum
    return weights


def equal_weight_distribution(signals: pd.Series, top_quantile: float = 0.2) -> pd.Series:
    """Long top quantile equally weighted."""
    n = max(1, int(len(signals) * top_quantile))
    return equal_weight(signals, top_n=n)


def risk_parity_weights(
    returns: pd.DataFrame, signals: pd.Series, top_n: int = 50
) -> pd.Series:
    """Risk parity: weight inversely proportional to volatility."""
    ranked = signals.sort_values(ascending=False)
    selected = ranked.head(top_n).index
    vols = returns[selected].std() if not returns.empty else pd.Series(1.0, index=selected)
    vols = vols.replace(0, np.nan)
    inv_vol = 1.0 / vols.fillna(vols.mean())
    weights = pd.Series(0.0, index=signals.index)
    if inv_vol.sum() > 0:
        weights.loc[inv_vol.index] = inv_vol / inv_vol.sum()
    return weights
