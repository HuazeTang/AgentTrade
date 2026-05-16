"""Risk controls: neutralization and constraint methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def industry_neutralize(
    signals: pd.Series, industries: pd.Series
) -> pd.Series:
    """Regress out industry dummies from signals.

    Args:
        signals: Series indexed by symbol with signal values.
        industries: Series indexed by symbol with industry labels.

    Returns:
        Residual signals after regressing out industry effects.
    """
    common_idx = signals.index.intersection(industries.index)
    if len(common_idx) < 10:
        return signals

    sig = signals.loc[common_idx]
    ind = industries.loc[common_idx]

    dummies = pd.get_dummies(ind, drop_first=True)
    if dummies.shape[1] == 0 or dummies.shape[1] > len(sig) // 2:
        return signals

    X = dummies.values.astype(float)
    y = sig.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    residual = y.flatten() - model.predict(X).flatten()

    result = signals.copy()
    result.loc[common_idx] = residual
    return result


def market_cap_neutralize(
    signals: pd.Series, market_caps: pd.Series
) -> pd.Series:
    """Regress out log market cap from signals.

    Args:
        signals: Series indexed by symbol with signal values.
        market_caps: Series indexed by symbol (typically ln_market_cap).

    Returns:
        Residual signals after regressing out size.
    """
    common_idx = signals.index.intersection(market_caps.index).dropna()
    if len(common_idx) < 10:
        return signals

    sig = signals.loc[common_idx]
    cap = market_caps.loc[common_idx].replace([np.inf, -np.inf], np.nan).dropna()

    common_idx = sig.index.intersection(cap.index)
    if len(common_idx) < 10:
        return signals

    X = cap.loc[common_idx].values.reshape(-1, 1).astype(float)
    y = sig.loc[common_idx].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    residual = y.flatten() - model.predict(X).flatten()

    result = signals.copy()
    result.loc[common_idx] = residual
    return result


def max_position_constraint(
    weights: pd.Series, max_pct: float = 0.10
) -> pd.Series:
    """Clip weights to max_pct and renormalize."""
    weights = weights.clip(upper=max_pct)
    total = weights.abs().sum()
    if total > 0:
        weights = weights / total
    return weights
