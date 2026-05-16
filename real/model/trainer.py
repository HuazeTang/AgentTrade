"""Model training pipeline orchestrator."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from model.base import AlphaModel
from model.features import build_features
from model.split import time_series_split

logger = logging.getLogger(__name__)


def compute_forward_returns(
    prices: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """Compute forward N-day returns.

    Args:
        prices: Multi-indexed (trade_date, symbol) Series of close prices.
        horizon: Number of trading days forward.

    Returns:
        Series with same index, forward return values.
    """
    close = prices.unstack()
    fwd = close.shift(-horizon) / close - 1
    return fwd.stack().sort_index()


def train_model(
    model: AlphaModel,
    factor_df: pd.DataFrame,
    prices: pd.Series,
    horizon: int = 5,
    train_start: str | None = None,
    train_end: str | None = None,
) -> tuple[AlphaModel, pd.DataFrame, pd.Series]:
    """Train an alpha model on factor data.

    Args:
        model: AlphaModel instance to train.
        factor_df: Multi-indexed DataFrame (trade_date, symbol) with factor columns.
        prices: Multi-indexed Series (trade_date, symbol) of close prices.
        horizon: Forward return horizon in days.
        train_start: Optional start date filter (iso format).
        train_end: Optional end date filter (iso format).

    Returns:
        (trained_model, feature_matrix, target_series)
    """
    # Build features
    X = build_features(factor_df)
    y = compute_forward_returns(prices, horizon=horizon)

    # Align
    common = X.index.intersection(y.dropna().index)
    X = X.loc[common]
    y = y.loc[common]

    # Filter by date
    dates = X.index.get_level_values("trade_date")
    if train_start:
        mask = dates >= pd.Timestamp(train_start)
        X, y = X[mask], y[mask]
    if train_end:
        mask = dates <= pd.Timestamp(train_end)
        X, y = X[mask], y[mask]

    # Drop any remaining NaNs
    valid = (~X.isna().any(axis=1)) & (~y.isna())
    X, y = X[valid], y[valid]

    if len(y) < 100:
        raise ValueError(
            f"Insufficient training data: {len(y)} samples after cleaning"
        )

    logger.info("Training on %d samples, %d features", len(y), X.shape[1])
    model.fit(X, y)

    # In-sample evaluation
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    ic = np.corrcoef(y, pred)[0, 1]
    logger.info("In-sample RMSE: %.6f, IC: %.4f", rmse, ic)

    return model, X, y
