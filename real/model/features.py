"""Feature engineering for alpha models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column cross-sectionally per date.

    Args:
        df: Multi-indexed DataFrame (trade_date, symbol).

    Returns:
        DataFrame with same shape, z-scored per date.
    """
    if isinstance(df.index, pd.MultiIndex):
        result = df.groupby("trade_date").transform(
            lambda x: (x - x.mean()) / x.std().replace(0, 1)
        )
        return result
    return df


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each column cross-sectionally per date (0 to 1)."""
    if isinstance(df.index, pd.MultiIndex):
        return df.groupby("trade_date").rank(pct=True)
    return df.rank(pct=True)


def handle_missing(
    df: pd.DataFrame, method: str = "cross_sectional_median"
) -> pd.DataFrame:
    """Fill missing factor values.

    method:
        "cross_sectional_median" - fill with median per date
        "cross_sectional_mean" - fill with mean per date
        "zero" - fill with 0
    """
    if df.empty:
        return df

    if method == "zero":
        return df.fillna(0.0)

    if isinstance(df.index, pd.MultiIndex):
        if method == "cross_sectional_mean":
            return df.groupby("trade_date").transform(lambda x: x.fillna(x.mean()))
        else:
            return df.groupby("trade_date").transform(lambda x: x.fillna(x.median()))
    else:
        if method == "cross_sectional_mean":
            return df.fillna(df.mean())
        else:
            return df.fillna(df.median())


def build_features(
    factor_df: pd.DataFrame,
    raw_data: pd.DataFrame | None = None,
    standardize: bool = True,
    fill_na: bool = True,
) -> pd.DataFrame:
    """Build model-ready feature matrix from factor values.

    Steps:
    1. Fill missing values (cross-sectional median)
    2. Standardize (cross-sectional z-score)

    Args:
        factor_df: Multi-indexed DataFrame (trade_date, symbol) with factor columns.
        raw_data: Optional raw OHLCV data for additional features.
        standardize: Whether to z-score features.
        fill_na: Whether to fill missing values.

    Returns:
        Feature DataFrame ready for model training/inference.
    """
    feats = factor_df.copy()

    if fill_na:
        feats = handle_missing(feats)

    if standardize:
        feats = cross_sectional_zscore(feats)

    # Add raw data features if provided
    if raw_data is not None:
        raw_feats = raw_data.select_dtypes(include=[np.number])
        if fill_na:
            raw_feats = handle_missing(raw_feats)
        if standardize:
            raw_feats = cross_sectional_zscore(raw_feats)
        feats = pd.concat([feats, raw_feats], axis=1)

    return feats
