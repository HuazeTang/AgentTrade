"""Time-ordered train/test split with purge gap to prevent leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def time_series_split(
    dates: pd.Series,
    n_splits: int = 5,
    gap: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create time-ordered splits with a purge gap.

    Args:
        dates: Series of dates aligned with the data.
        n_splits: Number of train/test splits.
        gap: Number of trading days to skip between train and test sets.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    unique_dates = sorted(dates.unique())
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    test_size = len(unique_dates) // (n_splits + 1)
    if test_size < 1:
        test_size = 1

    for i in range(n_splits):
        test_end = len(unique_dates) - i * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        train_start = 0

        if train_end <= train_start or test_start >= test_end:
            continue

        train_dates = set(unique_dates[train_start:train_end])
        test_dates = set(unique_dates[test_start:test_end])

        train_idx = np.where(dates.isin(train_dates))[0]
        test_idx = np.where(dates.isin(test_dates))[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def purged_kfold(
    dates: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold cross-validation for time series.

    Includes embargo: a fraction of the test set that is removed
    from the training set to prevent information leakage.

    Args:
        dates: Series of dates (must be sorted).
        n_splits: Number of folds.
        embargo_pct: Fraction of most recent training data to purge.

    Returns:
        List of (train_indices, test_indices).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_samples = len(dates)
    embargo_size = max(1, int(n_samples * embargo_pct))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_raw, test_raw in tscv.split(dates):
        # Apply embargo: remove the last `embargo_size` samples from train
        if len(train_raw) > embargo_size:
            train_idx = train_raw[:-embargo_size]
        else:
            train_idx = train_raw

        splits.append((train_idx, test_raw))

    return splits
