"""V4 feature computation: raw OHLCV only + per-stock z-score normalization.

Design principle: let the Transformer learn patterns from raw price/volume data
rather than hand-crafted consolidation/divergence features.

Feature set (16 dims):
  - RAW_OHLCV (8): open_ret, high_ret, low_ret, close_ret, log_volume, log_amount,
                    turnover, log_mcap — cross-sectional z-score
  - RAW_OHLCV_TS (8): same 8 features with per-stock rolling 60d z-score
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_OHLCV_COLUMNS: list[str] = [
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "log_volume",
    "log_amount",
    "turnover",
    "log_mcap",
]

# V4 uses per-stock z-score for ALL raw features (not just 4 like V3)
PER_STOCK_ZSCORE_COLS: list[str] = list(RAW_OHLCV_COLUMNS)

ALL_V4_COLUMNS = list(RAW_OHLCV_COLUMNS)


# ═════════════════════════════════════════════════════════════════════════════
# Feature Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_raw_ohlcv_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute raw OHLCV returns/logs (same as V3).

    All computations use shift(1) to avoid look-ahead bias.
    """
    if data.empty:
        return data

    result = data.copy()
    close = data["close"].unstack()
    high = data["high"].unstack() if "high" in data.columns else None
    low = data["low"].unstack() if "low" in data.columns else None
    open_ = data["open"].unstack() if "open" in data.columns else None
    pre_close = data["pre_close"].unstack() if "pre_close" in data.columns else close.shift(1)
    volume = data["volume"].unstack() if "volume" in data.columns else None
    amount = data["amount"].unstack() if "amount" in data.columns else None
    turnover = data["turnover"].unstack() if "turnover" in data.columns else None
    mcap = data["market_cap"].unstack() if "market_cap" in data.columns else None

    if open_ is not None:
        ser = ((open_.shift(1) - pre_close.shift(1)) / pre_close.shift(1).clip(lower=1e-8)).stack()
        ser.name = "open_ret"; result = result.join(ser, how="left")
    if high is not None:
        ser = ((high.shift(1) - pre_close.shift(1)) / pre_close.shift(1).clip(lower=1e-8)).stack()
        ser.name = "high_ret"; result = result.join(ser, how="left")
    if low is not None:
        ser = ((low.shift(1) - pre_close.shift(1)) / pre_close.shift(1).clip(lower=1e-8)).stack()
        ser.name = "low_ret"; result = result.join(ser, how="left")
    if close is not None:
        ser = ((close.shift(1) - pre_close.shift(1)) / pre_close.shift(1).clip(lower=1e-8)).stack()
        ser.name = "close_ret"; result = result.join(ser, how="left")

    if volume is not None:
        ser = np.log1p(volume.shift(1)).stack()
        ser.name = "log_volume"; result = result.join(ser, how="left")
    if amount is not None:
        ser = np.log1p(amount.shift(1)).stack()
        ser.name = "log_amount"; result = result.join(ser, how="left")

    if turnover is not None:
        if "turnover" in result.columns:
            result = result.drop(columns=["turnover"])
        ser = turnover.shift(1).stack()
        ser.name = "turnover"; result = result.join(ser, how="left")

    if mcap is not None:
        ser = np.log1p(mcap.shift(1)).stack()
        ser.name = "log_mcap"; result = result.join(ser, how="left")

    logger.info("Computed raw OHLCV features: %d columns", len(RAW_OHLCV_COLUMNS))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Normalization
# ═════════════════════════════════════════════════════════════════════════════

def winsorize_cross_sectional(
    df: pd.DataFrame,
    feature_cols: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Per-day winsorization: clip each feature at lower/upper quantiles within each day."""
    result = df.copy()
    available = [c for c in feature_cols if c in result.columns]
    if not available:
        return result

    for col in available:
        try:
            unstacked = result[col].unstack().astype(float)
            lo = unstacked.quantile(lower, axis=1)
            hi = unstacked.quantile(upper, axis=1)
            clipped = unstacked.clip(lo, hi, axis=0)
            result[col] = clipped.stack(future_stack=True)
        except Exception:
            pass

    return result


def cs_zscore_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Per-day cross-sectional z-score: (x - daily_mean) / daily_std."""
    result = df.copy()
    available = [c for c in feature_cols if c in result.columns]
    if not available:
        return result

    for col in available:
        try:
            unstacked = result[col].unstack().astype(float)
            m = unstacked.mean(axis=1)
            s = unstacked.std(axis=1).clip(lower=1e-8)
            zscored = unstacked.sub(m, axis=0).div(s, axis=0)
            result[col] = zscored.stack(future_stack=True)
        except Exception:
            pass

    return result


def compute_per_stock_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    rolling_window: int = 60,
    suffix: str = "_ts",
) -> pd.DataFrame:
    """Per-stock rolling z-score: (x - rolling_mean) / rolling_std.

    Each stock is normalized against its own history, capturing "how unusual
    is today for this specific stock" rather than "how does it rank today".

    Returns a DataFrame with new columns named {col}{suffix}.
    """
    result = df.copy()
    available = [c for c in feature_cols if c in result.columns]
    if not available:
        return result

    for col in available:
        try:
            unstacked = result[col].unstack().astype(float)
            m = unstacked.rolling(rolling_window, min_periods=20).mean()
            s = unstacked.rolling(rolling_window, min_periods=20).std().clip(lower=1e-8)
            z = (unstacked - m) / s
            new_col = f"{col}{suffix}"
            result[new_col] = z.stack(future_stack=True)
        except Exception:
            pass

    logger.info("Computed per-stock z-score for %d features (window=%d)", len(available), rolling_window)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Full Feature Cache Builder
# ═════════════════════════════════════════════════════════════════════════════

def build_v4_feature_cache(
    daily_cache: pd.DataFrame,
    *,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    min_cs_stocks: int = 30,
    per_stock_zscore: bool = True,
    per_stock_window: int = 60,
) -> pd.DataFrame:
    """Build the V4 feature cache.

    Pipeline:
      1. Compute raw OHLCV features (8 dims)
      2. Winsorize + CS z-score all features
      3. Add per-stock rolling z-score for ALL 8 features (+8 dims)

    Returns:
        MultiIndex DataFrame (trade_date, symbol) with ~16 normalized feature columns.
    """
    logger.info("Building V4 feature cache (raw OHLCV + per-stock z-score)...")

    df = compute_raw_ohlcv_features(daily_cache)

    cs_cols = [c for c in ALL_V4_COLUMNS if c in df.columns]
    logger.info("V4 raw features: %d columns", len(cs_cols))

    df = df[cs_cols]
    df = df.dropna(how="all")
    min_non_na = max(1, len(cs_cols) // 2)
    df = df[df.notna().sum(axis=1) >= min_non_na]
    df = df.fillna(0.0)

    # Filter dates with too few stocks
    trade_dates = df.index.get_level_values("trade_date").unique()
    small_dates = []
    for td in trade_dates:
        n_stocks = len(df.loc[pd.Timestamp(td)])
        if n_stocks < min_cs_stocks:
            small_dates.append(td)
    if small_dates:
        logger.info("Dropping %d dates with < %d stocks", len(small_dates), min_cs_stocks)
        df = df[~df.index.get_level_values("trade_date").isin(small_dates)]

    # Winsorize + CS z-score
    df = winsorize_cross_sectional(df, cs_cols, winsor_lower, winsor_upper)
    df = cs_zscore_features(df, cs_cols)

    # Per-stock z-score for ALL raw features (V4: all 8, not just 4 like V3)
    if per_stock_zscore:
        key_cols = [c for c in PER_STOCK_ZSCORE_COLS if c in df.columns]
        df = compute_per_stock_zscore(df, key_cols, rolling_window=per_stock_window)
        ts_cols = [f"{c}_ts" for c in key_cols]
        total_cols = cs_cols + [c for c in ts_cols if c in df.columns]
        logger.info("V4 feature cache: %d rows, %d columns (raw + per-stock z-score)",
                     len(df), len(total_cols))
    else:
        logger.info("V4 feature cache: %d rows, %d columns", len(df), len(cs_cols))

    return df.sort_index()
