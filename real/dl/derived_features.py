"""Shared derived feature computation for yaogu detection pipeline.

Two feature sets are available:
  - DERIVED_FEATURE_COLUMNS (14): Hand-crafted momentum/volatility/volume ratios.
  - RAW_OHLCV_COLUMNS (8): Raw OHLCV data normalized as returns/logs — lets the DL
    model learn its own representations from price action.

All features are ratio/position-based so they are cross-sectionally comparable
across stocks. Each feature is computed using only data known at time T (shifted
by 1 to avoid look-ahead bias).

Features are then per-day winsorized and CS z-scored for use in Stage 1 (LR)
and Stage 2 (DualTowerModel) training/inference.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Derived feature definitions ──────────────────────────────────────────

DERIVED_FEATURE_COLUMNS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_ratio_5d",
    "turnover_ratio",
    "amplitude_20d",
    "close_position_5d",
    "up_days_ratio_5d",
    "vol_surge_5d",
    "overnight_gap_5d",
    "hl_ratio",
    "amihud",
    "vol_20d",
    "ret_vol_ratio_20d",
]

# Columns requiring specific data fields to be present
REQUIRED_COLUMNS = {"close", "open", "high", "low", "volume", "amount", "pre_close"}

# ── Raw OHLCV columns (no pre-computed factors) ─────────────────────────

RAW_OHLCV_COLUMNS: list[str] = [
    "open_ret",       # (open - pre_close) / pre_close  — overnight gap
    "high_ret",       # (high - pre_close) / pre_close  — intraday high
    "low_ret",        # (low - pre_close) / pre_close   — intraday low
    "close_ret",      # (close - pre_close) / pre_close — daily return
    "log_volume",     # log(1 + volume)
    "log_amount",     # log(1 + amount)
    "turnover",       # raw turnover rate
    "log_mcap",       # log(1 + market_cap)
]


def compute_raw_ohlcv_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute raw OHLCV features: price returns + log volume/amount.

    All price columns are expressed as returns relative to pre_close so they
    are stationary and cross-sectionally comparable. No momentum/volatility
    ratios are pre-computed — the DL model learns patterns from raw price action.

    Uses shift(1) to avoid look-ahead: feature at T reflects T-1 data.

    Args:
        data: MultiIndex DataFrame (trade_date, symbol) with OHLCV columns.

    Returns:
        DataFrame with raw OHLCV feature columns added.
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

    # Price returns relative to pre_close, shifted to avoid look-ahead
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

    # Log volume / amount (shifted)
    if volume is not None:
        ser = np.log1p(volume.shift(1)).stack()
        ser.name = "log_volume"; result = result.join(ser, how="left")
    if amount is not None:
        ser = np.log1p(amount.shift(1)).stack()
        ser.name = "log_amount"; result = result.join(ser, how="left")

    # Turnover (already a ratio, shifted) — drop original first to avoid join conflict
    if turnover is not None:
        if "turnover" in result.columns:
            result = result.drop(columns=["turnover"])
        ser = turnover.shift(1).stack()
        ser.name = "turnover"; result = result.join(ser, how="left")

    # Log market cap (shifted)
    if mcap is not None:
        ser = np.log1p(mcap.shift(1)).stack()
        ser.name = "log_mcap"; result = result.join(ser, how="left")

    logger.info("Computed raw OHLCV features: %d columns added", len(RAW_OHLCV_COLUMNS))
    return result


def compute_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features, joining them into the input DataFrame.

    All computations use shift(1) to avoid look-ahead bias: the feature
    value at trade_date T is computed from data up to T only.

    Args:
        data: MultiIndex DataFrame (trade_date, symbol) with OHLCV columns.

    Returns:
        DataFrame with derived feature columns added.
    """
    if data.empty:
        return data

    result = data.copy()

    close = data["close"].unstack()
    high = data["high"].unstack() if "high" in data.columns else None
    low = data["low"].unstack() if "low" in data.columns else None
    volume = data["volume"].unstack() if "volume" in data.columns else None
    amount = data["amount"].unstack() if "amount" in data.columns else None
    turnover = data["turnover"].unstack() if "turnover" in data.columns else None

    pre_close = None
    if "pre_close" in data.columns:
        pre_close = data["pre_close"].unstack()
    else:
        pre_close = close.shift(1)

    syms = list(close.columns)

    # ── Returns (momentum) ──
    for days, name in [(1, "ret_1d"), (5, "ret_5d"), (20, "ret_20d")]:
        series = close.pct_change(days, fill_method=None).shift(1).stack()
        series.name = name
        result = result.join(series, how="left")

    # ── Volatility ──
    # 20d rolling std of daily returns
    daily_ret = close.pct_change(fill_method=None)
    vol_20d_ser = daily_ret.rolling(20, min_periods=10).std().shift(1).stack()
    vol_20d_ser.name = "vol_20d"
    result = result.join(vol_20d_ser, how="left")

    # ── Return-volatility ratio (Sharpe-like) ──
    ret_20d_val = close.pct_change(20, fill_method=None).shift(1)
    vol_20d_val = daily_ret.rolling(20, min_periods=10).std().shift(1)
    ret_vol = (ret_20d_val / vol_20d_val.clip(lower=1e-8)).stack()
    ret_vol.name = "ret_vol_ratio_20d"
    result = result.join(ret_vol, how="left")

    # ── Volume ratio ──
    if volume is not None:
        vol_5d = volume.rolling(5, min_periods=3).mean().shift(1)
        vol_20d = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = (vol_5d / vol_20d.clip(lower=1e-8)).stack()
        vol_ratio.name = "vol_ratio_5d"
        result = result.join(vol_ratio, how="left")

    # ── Turnover ratio ──
    if turnover is not None:
        t_5d = turnover.rolling(5, min_periods=3).mean().shift(1)
        t_20d = turnover.rolling(20, min_periods=10).mean().shift(1)
        t_ratio = (t_5d / t_20d.clip(lower=1e-8)).stack()
        t_ratio.name = "turnover_ratio"
        result = result.join(t_ratio, how="left")

    # ── Amplitude ──
    if high is not None and low is not None:
        roll_high = high.rolling(20, min_periods=10).max().shift(1)
        roll_low = low.rolling(20, min_periods=10).min().shift(1)
        roll_mean = close.rolling(20, min_periods=10).mean().shift(1)
        amp = ((roll_high - roll_low) / roll_mean.clip(lower=1e-8)).stack()
        amp.name = "amplitude_20d"
        result = result.join(amp, how="left")

    # ── Close position ──
    if high is not None and low is not None:
        daily_range = high - low
        cp = ((close - low) / daily_range.clip(lower=1e-8))
        cp_5d = cp.rolling(5, min_periods=3).mean().shift(1).stack()
        cp_5d.name = "close_position_5d"
        result = result.join(cp_5d, how="left")

    # ── Up days ratio ──
    if pre_close is not None:
        up = (close > pre_close.shift(1)).astype(float)
        up_5d = up.rolling(5, min_periods=3).mean().shift(1).stack()
        up_5d.name = "up_days_ratio_5d"
        result = result.join(up_5d, how="left")

    # ── Volume surge ──
    if volume is not None:
        vol_20d_mean = volume.rolling(20, min_periods=10).mean()
        surge = (volume > vol_20d_mean * 1.5).astype(float)
        surge_5d = surge.rolling(5, min_periods=3).mean().shift(1).stack()
        surge_5d.name = "vol_surge_5d"
        result = result.join(surge_5d, how="left")

    # ── Overnight gap ──
    if pre_close is not None:
        og = (close.shift(1) - pre_close) / pre_close.clip(lower=1e-8)
        og_5d = og.rolling(5, min_periods=3).mean().shift(1).stack()
        og_5d.name = "overnight_gap_5d"
        result = result.join(og_5d, how="left")

    # ── HL ratio ──
    if high is not None and low is not None:
        hl = ((high - low) / close.clip(lower=1e-8)).stack()
        hl.name = "hl_ratio"
        result = result.join(hl, how="left")

    # ── Amihud ──
    if amount is not None:
        ami = (daily_ret.abs() / amount.clip(lower=1) * 1e6).shift(1).stack()
        ami.name = "amihud"
        result = result.join(ami, how="left")

    logger.info("Computed derived features: %d columns added", len(DERIVED_FEATURE_COLUMNS))
    return result


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
        except Exception as e:
            logger.warning("winsorize failed for %s: %s", col, e)

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
        except Exception as e:
            logger.warning("cs_zscore failed for %s: %s", col, e)

    return result


def build_normalized_feature_cache(
    daily_cache: pd.DataFrame,
    feature_cols: list[str] | None = None,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    min_cs_stocks: int = 30,
    raw_ohlcv: bool = False,
) -> pd.DataFrame:
    """Compute features, winsorize, and CS z-score — the full normalization pipeline.

    Returns a MultiIndex DataFrame with only the normalized feature columns,
    ready for both Stage 1 (LR) and Stage 2 (sequence extraction).

    Dates with fewer than min_cs_stocks are dropped to avoid noisy CS stats.

    Args:
        daily_cache: Raw MultiIndex (trade_date, symbol) OHLCV data.
        feature_cols: Feature columns to normalize (defaults to DERIVED_FEATURE_COLUMNS).
        winsor_lower, winsor_upper: Quantile bounds for winsorization.
        min_cs_stocks: Minimum stocks per day for reliable CS statistics.
        raw_ohlcv: If True, compute raw OHLCV features instead of derived factors.

    Returns:
        MultiIndex DataFrame with normalized feature columns only.
    """
    if raw_ohlcv:
        cols = feature_cols or RAW_OHLCV_COLUMNS
    else:
        cols = feature_cols or DERIVED_FEATURE_COLUMNS

    logger.info("Building normalized feature cache (raw_ohlcv=%s)...", raw_ohlcv)
    if raw_ohlcv:
        df = compute_raw_ohlcv_features(daily_cache)
    else:
        df = compute_derived_features(daily_cache)

    # Keep only needed feature columns (plus index)
    keep_cols = [c for c in cols if c in df.columns]
    df = df[keep_cols]

    # Drop rows where ALL features are NaN, keep rows with at least some data
    df = df.dropna(how="all")
    # Keep rows with at least half of the features available
    min_non_na = max(1, len(keep_cols) // 2)
    df = df[df.notna().sum(axis=1) >= min_non_na]
    # Fill remaining NaN with 0 (will be z-scored to ~0 after normalization)
    df = df.fillna(0.0)

    # Filter dates with too few stocks for reliable CS stats
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
    df = winsorize_cross_sectional(df, keep_cols, winsor_lower, winsor_upper)
    df = cs_zscore_features(df, keep_cols)

    logger.info("Normalized feature cache: %d rows, %d columns", len(df), len(keep_cols))
    return df.sort_index()


def extract_sequence(
    feature_cache: pd.DataFrame,
    symbol: str,
    date_idx: int,
    all_dates: list,
    seq_length: int = 60,
    feature_cols: list[str] | None = None,
) -> np.ndarray | None:
    """Extract a single (seq_length, n_features) array for one stock at a date index.

    Returns None if the stock doesn't have enough history.
    """
    cols = feature_cols or [c for c in DERIVED_FEATURE_COLUMNS if c in feature_cache.columns]
    seq_start = date_idx - seq_length + 1
    seq_dates = all_dates[seq_start:date_idx + 1]

    try:
        rows = feature_cache.loc[
            (feature_cache.index.get_level_values("trade_date").isin(seq_dates)) &
            (feature_cache.index.get_level_values("symbol") == symbol)
        ][cols]
    except KeyError:
        return None

    if len(rows) < seq_length:
        return None

    # Ensure rows are in chronological order
    rows = rows.sort_index(level="trade_date")
    values = rows.values.astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values
