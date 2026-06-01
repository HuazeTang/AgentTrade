"""V3 feature computation: momentum + consolidation + divergence + per-stock z-score.

Feature sets (total ~24 dims):
  - RAW_OHLCV (8): price returns + log volume/amount (same as V2)
  - CONSOLIDATION (8): volatility contraction, volume drying, amplitude narrowing
  - DIVERGENCE (4): price-volume divergence, OBV, gap ratio

Normalization:
  1. All features: cross-sectional winsorize (1%/99%) + z-score (same as V2)
  2. 4 key features: additional per-stock rolling 60d z-score → concat (4 extra dims)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── V2 raw OHLCV columns (reused) ──────────────────────────────────────────

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

# ── V3 new feature columns ─────────────────────────────────────────────────

CONSOLIDATION_COLUMNS: list[str] = [
    "vol_20d",              # 20d rolling std of daily returns (low = tight range)
    "vol_ratio_5d",         # 5d avg volume / 20d avg volume (< 0.5 = drying up)
    "amplitude_10d",        # 10d average (high-low)/close (contracting)
    "close_position_5d",    # 5d avg close position in daily range (> 0.5 = accumulation)
    "turnover_trend",       # 5d avg turnover / 20d avg turnover
    "excess_return_10d",    # 10d stock return - median market return
    "ret_skew_20d",         # Skewness of 20 daily returns (positive = more up days)
    "volume_cv_20d",        # CV of volume over 20d (low = stable volume)
]

DIVERGENCE_COLUMNS: list[str] = [
    "price_vol_div_10d",    # abs(10d ret) < 5% AND vol_ratio > 1.2 → price flat, volume up
    "obv_divergence_10d",   # OBV slope - price slope over 10d
    "high_vol_fade",        # Max vol day in last 5d was a down day (1) or up day (0)
    "gap_ratio_10d",        # Fraction of days with positive overnight gap in last 10d
]

PER_STOCK_ZSCORE_COLS: list[str] = [
    "close_ret",
    "log_volume",
    "turnover",
    "vol_20d",
]

ALL_V3_COLUMNS = RAW_OHLCV_COLUMNS + CONSOLIDATION_COLUMNS + DIVERGENCE_COLUMNS


# ═════════════════════════════════════════════════════════════════════════════
# Feature Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_raw_ohlcv_features(data: pd.DataFrame) -> pd.DataFrame:
    """Copied from dl.derived_features — compute raw OHLCV returns/logs.

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


def compute_consolidation_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute consolidation / coiling features.

    These capture stocks that are "coiling" — tight range, shrinking volume,
    accumulating quietly before a potential breakout.
    """
    if data.empty:
        return data

    result = data.copy()
    close = data["close"].unstack()
    high = data["high"].unstack() if "high" in data.columns else None
    low = data["low"].unstack() if "low" in data.columns else None
    volume = data["volume"].unstack() if "volume" in data.columns else None
    pre_close = data["pre_close"].unstack() if "pre_close" in data.columns else close.shift(1)
    turnover = data["turnover"].unstack() if "turnover" in data.columns else None

    # ── vol_20d: 20d rolling std of daily returns ──
    daily_ret = close.pct_change(fill_method=None)
    vol_20d_ser = daily_ret.rolling(20, min_periods=10).std().shift(1).stack()
    vol_20d_ser.name = "vol_20d"
    result = result.join(vol_20d_ser, how="left")

    # ── vol_ratio_5d: 5d avg vol / 20d avg vol ──
    if volume is not None:
        v5 = volume.rolling(5, min_periods=3).mean().shift(1)
        v20 = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio_5d_ser = (v5 / v20.clip(lower=1e-8)).stack()
        vol_ratio_5d_ser.name = "vol_ratio_5d"
        result = result.join(vol_ratio_5d_ser, how="left")

    # ── amplitude_10d: 10d avg (high-low)/close ──
    if high is not None and low is not None:
        daily_amp = (high - low) / close.clip(lower=1e-8)
        amp_10d_ser = daily_amp.rolling(10, min_periods=5).mean().shift(1).stack()
        amp_10d_ser.name = "amplitude_10d"
        result = result.join(amp_10d_ser, how="left")

    # ── close_position_5d: 5d avg of close position in daily range ──
    if high is not None and low is not None:
        daily_range = high - low
        cp = (close - low) / daily_range.clip(lower=1e-8)
        cp_5d_ser = cp.rolling(5, min_periods=3).mean().shift(1).stack()
        cp_5d_ser.name = "close_position_5d"
        result = result.join(cp_5d_ser, how="left")

    # ── turnover_trend: 5d avg turnover / 20d avg turnover ──
    if turnover is not None:
        t5 = turnover.rolling(5, min_periods=3).mean().shift(1)
        t20 = turnover.rolling(20, min_periods=10).mean().shift(1)
        turnover_trend_ser = (t5 / t20.clip(lower=1e-8)).stack()
        turnover_trend_ser.name = "turnover_trend"
        result = result.join(turnover_trend_ser, how="left")
    else:
        # Fallback: use volume ratio as proxy
        if "vol_ratio_5d" in result.columns:
            result["turnover_trend"] = result["vol_ratio_5d"]

    # ── excess_return_10d: 10d return - median market return ──
    ret_10d = close.pct_change(10, fill_method=None).shift(1)
    median_ret_10d = ret_10d.median(axis=1)
    excess_10d_ser = ret_10d.sub(median_ret_10d, axis=0).stack()
    excess_10d_ser.name = "excess_return_10d"
    result = result.join(excess_10d_ser, how="left")

    # ── ret_skew_20d: skewness of daily returns over 20d ──
    ret_skew_ser = daily_ret.rolling(20, min_periods=10).skew().shift(1).stack()
    ret_skew_ser.name = "ret_skew_20d"
    result = result.join(ret_skew_ser, how="left")

    # ── volume_cv_20d: CV of volume over 20d ──
    if volume is not None:
        v_mean_20 = volume.rolling(20, min_periods=10).mean().shift(1)
        v_std_20 = volume.rolling(20, min_periods=10).std().shift(1)
        vol_cv_ser = (v_std_20 / v_mean_20.clip(lower=1e-8)).stack()
        vol_cv_ser.name = "volume_cv_20d"
        result = result.join(vol_cv_ser, how="left")

    logger.info("Computed consolidation features: %d columns", len(CONSOLIDATION_COLUMNS))
    return result


def compute_divergence_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute price-volume divergence features.

    These capture disagreement between price and volume — potential
    accumulation or distribution signals.
    """
    if data.empty:
        return data

    result = data.copy()
    close = data["close"].unstack()
    high = data["high"].unstack() if "high" in data.columns else None
    low = data["low"].unstack() if "low" in data.columns else None
    volume = data["volume"].unstack() if "volume" in data.columns else None
    pre_close = data["pre_close"].unstack() if "pre_close" in data.columns else close.shift(1)
    open_ = data["open"].unstack() if "open" in data.columns else None

    # ── price_vol_div_10d: price flat + volume rising ──
    ret_10d = close.pct_change(10, fill_method=None)
    price_flat = ret_10d.abs() < 0.05
    if volume is not None:
        v5 = volume.rolling(5, min_periods=3).mean()
        v20 = volume.rolling(20, min_periods=10).mean()
        vol_rising = v5 > v20 * 1.2
        pv_div = (price_flat & vol_rising).astype(float)
        # Smooth over 5 days and shift
        pv_div_ser = pv_div.rolling(5, min_periods=3).mean().shift(1).stack()
        pv_div_ser.name = "price_vol_div_10d"
        result = result.join(pv_div_ser, how="left")

    # ── obv_divergence_10d: OBV trend - price trend ──
    if volume is not None:
        daily_ret_sign = np.sign(close.pct_change(fill_method=None))
        obv_flow = daily_ret_sign * volume
        obv_cum = obv_flow.cumsum()
        # 10d slope of OBV vs 10d slope of price
        obv_slope_10d = (obv_cum - obv_cum.shift(10)) / 10
        price_slope_10d = (close - close.shift(10)) / 10
        # Normalize by own std for comparability
        obv_slope_norm = obv_slope_10d / obv_slope_10d.rolling(60, min_periods=20).std().clip(lower=1e-8)
        price_slope_norm = price_slope_10d / price_slope_10d.rolling(60, min_periods=20).std().clip(lower=1e-8)
        obv_div = (obv_slope_norm - price_slope_norm).shift(1).stack()
        obv_div.name = "obv_divergence_10d"
        result = result.join(obv_div, how="left")

    # ── high_vol_fade: max volume day in last 5d was down ──
    if volume is not None:
        daily_ret_sign2 = np.sign(close.pct_change(fill_method=None))
        max_vol = None
        max_vol_sign = None
        for lookback in range(1, 6):
            vs = volume.shift(lookback).fillna(0)
            rs = daily_ret_sign2.shift(lookback).fillna(0)
            if max_vol is None:
                max_vol = vs
                max_vol_sign = rs
            else:
                better = vs > max_vol
                max_vol = max_vol.mask(better, vs)
                max_vol_sign = max_vol_sign.mask(better, rs)
        fade = (max_vol_sign < 0).astype(float).shift(1).stack()
        fade.name = "high_vol_fade"
        result = result.join(fade, how="left")

    # ── gap_ratio_10d: fraction of days with positive overnight gap ──
    if open_ is not None and pre_close is not None:
        overnight = (open_ - pre_close.shift(1)) / pre_close.shift(1).clip(lower=1e-8)
        pos_gap = (overnight > 0.01).astype(float)  # >1% gap up
        gap_ratio_ser = pos_gap.rolling(10, min_periods=5).mean().shift(1).stack()
        gap_ratio_ser.name = "gap_ratio_10d"
        result = result.join(gap_ratio_ser, how="left")

    logger.info("Computed divergence features: %d columns", len(DIVERGENCE_COLUMNS))
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

def build_v3_feature_cache(
    daily_cache: pd.DataFrame,
    *,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    min_cs_stocks: int = 30,
    per_stock_zscore: bool = True,
    per_stock_window: int = 60,
) -> pd.DataFrame:
    """Build the full V3 feature cache.

    Pipeline:
      1. Compute raw OHLCV features (8 dims)
      2. Compute consolidation features (8 dims)
      3. Compute divergence features (4 dims)
      4. Winsorize + CS z-score all features
      5. Optionally add per-stock rolling z-score for key features (+4 dims)

    Returns:
        MultiIndex DataFrame (trade_date, symbol) with ~24 normalized feature columns.
    """
    logger.info("Building V3 feature cache...")

    # Step 1-3: Compute all features
    df = compute_raw_ohlcv_features(daily_cache)
    df = compute_consolidation_features(df)
    df = compute_divergence_features(df)

    cs_cols = [c for c in ALL_V3_COLUMNS if c in df.columns]
    logger.info("Total V3 features: %d columns", len(cs_cols))

    # Keep only feature columns
    df = df[cs_cols]

    # Drop rows where ALL features are NaN
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

    # Step 4: Winsorize + CS z-score
    df = winsorize_cross_sectional(df, cs_cols, winsor_lower, winsor_upper)
    df = cs_zscore_features(df, cs_cols)

    # Step 5: Per-stock z-score for key features
    if per_stock_zscore:
        key_cols = [c for c in PER_STOCK_ZSCORE_COLS if c in df.columns]
        df = compute_per_stock_zscore(df, key_cols, rolling_window=per_stock_window)
        ts_cols = [f"{c}_ts" for c in key_cols]
        total_cols = cs_cols + [c for c in ts_cols if c in df.columns]
        logger.info("V3 feature cache: %d rows, %d columns (with per-stock z-score)",
                     len(df), len(total_cols))
    else:
        logger.info("V3 feature cache: %d rows, %d columns", len(df), len(cs_cols))

    return df.sort_index()
