"""Data quality checks for A-share market data."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PRICE_LIMITS

logger = logging.getLogger(__name__)


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing rate report."""
    return df.isnull().mean().rename("missing_rate").to_frame()


def check_price_limits(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where the daily price change exceeds the board's limit.

    Expects columns: close, pre_close, price_limit_frac.
    """
    if "pre_close" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    if "price_limit_frac" not in df.columns:
        limit = 0.10
    else:
        limit = df.get("price_limit_frac", 0.10)

    change_pct = (df["close"] - df["pre_close"]) / df["pre_close"].abs()
    violation = change_pct.abs() > limit + 0.005  # 0.5% tolerance
    return df.loc[violation, ["close", "pre_close"]].assign(
        change_pct=change_pct[violation], limit=limit
    )


def check_suspensions(df: pd.DataFrame) -> int:
    """Count rows flagged as suspended (volume == 0 or is_suspended == True)."""
    if "is_suspended" in df.columns:
        return int(df["is_suspended"].sum())
    if "volume" in df.columns:
        return int((df["volume"] <= 0).sum())
    return 0


def check_st_stocks(df: pd.DataFrame) -> int:
    """Count unique ST stocks."""
    if "is_st" not in df.columns:
        return 0
    return int(df["is_st"].sum())


def generate_report(df: pd.DataFrame) -> str:
    """Generate a human-readable data quality report."""
    lines = []
    lines.append("=" * 50)
    lines.append("Data Quality Report")
    lines.append("=" * 50)

    if df.empty:
        lines.append("Empty dataset.")
        return "\n".join(lines)

    idx_names = df.index.names if isinstance(df.index, pd.MultiIndex) else None
    if idx_names and "trade_date" in idx_names and "symbol" in idx_names:
        dates = df.index.get_level_values("trade_date")
        symbols = df.index.get_level_values("symbol")
        lines.append(f"Date range: {dates.min().date()} ~ {dates.max().date()}")
        lines.append(f"Trading days: {dates.nunique()}")
        lines.append(f"Unique symbols: {symbols.nunique()}")
        lines.append(f"Total rows: {len(df)}")

    missing = check_missing(df)
    if not missing.empty and missing["missing_rate"].max() > 0:
        lines.append("\nMissing rates:")
        for col, rate in missing[missing["missing_rate"] > 0].itertuples():
            lines.append(f"  {col}: {rate:.2%}")

    limit_violations = check_price_limits(df)
    if len(limit_violations) > 0:
        lines.append(f"\nPrice limit violations: {len(limit_violations)}")

    suspended = check_suspensions(df)
    if suspended > 0:
        lines.append(f"Suspended records: {suspended}")

    st_count = check_st_stocks(df)
    if st_count > 0:
        lines.append(f"ST records: {st_count}")

    return "\n".join(lines)
