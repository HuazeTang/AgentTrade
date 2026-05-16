"""Parquet-based caching for market data, partitioned by year/month."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

_COLUMNS = [
    "trade_date", "symbol", "open", "high", "low", "close",
    "pre_close", "volume", "amount", "adj_factor", "turnover",
    "is_suspended", "is_st", "price_limit_frac", "board",
    "market_cap", "tradable_shares",
]


def _partition_path(prefix: str, dt: datetime | pd.Timestamp) -> Path:
    return DATA_DIR / prefix / f"year={dt.year}" / f"month={dt.month:02d}"


def write_daily(df: pd.DataFrame, prefix: str = "daily") -> None:
    """Write daily data partitioned by date. Accepts either a flat
    DataFrame with trade_date/symbol columns or a multi-index."""
    if df.empty:
        return
    work = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    if "trade_date" not in work.columns:
        raise ValueError("DataFrame must have a 'trade_date' column")
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for (year, month), group in work.groupby(
        [work["trade_date"].dt.year, work["trade_date"].dt.month]
    ):
        out_dir = DATA_DIR / prefix / f"year={year}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{year}{month:02d}.parquet"
        # Remove partition columns from data
        save = group.drop(columns=[c for c in group.columns if c.startswith("year=")], errors="ignore")
        save.to_parquet(path, index=False)


def read_daily(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    prefix: str = "daily",
) -> pd.DataFrame:
    """Read daily data for a date range, optionally filtered by symbol."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []

    base = DATA_DIR / prefix
    if not base.exists():
        return pd.DataFrame()

    for year_dir in sorted(base.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        for month_dir in sorted(year_dir.glob("month=*")):
            month = int(month_dir.name.split("=")[1])
            month_first = pd.Timestamp(year=year, month=month, day=1)
            # Compute last day of month
            if month == 12:
                month_last = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                month_last = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
            if month_last < start_ts or month_first > end_ts:
                continue
            for pq_file in month_dir.glob("*.parquet"):
                chunk = pd.read_parquet(pq_file)
                if symbols:
                    chunk = chunk[chunk["symbol"].isin(symbols)]
                if "trade_date" in chunk.columns:
                    chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
                    chunk = chunk[
                        (chunk["trade_date"] >= start_ts)
                        & (chunk["trade_date"] <= end_ts)
                    ]
                if not chunk.empty:
                    frames.append(chunk)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "trade_date" in result.columns and "symbol" in result.columns:
        result = result.set_index(["trade_date", "symbol"]).sort_index()
    return result


def data_summary(prefix: str = "daily") -> dict:
    """Return summary stats about cached data."""
    base = DATA_DIR / prefix
    if not base.exists():
        return {"dates": (None, None), "symbols": 0, "rows": 0}

    all_files = list(base.glob("**/*.parquet"))
    if not all_files:
        return {"dates": (None, None), "symbols": 0, "rows": 0}

    dates: list[pd.Timestamp] = []
    symbols: set[str] = set()
    total_rows = 0

    for f in all_files:
        chunk = pd.read_parquet(f, columns=["trade_date", "symbol"])
        if "trade_date" in chunk.columns:
            dates.extend(pd.to_datetime(chunk["trade_date"]).tolist())
        if "symbol" in chunk.columns:
            symbols.update(chunk["symbol"].unique())
        total_rows += len(chunk)

    if dates:
        return {
            "dates": (min(dates).date(), max(dates).date()),
            "symbols": len(symbols),
            "rows": total_rows,
        }
    return {"dates": (None, None), "symbols": 0, "rows": 0}
