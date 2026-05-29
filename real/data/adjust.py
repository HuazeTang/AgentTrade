"""Price adjustment utilities: raw → badj/qfq from adj_factor.

Core formula (verified): 后复权 = raw_price × adj_factor

Usage:
    from data.adjust import raw_to_badj, rebuild_adjusted
    badj = raw_to_badj(raw_df)
    rebuild_adjusted()  # reads daily_raw, writes daily_badj + daily_qfq
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close", "pre_close"]


def raw_to_badj(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw (未复权) OHLCV to 后复权 (backward-adjusted).

    badj_price = raw_price × adj_factor

    Returns a copy; does not modify the input.
    """
    result = df.copy()
    factor = result["adj_factor"]
    for col in PRICE_COLS:
        if col in result.columns:
            result[col] = result[col] * factor
    return result


def raw_to_qfq(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw (未复权) OHLCV to 前复权 (forward-adjusted).

    qfq_price = raw_price / adj_factor

    Returns a copy; does not modify the input.
    """
    result = df.copy()
    factor = result["adj_factor"].clip(lower=1e-8)
    for col in PRICE_COLS:
        if col in result.columns:
            result[col] = result[col] / factor
    return result


def _merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new data with existing parquet file, deduplicating on (trade_date, symbol)."""
    if path.exists():
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
        return merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return new_df


def write_partitioned(df: pd.DataFrame, prefix_dir: Path) -> None:
    """Write DataFrame to year/month partitioned parquet, merging with existing files.

    This is the canonical write function for the daily cache — it reads the
    existing parquet for each year/month partition, concatenates, deduplicates,
    and writes back.  Safe for incremental updates.
    """
    if df.empty:
        return
    work = df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for (year, month), group in work.groupby(
        [work["trade_date"].dt.year, work["trade_date"].dt.month]
    ):
        out_dir = prefix_dir / f"year={year}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{year}{month:02d}.parquet"
        merged = _merge_with_existing(group, path)
        merged.to_parquet(path, index=False)


def rebuild_adjusted(
    raw_prefix: str = "daily_raw",
    badj_prefix: str = "daily_badj",
    qfq_prefix: str = "daily_qfq",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Rebuild daily_badj and daily_qfq from daily_raw + adj_factor.

    Reads raw data from DATA_DIR/<raw_prefix>, computes both adjusted versions,
    and writes them to DATA_DIR/<badj_prefix> and DATA_DIR/<qfq_prefix>.

    Args:
        raw_prefix: subdirectory name for raw (未复权) cache.
        badj_prefix: subdirectory for 后复权 output.
        qfq_prefix: subdirectory for 前复权 output.
        start_date, end_date: optional date filter (YYYY-MM-DD).

    Returns:
        dict with row counts: {"badj": int, "qfq": int}
    """
    raw_dir = DATA_DIR / raw_prefix
    if not raw_dir.exists():
        logger.warning("Raw cache directory %s does not exist", raw_dir)
        return {"badj": 0, "qfq": 0}

    badj_dir = DATA_DIR / badj_prefix
    qfq_dir = DATA_DIR / qfq_prefix

    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    logger.info("Rebuilding adjusted data from %d parquet files in %s", len(parquet_files), raw_dir)

    badj_count = 0
    qfq_count = 0

    for pq_file in parquet_files:
        raw = pd.read_parquet(pq_file)
        if raw.empty:
            continue

        # Date filtering
        if start_date or end_date:
            if "trade_date" not in raw.columns:
                continue
            raw["trade_date"] = pd.to_datetime(raw["trade_date"])
            if start_date:
                raw = raw[raw["trade_date"] >= start_date]
            if end_date:
                raw = raw[raw["trade_date"] <= end_date]
            if raw.empty:
                continue

        # Build adjusted versions
        badj = raw_to_badj(raw)
        qfq = raw_to_qfq(raw)

        # Write to badj
        badj_path = badj_dir / pq_file.relative_to(raw_dir)
        badj_path.parent.mkdir(parents=True, exist_ok=True)
        badj_merged = _merge_with_existing(badj, badj_path)
        badj_merged.to_parquet(badj_path, index=False)

        # Write to qfq
        qfq_path = qfq_dir / pq_file.relative_to(raw_dir)
        qfq_path.parent.mkdir(parents=True, exist_ok=True)
        qfq_merged = _merge_with_existing(qfq, qfq_path)
        qfq_merged.to_parquet(qfq_path, index=False)

        badj_count += len(badj)
        qfq_count += len(qfq)

    logger.info("Rebuilt: %d badj rows, %d qfq rows", badj_count, qfq_count)
    return {"badj": badj_count, "qfq": qfq_count}
