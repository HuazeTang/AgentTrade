"""Orchestrated data ingestion pipeline."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from data.cache import read_daily, write_daily, data_summary
from data.calendar import get_trading_days
from data.sources.base import DataSource
from data.sources.baostock import BaoStockSource

logger = logging.getLogger(__name__)


def ingest_daily(
    symbols: list[str],
    start: date,
    end: date,
    source: DataSource | None = None,
    chunk_size: int = 30,
) -> pd.DataFrame:
    """Fetch daily data and cache it to parquet.

    Uses chunked fetching to avoid akshare rate limits.
    Reads existing cache once, then only fetches uncached symbols from source.
    """
    if source is None:
        source = BaoStockSource()

    all_frames: list[pd.DataFrame] = []

    # Read existing cache once (not per-chunk) to avoid massive repeated I/O
    existing = read_daily(start, end, symbols)
    if not existing.empty:
        all_frames.append(existing)

    # Determine which symbols still need fetching
    if not existing.empty:
        if isinstance(existing.index, pd.MultiIndex):
            cached_syms = set(existing.index.get_level_values("symbol").unique())
        else:
            cached_syms = set(existing["symbol"].unique())
        to_fetch = [s for s in symbols if s not in cached_syms]
    else:
        to_fetch = list(symbols)

    if to_fetch:
        logger.info("Fetching %d uncached symbols...", len(to_fetch))
    for i in range(0, len(to_fetch), chunk_size):
        chunk = to_fetch[i : i + chunk_size]
        logger.info("Fetching symbols %d-%d/%d ...", i, min(i + chunk_size, len(to_fetch)), len(to_fetch))
        df = source.fetch_daily(chunk, start, end)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No data fetched for %d symbols in %s ~ %s", len(symbols), start, end)
        return pd.DataFrame()

    result = pd.concat(all_frames)
    # Deduplicate: keep the last occurrence (freshly fetched wins)
    if isinstance(result.index, pd.MultiIndex):
        result = result[~result.index.duplicated(keep="last")]
    else:
        result = result.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
    result = result.sort_index()

    # Merge with full existing cache to preserve symbols not in this batch
    result = _merge_with_full_cache(result, start, end)

    write_daily(result)
    logger.info("Cached %d rows to daily parquet.", len(result))
    return result


def _merge_with_full_cache(
    new_data: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Merge new data with the full existing cache to avoid losing symbols."""
    full = read_daily(start, end)
    if full.empty:
        return new_data
    merged = pd.concat([full, new_data])
    if isinstance(merged.index, pd.MultiIndex):
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = merged.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
    return merged.sort_index()


def ensure_data(
    symbols: list[str],
    start: date,
    end: date,
    source: DataSource | None = None,
) -> pd.DataFrame:
    """Ensure data is available in cache; fetch missing as needed."""
    cached = read_daily(start, end, symbols)
    cached_syms = set()
    if not cached.empty:
        symbols_level = cached.index.names
        if "symbol" in symbols_level:
            cached_syms = set(cached.index.get_level_values("symbol").unique())
        elif "symbol" in cached.columns:
            cached_syms = set(cached["symbol"].unique())

    missing = [s for s in symbols if s not in cached_syms]
    if missing:
        logger.info("Fetching %d uncached symbols...", len(missing))
        fetched = ingest_daily(missing, start, end, source=source)
        if not fetched.empty:
            result = pd.concat([cached, fetched])
            result = result[~result.index.duplicated(keep="last")]
            result = result.sort_index()
            return result

    return cached
