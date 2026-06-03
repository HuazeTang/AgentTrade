"""Download A-share OHLCV data via tushare and rebuild adjusted caches.

Pipeline: raw → merge adj_factor → raw_to_badj → daily_badj cache.

Usage:
    python scripts/download_data.py --start 2026-04-20 --end 2026-06-01
    python scripts/download_data.py --start 2026-05-01  # single date → same start/end
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DATA_DIR
from data.adjust import raw_to_badj, write_partitioned
from data.cache import read_daily, write_daily
from data.sources.tushare import TushareSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_stock_pool() -> pd.DataFrame:
    """Load stock_list.parquet, filter to non-ST main-board A-shares."""
    path = DATA_DIR / "stock_list.parquet"
    if not path.exists():
        raise FileNotFoundError(f"stock_list.parquet not found at {path}")
    info = pd.read_parquet(path)
    st_mask = info["name"].str.contains(r"\*?ST", na=False)
    info = info[~st_mask].copy()
    logger.info("Stock pool: %d non-ST symbols", len(info))
    return info


def download_range(start: date, end: date) -> dict:
    """Download raw + adj_factor, build badj, write to cache. Returns stats."""
    info = load_stock_pool()
    symbols = sorted(info["symbol"].unique().tolist())

    source = TushareSource()

    # ── 1. Download raw daily ──
    logger.info("=== Step 1: Fetch raw daily %s ~ %s ===", start, end)
    raw = source.fetch_daily(symbols, start, end)
    if raw.empty:
        logger.error("No raw data fetched!")
        return {"raw": 0, "badj": 0}

    raw_dates = sorted(raw.index.get_level_values("trade_date").unique())
    raw_syms = raw.index.get_level_values("symbol").nunique()
    logger.info("Raw daily: %d rows, %d dates, %d symbols",
                 len(raw), len(raw_dates), raw_syms)

    # ── 2. Save raw to daily_raw cache ──
    logger.info("=== Step 2: Write raw to daily_raw cache ===")
    write_daily(raw.reset_index(), prefix="daily_raw", merge=True)

    # ── 3. Download adj_factor ──
    logger.info("=== Step 3: Fetch adj_factor %s ~ %s ===", start, end)
    adj = source.fetch_adj_factor(symbols, start, end)
    logger.info("Adj factors fetched: %d rows", len(adj))

    # ── 4. Merge adj_factor into raw ──
    logger.info("=== Step 4: Merge adj_factor into raw ===")
    raw_df = raw.reset_index()
    # Keep only needed adj_factor columns
    if not adj.empty:
        adj_merge = adj[["symbol", "trade_date", "adj_factor"]].copy()
        adj_merge["trade_date"] = pd.to_datetime(adj_merge["trade_date"])
        # Drop placeholder adj_factor from raw
        if "adj_factor" in raw_df.columns:
            raw_df = raw_df.drop(columns=["adj_factor"])
        raw_df = raw_df.merge(adj_merge, on=["symbol", "trade_date"], how="left")
        raw_df["adj_factor"] = raw_df["adj_factor"].fillna(1.0)
    else:
        raw_df["adj_factor"] = 1.0

    # ── 5. Update adj_factor.parquet ──
    cache_adj = DATA_DIR / "adj_factor.parquet"
    if not adj.empty and cache_adj.exists():
        existing = pd.read_parquet(cache_adj)
        existing["trade_date"] = pd.to_datetime(existing["trade_date"])
        adj_save = adj[["symbol", "trade_date", "adj_factor"]].copy()
        adj_save["trade_date"] = pd.to_datetime(adj_save["trade_date"])
        merged = pd.concat([existing, adj_save], ignore_index=True)
        merged = merged.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        merged.to_parquet(cache_adj, index=False)
        logger.info("Updated adj_factor.parquet: %d → %d rows", len(existing), len(merged))

    # ── 6. raw → badj ──
    logger.info("=== Step 5: raw_to_badj ===")
    raw_indexed = raw_df.set_index(["trade_date", "symbol"]).sort_index()
    badj = raw_to_badj(raw_indexed)
    logger.info("Badj: %d rows", len(badj))

    # ── 7. Write badj to daily_badj cache ──
    logger.info("=== Step 6: Write badj to daily_badj cache ===")
    write_daily(badj.reset_index(), prefix="daily_badj", merge=True)

    # ── 8. Verify ──
    verify = read_daily(start, end, prefix="daily_badj")
    v_dates = sorted(verify.index.get_level_values("trade_date").unique())
    logger.info("Verification: %d rows, %d dates (%s ~ %s) in daily_badj",
                 len(verify), len(v_dates),
                 v_dates[0].date() if v_dates else "N/A",
                 v_dates[-1].date() if v_dates else "N/A")

    # Spot check: first symbol, last date
    sample = verify.xs(v_dates[-1], level="trade_date").head(3)
    for sym, row in sample.iterrows():
        pc = row.get("pre_close", float("nan"))
        chg = (row["close"] - pc) / pc * 100 if pc and pc > 0 else 0
        logger.info("  %s: open=%.2f close=%.2f chg=%+.2f%% vol=%.0f",
                     sym, row["open"], row["close"], chg, row["volume"])

    return {"raw": len(raw), "badj": len(badj), "dates": len(raw_dates)}


def main():
    p = argparse.ArgumentParser(description="Download A-share data via tushare")
    p.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: same as start)")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else start

    logger.info("Downloading %s ~ %s via tushare...", start, end)
    stats = download_range(start, end)

    print(f"\nDone. Raw: {stats['raw']} rows, Badj: {stats['badj']} rows, {stats['dates']} dates")
    print(f"Cache: {DATA_DIR / 'daily_badj'}")


if __name__ == "__main__":
    main()
