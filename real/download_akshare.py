"""Download main board A-share data using akshare (HTTP).

More reliable than baostock's raw TCP. Handles rate limiting and retries.

Usage: python download_akshare.py
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download")

START_DATE = date(2024, 5, 18)
END_DATE = date(2026, 5, 18)
SAVE_EVERY = 50
RATE_LIMIT = 0.3  # seconds between akshare API calls


def infer_board(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("688"):
        return "star_market"
    if code.startswith("300") or code.startswith("301"):
        return "chinext"
    if code.startswith("8") or code.startswith("4"):
        return "beijing"
    return "main_board"


def fetch_one_stock_ak(sym: str, start: str, end: str, max_retries: int = 4):
    """Fetch daily data for a single stock via akshare."""
    import akshare as ak

    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=sym, period="daily",
                start_date=start, end_date=end,
                adjust="qfq",
            )
        except Exception as e:
            wait = min(2 ** attempt, 30)
            if attempt < max_retries - 1:
                logger.debug("akshare fetch failed for %s: %s, retrying in %ds", sym, e, wait)
                time.sleep(wait)
                continue
            return None

        if df is None or df.empty:
            return None

        # Rename columns to match cache schema
        df = df.rename(columns={
            "日期": "trade_date",
            "股票代码": "symbol",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        })

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["symbol"] = str(sym).zfill(6)

        for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Infer fields
        df["pre_close"] = df.groupby("symbol")["close"].shift(1)
        df["adj_factor"] = 1.0  # qfq already adjusted
        df["is_suspended"] = df["volume"] == 0
        df["is_st"] = False
        board = infer_board(sym)
        df["board"] = board
        df["price_limit_frac"] = 0.10
        if board != "main_board":
            df["price_limit_frac"] = 0.20
        df["market_cap"] = np.nan
        df["tradable_shares"] = np.nan

        keep = [
            "trade_date", "symbol", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "adj_factor", "turnover",
            "is_suspended", "is_st", "price_limit_frac", "board",
            "market_cap", "tradable_shares",
        ]
        return df[keep]

    return None


def write_partitioned(df: pd.DataFrame, base_dir: Path):
    """Write DataFrame to year/month partitioned parquet, merging with existing."""
    if df.empty:
        return
    work = df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for (year, month), group in work.groupby(
        [work["trade_date"].dt.year, work["trade_date"].dt.month]
    ):
        out_dir = base_dir / f"year={year}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{year}{month:02d}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, group], ignore_index=True)
            merged = merged.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
            merged.sort_values(["trade_date", "symbol"]).to_parquet(path, index=False)
        else:
            group.to_parquet(path, index=False)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("  akshare 模式: 逐只下载主板A股数据")
    print(f"  区间: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    start_str = START_DATE.strftime("%Y%m%d")
    end_str = END_DATE.strftime("%Y%m%d")

    # Load stock list
    stock_list_path = Path("data/cache/stock_list.parquet")
    all_stocks = pd.read_parquet(stock_list_path)
    main = all_stocks[(all_stocks["board"] == "main_board") & (~all_stocks["is_st"])]
    syms = sorted(main["symbol"].tolist())
    print(f"主板非ST: {len(syms)} 只 (全A共 {len(all_stocks)} 只)")

    # Check already cached
    cache_dir = Path("data/cache/daily")
    already_cached = set()
    if cache_dir.exists():
        for pq in cache_dir.rglob("*.parquet"):
            try:
                meta = pd.read_parquet(pq, columns=["symbol"])
                already_cached.update(meta["symbol"].unique())
            except Exception:
                pass

    to_fetch = [s for s in syms if s not in already_cached]
    print(f"已缓存: {len(already_cached)} 只, 待下载: {len(to_fetch)} 只")

    if not to_fetch:
        print("全部已缓存!")
        sys.exit(0)

    # Download
    buffer: list[pd.DataFrame] = []
    done = 0
    failed = 0
    failed_syms: list[str] = []
    t_start = time.time()
    last_checkpoint = time.time()
    last_call = 0.0

    for idx, sym in enumerate(to_fetch):
        # Rate limit
        elapsed_since_call = time.time() - last_call
        if elapsed_since_call < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed_since_call)
        last_call = time.time()

        t_sym = time.time()
        df = fetch_one_stock_ak(sym, start_str, end_str)

        if df is not None and not df.empty:
            buffer.append(df)
            done += 1
        else:
            failed += 1
            failed_syms.append(sym)

        total = done + failed
        if total % 10 == 0 or idx == len(to_fetch) - 1:
            elapsed = time.time() - t_start
            rate = total / elapsed * 60 if elapsed > 0 else 0
            pct = total / len(to_fetch) * 100
            eta = (len(to_fetch) - total) / rate if rate > 0 else 0
            logger.info("进度: %d/%d (%.1f%%) | 成功: %d | 失败: %d | %.1f只/分 | 剩余: %.0f分",
                       total, len(to_fetch), pct, done, failed, rate, eta)

        # Flush
        if len(buffer) >= SAVE_EVERY or idx == len(to_fetch) - 1:
            if buffer:
                merged = pd.concat(buffer, ignore_index=True)
                write_partitioned(merged, cache_dir)
                buffer.clear()
                logger.info("写入 %d 行 (%.0fs)", len(merged), time.time() - last_checkpoint)
                last_checkpoint = time.time()

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"下载完成! 成功: {done}, 失败: {failed}, 耗时: {total_elapsed/60:.1f}min")
    if failed_syms:
        print(f"失败股票(前30): {failed_syms[:30]}")

    # Stats
    if cache_dir.exists():
        files = list(cache_dir.rglob("*.parquet"))
        total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"缓存: {len(files)} 文件, {total_mb:.1f} MB")
    print("=" * 60)
