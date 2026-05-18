"""Robust stock-by-stock download of main board A-share data.

Fetches 2 years of daily OHLCV for all main board non-ST stocks.
Handles connection errors, timeouts, and retries gracefully.
Saves each stock individually then merges into partitioned parquet.

Usage: python download_robust.py
"""

from __future__ import annotations

import logging
import socket
import sys
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
SAVE_EVERY = 50  # flush to disk every N stocks

# ---- baostock helpers ----

def bs_login(max_retries: int = 5):
    import baostock as bs
    import baostock.common.context as conx

    for attempt in range(max_retries):
        socket.setdefaulttimeout(10)
        lg = bs.login()
        if lg.error_code == "0":
            if hasattr(conx, "default_socket"):
                sock_obj = getattr(conx, "default_socket")
                if sock_obj is not None:
                    try:
                        sock_obj.settimeout(5)
                    except Exception:
                        pass
            return
        wait = min(2 ** attempt, 60)
        logger.warning("login failed (attempt %d/%d): %s, retrying in %ds...",
                       attempt + 1, max_retries, lg.error_msg, wait)
        time.sleep(wait)
    raise RuntimeError(f"baostock login failed after {max_retries} attempts")

def bs_logout():
    import baostock as bs
    try:
        bs.logout()
    except Exception:
        pass

def safe_reconnect():
    """Reconnect with exponential backoff on failure."""
    for attempt in range(5):
        try:
            bs_logout()
            time.sleep(1)
            bs_login()
            return
        except Exception as e:
            wait = min(2 ** attempt, 60)
            logger.warning("reconnect failed (attempt %d/5): %s, retrying in %ds...",
                           attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError("Unable to reconnect to baostock")

def to_bs_code(symbol: str) -> str:
    symbol = str(symbol).strip().zfill(6)
    prefix = "sh" if symbol.startswith("6") else "sz"
    return f"{prefix}.{symbol}"

def infer_board(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("688"):
        return "star_market"
    if code.startswith("300") or code.startswith("301"):
        return "chinext"
    if code.startswith("8") or code.startswith("4"):
        return "beijing"
    return "main_board"

def fetch_one_stock(sym: str, start: str, end: str, max_retries: int = 3):
    """Fetch daily data for a single stock with retries."""
    import baostock as bs

    bs_code = to_bs_code(sym)
    fields = "date,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST"

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, fields,
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",
            )
        except Exception as e:
            logger.debug("query exception for %s: %s", sym, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                try:
                    safe_reconnect()
                except Exception:
                    pass
                continue
            return None

        if rs is None or rs.error_code != "0":
            err_msg = getattr(rs, 'error_msg', rs.error_code if rs else 'None')
            logger.debug("query error for %s: %s", sym, err_msg)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                try:
                    safe_reconnect()
                except Exception:
                    pass
                continue
            return None

        rows = []
        try:
            while rs.next():
                rows.append(rs.get_row_data())
                if time.time() - t0 > 90:  # per-stock deadline
                    rows = []
                    break
        except (socket.timeout, OSError, TimeoutError, Exception) as e:
            logger.debug("row iteration error for %s: %s", sym, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                try:
                    safe_reconnect()
                except Exception:
                    pass
                continue
            return None

        if not rows:
            # Empty result - could be delisted or no trading in range
            return None

        raw = pd.DataFrame(rows, columns=rs.fields)
        raw = raw.rename(columns={
            "date": "trade_date",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "preclose": "pre_close",
            "volume": "volume", "amount": "amount", "turn": "turnover",
        })

        for col in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        raw["symbol"] = sym
        raw["adj_factor"] = 1.0
        raw["is_suspended"] = raw.get("tradestatus", "1") == "0"
        raw["is_st"] = raw.get("isST", "0") == "1"
        board = infer_board(sym)
        raw["board"] = board
        raw["price_limit_frac"] = 0.10 if board == "main_board" else (0.20 if board in ("star_market", "chinext") else 0.30)
        st_mask = raw["is_st"]
        if st_mask.any():
            raw.loc[st_mask, "price_limit_frac"] = 0.05
        raw["market_cap"] = np.nan
        raw["tradable_shares"] = np.nan

        keep = [
            "trade_date", "symbol", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "adj_factor", "turnover",
            "is_suspended", "is_st", "price_limit_frac", "board",
            "market_cap", "tradable_shares",
        ]
        return raw[keep]

    return None


def write_partitioned(df: pd.DataFrame, base_dir: Path):
    """Write DataFrame to year/month partitioned parquet (merge with existing)."""
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


# ---- main ----

if __name__ == "__main__":
    print("=" * 60)
    print("  稳健模式: 逐只下载主板A股数据")
    print(f"  区间: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    # Get stock list
    bs_login()

    # Load from saved stock list if available
    stock_list_path = Path("data/cache/stock_list.parquet")
    if stock_list_path.exists():
        all_stocks = pd.read_parquet(stock_list_path)
        print(f"从缓存加载股票列表: {len(all_stocks)} 只")
    else:
        import baostock as bs
        today = date.today()
        for offset in range(5):
            query_date = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            rs = bs.query_all_stock(day=query_date)
            rows_data = []
            while (rs.error_code == "0") and rs.next():
                rows_data.append(rs.get_row_data())
            if rows_data:
                df = pd.DataFrame(rows_data, columns=rs.fields)
                df = df[df["code"].str.match(r"^(sh\.6|sz\.00|sz\.300|sz\.301|bj\.)")].copy()
                all_stocks = pd.DataFrame()
                all_stocks["symbol"] = df["code"].apply(lambda x: x.split(".")[-1])
                all_stocks["name"] = df["code_name"].astype(str)
                all_stocks["board"] = all_stocks["symbol"].apply(infer_board)
                all_stocks["is_st"] = all_stocks["name"].str.contains("ST|\\*ST", case=True, na=False)
                all_stocks["list_date"] = pd.NaT
                all_stocks["delist_date"] = pd.NaT
                all_stocks.to_parquet(stock_list_path, index=False)
                break

    # Filter main board non-ST
    main = all_stocks[(all_stocks["board"] == "main_board") & (~all_stocks["is_st"])]
    syms = sorted(main["symbol"].tolist())
    print(f"主板非ST: {len(syms)} 只 (全A共 {len(all_stocks)} 只)")

    # Check which are already cached
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
        bs_logout()
        sys.exit(0)

    # Download stock by stock
    buffer: list[pd.DataFrame] = []
    done = 0
    failed = 0
    failed_syms: list[str] = []
    t_start = time.time()
    last_checkpoint = time.time()

    for idx, sym in enumerate(to_fetch):
        t_sym_start = time.time()
        df = fetch_one_stock(sym, start_str, end_str)

        if df is not None and not df.empty:
            buffer.append(df)
            done += 1
            elapsed = time.time() - t_sym_start
            if elapsed > 5:
                logger.debug("%s took %.1fs (%d rows)", sym, elapsed, len(df))
        else:
            failed += 1
            failed_syms.append(sym)

        # Progress report
        total_processed = done + failed
        if total_processed % 10 == 0 or idx == len(to_fetch) - 1:
            elapsed_total = time.time() - t_start
            rate = total_processed / elapsed_total if elapsed_total > 0 else 0
            pct = total_processed / len(to_fetch) * 100
            eta = (len(to_fetch) - total_processed) / rate if rate > 0 else 0
            logger.info(
                "进度: %d/%d (%.1f%%) | 成功: %d | 失败: %d | 速率: %.1f只/分 | 预计剩余: %.0f分",
                total_processed, len(to_fetch), pct,
                done, failed, rate * 60, eta / 60,
            )

        # Periodic reconnect (every 200 stocks)
        if total_processed > 0 and total_processed % 200 == 0:
            logger.info("定期重连...")
            try:
                safe_reconnect()
            except Exception:
                logger.warning("定期重连失败，尝试继续...")

        # Flush buffer to disk
        if len(buffer) >= SAVE_EVERY or idx == len(to_fetch) - 1:
            if buffer:
                merged = pd.concat(buffer, ignore_index=True)
                write_partitioned(merged, cache_dir)
                rows_written = len(merged)
                buffer.clear()
                flush_time = time.time() - last_checkpoint
                logger.info("已写入 %d 行到磁盘 (%.0fs)", rows_written, flush_time)
                last_checkpoint = time.time()

        # Small delay between stocks to avoid overwhelming baostock
        time.sleep(0.02)

    # Final stats
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"成功: {done} 只, 失败: {failed} 只")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")

    # Show failed symbols
    if failed_syms:
        print(f"失败股票 (前20): {failed_syms[:20]}")

    # Disk usage
    if cache_dir.exists():
        total_mb = sum(f.stat().st_size for f in cache_dir.rglob("*.parquet")) / 1024 / 1024
        total_files = len(list(cache_dir.rglob("*.parquet")))
        print(f"缓存: {total_files} 个文件, {total_mb:.1f} MB")

        # Row count
        total_rows = 0
        for pq in cache_dir.rglob("*.parquet"):
            try:
                total_rows += len(pd.read_parquet(pq, columns=["symbol"]))
            except Exception:
                pass
        print(f"总行数: {total_rows:,}")

    bs_logout()
    print("=" * 60)
