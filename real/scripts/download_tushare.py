"""Download all main-board A-share daily data (2010-2026) via Tushare Pro.

Outputs two partitioned parquet datasets:
  data/cache/daily_raw/  — unadjusted (不复权) OHLCV + adj_factor
  data/cache/daily_badj/ — back-adjusted (后复权) OHLCV = raw × adj_factor

Usage: python scripts/download_tushare.py

~4000 trading days × 0.5s/call ≈ 30-35 min for the full run.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tushare_dl")

# ── Config ──────────────────────────────────────────────────────────────────
TOKEN = "0ff4745f6b01cd08fe571d921e4a59ba8bd976dc3cb4e8ea0bd32c22"
START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 5, 22)
RAW_DIR = Path("data/cache/daily_raw")
BADJ_DIR = Path("data/cache/daily_badj")
ADJ_PATH = Path("data/cache/adj_factor.parquet")
RATE_LIMIT = 0.12
FLUSH_EVERY = 60
RETRY_LIMIT = 5

PRICE_LIMITS = {
    "main_board": 0.10, "star_market": 0.20,
    "chinext": 0.20, "beijing": 0.30,
}


def infer_board(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("688"):
        return "star_market"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("8", "4")):
        return "beijing"
    return "main_board"


def get_trading_days(pro) -> list[str]:
    logger.info("Fetching trading calendar...")
    df = pro.trade_cal(
        exchange="SSE",
        start_date=START_DATE.strftime("%Y%m%d"),
        end_date=END_DATE.strftime("%Y%m%d"),
        fields="cal_date,is_open",
    )
    if df is None or df.empty:
        logger.error("Failed to fetch trade_cal")
        sys.exit(1)
    days = sorted(df[df["is_open"] == 1]["cal_date"].tolist())
    logger.info("Trading days: %d (%s ~ %s)", len(days), days[0], days[-1])
    return days


def fetch_one_day(pro, trade_date: str) -> pd.DataFrame | None:
    for attempt in range(RETRY_LIMIT):
        try:
            df = pro.daily(
                trade_date=trade_date,
                fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
            )
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            wait = min(2 ** attempt, 30)
            if attempt < RETRY_LIMIT - 1:
                logger.warning("daily %s attempt %d: %s, retry in %ds",
                               trade_date, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.error("daily %s FAILED after %d retries: %s",
                             trade_date, RETRY_LIMIT, e)
                return None
    return None


def fetch_adj_factors(pro, ts_codes: list[str], start: str, end: str) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    batch_size = 1  # one stock per call: 100 stocks × 4000 days > 6000 limit
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i + batch_size]
        ts_code_str = ",".join(batch)
        for attempt in range(RETRY_LIMIT):
            try:
                df = pro.adj_factor(ts_code=ts_code_str, start_date=start, end_date=end)
                break
            except Exception as e:
                wait = min(2 ** attempt, 30)
                if attempt < RETRY_LIMIT - 1:
                    logger.warning("adj_factor batch %d attempt %d: %s, retry in %ds",
                                   i // batch_size, attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("adj_factor batch %d FAILED", i // batch_size)
                    df = None
        if df is not None and not df.empty:
            all_frames.append(df)
        time.sleep(RATE_LIMIT)
    if not all_frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    return pd.concat(all_frames, ignore_index=True)


def write_partitioned(df: pd.DataFrame, base_dir: Path):
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


def build_raw_row(df_mb: pd.DataFrame, stock_info: pd.DataFrame) -> pd.DataFrame:
    """Transform one day of tushare data into the cache schema (raw prices)."""
    out = df_mb.copy()
    out["symbol"] = out["ts_code"].str.split(".").str[0].str.zfill(6)
    out = out.rename(columns={"vol": "volume"})
    out["board"] = out["ts_code"].map(
        lambda tc: stock_info.loc[tc, "board"] if tc in stock_info.index else "main_board"
    )
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["adj_factor"] = 1.0   # placeholder, filled in step 4
    out["turnover"] = np.nan
    out["is_suspended"] = out["volume"] == 0
    out["is_st"] = False
    out["price_limit_frac"] = out["board"].map(lambda b: PRICE_LIMITS.get(b, 0.10)).fillna(0.10)
    out["market_cap"] = np.nan
    out["tradable_shares"] = np.nan
    for col in ["open", "high", "low", "close", "pre_close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keep = [
        "trade_date", "symbol", "open", "high", "low", "close",
        "pre_close", "volume", "amount", "adj_factor", "turnover",
        "is_suspended", "is_st", "price_limit_frac", "board",
        "market_cap", "tradable_shares",
    ]
    return out[keep]


def merge_adj_into_raw():
    """Merge adj_factor values into daily_raw parquets (updates adj_factor from 1.0 to real)."""
    if not ADJ_PATH.exists():
        logger.warning("adj_factor.parquet not found, skipping merge")
        return
    adj_all = pd.read_parquet(ADJ_PATH)
    adj_lookup = adj_all[["symbol", "trade_date", "adj_factor"]].copy()
    adj_lookup["trade_date"] = pd.to_datetime(adj_lookup["trade_date"])

    for pq_path in sorted(RAW_DIR.rglob("*.parquet")):
        try:
            raw = pd.read_parquet(pq_path)
        except Exception:
            continue
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        merged = raw.merge(adj_lookup, on=["trade_date", "symbol"], how="left", suffixes=("_old", ""))
        if "adj_factor" in merged.columns:
            merged["adj_factor"] = merged["adj_factor"].fillna(
                merged.get("adj_factor_old", raw["adj_factor"])
            )
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_old")], errors="ignore")
        merged.to_parquet(pq_path, index=False)
    logger.info("adj_factor merged into daily_raw.")


def build_badj_from_raw():
    """Derive daily_badj from daily_raw: back-adjusted = raw × adj_factor."""
    logger.info("Building daily_badj from daily_raw × adj_factor...")
    adj_all = pd.read_parquet(ADJ_PATH)
    adj_lookup = adj_all.set_index(["symbol", "trade_date"])["adj_factor"]

    n_files = 0
    for pq_path in sorted(RAW_DIR.rglob("*.parquet")):
        raw = pd.read_parquet(pq_path)
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])

        # Map adj_factor; keep 1.0 for missing entries
        idx = pd.MultiIndex.from_frame(raw[["symbol", "trade_date"]])
        raw["adj"] = adj_lookup.reindex(idx).fillna(1.0).values

        # Apply back-adjustment to OHLCV prices
        price_cols = ["open", "high", "low", "close", "pre_close"]
        for col in price_cols:
            if col in raw.columns:
                raw[col] = raw[col] * raw["adj"]

        raw["adj_factor"] = raw["adj"]
        raw = raw.drop(columns=["adj"])

        # Write to badj dir (same year/month partition)
        out_dir = pq_path.relative_to(RAW_DIR)
        badj_path = BADJ_DIR / out_dir
        badj_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(badj_path, index=False)
        n_files += 1

    logger.info("daily_badj built: %d files.", n_files)


def main():
    import tushare as ts

    pro = ts.pro_api(TOKEN)

    # ── Step 1: Stock list ──
    logger.info("=== Step 1: Stock list ===")
    stocks = pro.stock_basic(exchange="", list_status="L",
                             fields="ts_code,symbol,name,list_date")
    if stocks is None or stocks.empty:
        logger.error("stock_basic failed")
        sys.exit(1)

    stocks["board"] = stocks["symbol"].apply(infer_board)
    stocks["symbol"] = stocks["symbol"].astype(str).str.zfill(6)
    main = stocks[stocks["board"] == "main_board"].copy()
    main_ts_codes = sorted(main["ts_code"].unique().tolist())
    logger.info("Total: %d A-shares, Main-board: %d", len(stocks), len(main))

    list_path = Path("data/cache/stock_list.parquet")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    main.to_parquet(list_path, index=False)
    stock_info = main.set_index("ts_code")[["symbol", "board"]]

    # ── Step 2: Trading days ──
    logger.info("=== Step 2: Trading calendar ===")
    trading_days = get_trading_days(pro)
    total_days = len(trading_days)

    # ── Step 3: Download raw daily data → daily_raw ──
    logger.info("=== Step 3: Download raw OHLCV → daily_raw ===")
    start_str = START_DATE.strftime("%Y%m%d")
    end_str = END_DATE.strftime("%Y%m%d")

    buffer: list[pd.DataFrame] = []
    total_rows = 0
    done_days = 0
    failed_days = 0
    t_start = time.time()
    last_log = t_start

    for i, td in enumerate(trading_days):
        time.sleep(RATE_LIMIT)

        df_raw = fetch_one_day(pro, td)
        if df_raw is None:
            failed_days += 1
            done_days += 1
            continue

        # Filter to main-board only
        df_raw["sym"] = df_raw["ts_code"].str.split(".").str[0].str.zfill(6)
        mask = df_raw["sym"].str.match(r"^(60|00)")
        df_mb = df_raw[mask].drop(columns=["sym"])
        if df_mb.empty:
            done_days += 1
            continue

        out = build_raw_row(df_mb, stock_info)
        buffer.append(out)

        total_rows += len(out)
        done_days += 1

        now = time.time()
        if (i + 1) % 10 == 0 or (now - last_log) > 30 or i == total_days - 1:
            elapsed = now - t_start
            rate = done_days / elapsed if elapsed > 0 else 0
            pct = done_days / total_days * 100
            eta_min = (total_days - done_days) / rate / 60 if rate > 0 else 0
            logger.info("raw: %d/%d days (%.1f%%) | %d rows | %.1f d/min | ETA %.0f min | %d fail",
                        done_days, total_days, pct, total_rows, rate * 60, eta_min, failed_days)
            last_log = now

        if len(buffer) >= FLUSH_EVERY:
            merged = pd.concat(buffer, ignore_index=True)
            write_partitioned(merged, RAW_DIR)
            buffer.clear()

    if buffer:
        merged = pd.concat(buffer, ignore_index=True)
        write_partitioned(merged, RAW_DIR)
        buffer.clear()

    elapsed_daily = time.time() - t_start
    logger.info("Raw download: %d rows, %d/%d days, %d failed, %.0fs",
                total_rows, done_days, total_days, failed_days, elapsed_daily)

    # ── Step 4: adj_factor ──
    logger.info("=== Step 4: adj_factor ===")
    t_adj = time.time()
    adj_df = fetch_adj_factors(pro, main_ts_codes, start_str, end_str)
    logger.info("adj_factor: %d rows in %.0fs", len(adj_df), time.time() - t_adj)

    if not adj_df.empty:
        adj_df["symbol"] = adj_df["ts_code"].str.split(".").str[0].str.zfill(6)
        adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
        adj_df.to_parquet(ADJ_PATH, index=False)
        logger.info("Saved %d adj_factor rows to %s", len(adj_df), ADJ_PATH)

    # ── Step 5: Merge adj_factor into daily_raw ──
    if ADJ_PATH.exists():
        logger.info("=== Step 5: Merge adj_factor → daily_raw ===")
        merge_adj_into_raw()

    # ── Step 6: Build daily_badj from daily_raw × adj_factor ──
    if ADJ_PATH.exists():
        logger.info("=== Step 6: Build daily_badj ===")
        build_badj_from_raw()

    # ── Stats ──
    total_elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Complete! Total: %.0fs (%.1f min)", total_elapsed, total_elapsed / 60)
    for label, d in [("daily_raw", RAW_DIR), ("daily_badj", BADJ_DIR)]:
        if d.exists():
            files = list(d.rglob("*.parquet"))
            mb = sum(f.stat().st_size for f in files) / 1024 / 1024
            logger.info("%s: %d files, %.1f MB", label, len(files), mb)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
