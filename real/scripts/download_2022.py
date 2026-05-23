"""Download 2022 daily data via baostock and cache to parquet.

Downloads full year per symbol (1 query/symbol) then partitions into monthly parquets.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sources.baostock import BaoStockSource, _to_bs_code, _infer_board, _BS_DAILY_FIELDS
from data.cache import write_daily
from config.settings import PRICE_LIMITS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_2022")


def fetch_2022(source: BaoStockSource, symbols: list[str], start_str: str, end_str: str) -> pd.DataFrame:
    """Download 2022 daily data for all symbols."""
    import numpy as np
    import socket

    source._login()

    frames: list[pd.DataFrame] = []
    errors = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        if i > 0 and i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(symbols) - i) / rate if rate > 0 else 0
            logger.info("Progress: %d/%d symbols (%.1f%%) | %d errors | %.0fs elapsed | ETA %.0fs",
                         i, len(symbols), 100 * i / len(symbols), errors, elapsed, eta)
            source._reconnect()

        source._throttle()
        bs_code = _to_bs_code(sym)

        try:
            import baostock as bs
            rs = bs.query_history_k_data_plus(
                bs_code, _BS_DAILY_FIELDS,
                start_date=start_str, end_date=end_str,
                frequency="d", adjustflag="2",
            )
        except Exception:
            errors += 1
            continue

        if rs is None or rs.error_code != "0":
            errors += 1
            if errors % 50 == 0:
                source._reconnect()
            continue

        rows = []
        try:
            sym_t0 = time.time()
            while rs.next():
                rows.append(rs.get_row_data())
                if time.time() - sym_t0 > 60:
                    rows = []
                    break
        except (socket.timeout, OSError, Exception):
            errors += 1
            source._reconnect()
            continue

        if not rows:
            continue

        raw = pd.DataFrame(rows, columns=rs.fields)
        raw = raw.rename(columns={
            "date": "trade_date", "open": "open", "high": "high", "low": "low",
            "close": "close", "preclose": "pre_close", "volume": "volume",
            "amount": "amount", "turn": "turnover",
        })

        for col in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        raw["symbol"] = sym
        raw["adj_factor"] = 1.0
        raw["is_suspended"] = raw.get("tradestatus", "1") == "0"
        raw["is_st"] = raw.get("isST", "0") == "1"
        raw["price_limit_frac"] = raw["symbol"].apply(
            lambda s: PRICE_LIMITS.get(_infer_board(s), PRICE_LIMITS["main_board"])
        )
        st_mask = raw["is_st"]
        if st_mask.any():
            raw.loc[st_mask, "price_limit_frac"] = PRICE_LIMITS["st"]
        raw["board"] = raw["symbol"].apply(_infer_board)
        raw["market_cap"] = np.nan
        raw["tradable_shares"] = np.nan

        keep = [
            "trade_date", "symbol", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "adj_factor", "turnover",
            "is_suspended", "is_st", "price_limit_frac", "board",
            "market_cap", "tradable_shares",
        ]
        frames.append(raw[keep])

    source._logout()
    logger.info("Download complete: %d/%d symbols returned data, %d errors",
                 len(frames), len(symbols), errors)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.set_index(["trade_date", "symbol"]).sort_index()


def main():
    start = date(2022, 1, 1)
    end = date(2022, 12, 31)
    start_str = "2022-01-01"
    end_str = "2022-12-31"

    source = BaoStockSource(rate_limit=0.15)
    stocks_df = source.list_stocks()
    symbols = sorted(stocks_df["symbol"].tolist())
    logger.info("Got %d stock symbols, downloading %s ~ %s", len(symbols), start, end)

    df = fetch_2022(source, symbols, start_str, end_str)

    if df.empty:
        logger.error("No data downloaded!")
        sys.exit(1)

    logger.info("Writing %d rows to cache...", len(df))
    write_daily(df)
    logger.info("Done. %d rows cached for 2022.", len(df))


if __name__ == "__main__":
    main()
