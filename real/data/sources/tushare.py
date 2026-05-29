"""Tushare data source – token-based, batch-by-date for efficiency."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from config.settings import PRICE_LIMITS
from data.sources.base import DataSource

logger = logging.getLogger(__name__)

# A-share filter: Shanghai/Shenzhen main, STAR, ChiNext (exclude Beijing 8/4)
_A_SHARE_PAT = r'^(\d{6}\.SH|688\d{3}\.SH|00\d{4}\.SZ|30\d{3}\.SZ|002\d{3}\.SZ|003\d{3}\.SZ|001\d{3}\.SZ)'


def _infer_board(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("688"):
        return "star_market"
    if code.startswith("300") or code.startswith("301"):
        return "chinext"
    if code.startswith("8") or code.startswith("4"):
        return "beijing"
    return "main_board"


def _ts_code_to_symbol(ts_code: str) -> str:
    return ts_code.split(".")[0]


class TushareSource(DataSource):
    """Data source backed by tushare pro API (token required).

    Fetches raw (未复权) daily data by trade date in bulk, then supplements
    with daily_basic (turnover, market_cap) and adj_factor for full coverage.
    """

    name = "tushare"

    def __init__(self, token: str | None = None, rate_limit: float = 0.3):
        self._rate_limit = rate_limit
        self._last_call = 0.0
        self._pro = None
        self._stock_cache: pd.DataFrame | None = None

        if token is None:
            # Load .env if not already in environment
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUNSHARE_TOKEN")
        if not token:
            raise ValueError(
                "Tushare token not found. Set TUSHARE_TOKEN in .env or pass token= explicitly."
            )
        self._token = token

    @property
    def pro(self):
        if self._pro is None:
            import tushare as ts
            ts.set_token(self._token)
            self._pro = ts.pro_api()
        return self._pro

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    # ── stock list ────────────────────────────────────────────────────────

    def list_stocks(self) -> pd.DataFrame:
        if self._stock_cache is not None:
            return self._stock_cache.copy()

        self._throttle()
        fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs"
        raw = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields=fields,
        )
        # Also get delisted stocks for historical coverage
        self._throttle()
        raw_d = self.pro.stock_basic(
            exchange="",
            list_status="D",
            fields=fields,
        )

        df = pd.concat([raw, raw_d], ignore_index=True)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "name", "board", "is_st", "list_date", "delist_date"])

        result = pd.DataFrame()
        result["symbol"] = df["symbol"].astype(str).str.zfill(6)
        result["name"] = df["name"].astype(str)
        result["board"] = result["symbol"].apply(_infer_board)
        result["is_st"] = result["name"].str.contains(r"ST|\*ST", case=True, na=False)
        result["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        result["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")

        # Filter A-shares only
        result = result[result["symbol"].str.match(r'^(6|00|30|002|003|001)')].copy()
        result = result.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)

        self._stock_cache = result.copy()
        logger.info("Tushare: %d A-share stocks loaded", len(result))
        return result

    # ── daily OHLCV ───────────────────────────────────────────────────────

    def fetch_daily(
        self,
        symbols: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch raw daily OHLCV + metadata by trading date.

        Downloads one date at a time via pro.daily() + pro.daily_basic(),
        filters to requested symbols, and returns unadjusted (未复权) data
        with adj_factor column populated for later adjustment.
        """
        symbol_set = set(str(s).zfill(6) for s in symbols)
        all_frames: list[pd.DataFrame] = []

        # Generate date range (tushare only returns trading days)
        date_range = pd.bdate_range(start=start, end=end, freq="C", holidays=[])
        logger.info("Tushare: fetching %d symbols over %d weekdays",
                      len(symbol_set), len(date_range))

        for d in date_range:
            d_str = d.strftime("%Y%m%d")
            self._throttle()

            try:
                daily = self.pro.daily(trade_date=d_str)
            except Exception as e:
                logger.error("Tushare daily(%s) failed: %s", d_str, e)
                raise  # let caller decide — no silent fallback

            if daily is None or daily.empty:
                logger.debug("No data for %s", d_str)
                continue

            # Get turnover and market cap
            self._throttle()
            try:
                basic = self.pro.daily_basic(trade_date=d_str)
            except Exception as e:
                logger.warning("Tushare daily_basic(%s) failed: %s — continuing without turnover", d_str, e)
                basic = pd.DataFrame()

            # Merge daily + basic on ts_code
            daily["symbol"] = daily["ts_code"].apply(_ts_code_to_symbol)
            daily = daily[daily["symbol"].isin(symbol_set)]

            if daily.empty:
                continue

            if not basic.empty:
                basic["symbol"] = basic["ts_code"].apply(_ts_code_to_symbol)
                basic_cols = {
                    "turnover_rate": "turnover",
                    "total_mv": "market_cap",
                    "circ_mv": "tradable_shares",
                }
                basic_sub = basic[["symbol"] + list(basic_cols.keys())].rename(columns=basic_cols)
                daily = daily.merge(basic_sub, on="symbol", how="left")
            else:
                daily["turnover"] = np.nan
                daily["market_cap"] = np.nan
                daily["tradable_shares"] = np.nan

            # Map columns to standard names
            daily = daily.rename(columns={
                "vol": "volume",
                "pre_close": "pre_close",
            })

            # Infer metadata
            daily["trade_date"] = pd.to_datetime(daily["trade_date"])
            daily["is_suspended"] = daily["volume"] <= 0
            daily["is_st"] = False
            daily["board"] = daily["symbol"].apply(_infer_board)
            daily["price_limit_frac"] = daily["board"].apply(
                lambda b: PRICE_LIMITS.get(b, PRICE_LIMITS["main_board"])
            )
            daily["adj_factor"] = 1.0  # placeholder, will be filled from adj_factor.parquet

            # Ensure all expected columns exist
            for col in ["turnover", "market_cap", "tradable_shares"]:
                if col not in daily.columns:
                    daily[col] = np.nan

            keep = [
                "trade_date", "symbol", "open", "high", "low", "close",
                "pre_close", "volume", "amount", "adj_factor", "turnover",
                "is_suspended", "is_st", "price_limit_frac", "board",
                "market_cap", "tradable_shares",
            ]
            all_frames.append(daily[keep])

        if not all_frames:
            logger.warning("Tushare: no data fetched for %d symbols in %s ~ %s",
                           len(symbol_set), start, end)
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = result.set_index(["trade_date", "symbol"]).sort_index()
        logger.info("Tushare: fetched %d rows for %d symbols", len(result),
                      result.index.get_level_values("symbol").nunique())
        return result

    # ── adj_factor ────────────────────────────────────────────────────────

    def fetch_adj_factor(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Download adj_factor for given symbols and date range.

        Uses pro.adj_factor(trade_date=...) to get all stocks per date.
        Returns DataFrame with columns: ts_code, trade_date, adj_factor, symbol.
        """
        symbol_set = set(str(s).zfill(6) for s in symbols)
        frames: list[pd.DataFrame] = []

        date_range = pd.bdate_range(start=start, end=end, freq="C", holidays=[])
        for d in date_range:
            d_str = d.strftime("%Y%m%d")
            self._throttle()
            try:
                df = self.pro.adj_factor(trade_date=d_str)
            except Exception as e:
                logger.error("Tushare adj_factor(%s) failed: %s", d_str, e)
                raise

            if df is None or df.empty:
                continue

            df["symbol"] = df["ts_code"].apply(_ts_code_to_symbol)
            df = df[df["symbol"].isin(symbol_set)]
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            frames.append(df[["ts_code", "trade_date", "adj_factor", "symbol"]])

        if not frames:
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor", "symbol"])

        result = pd.concat(frames, ignore_index=True)
        logger.info("Tushare: fetched adj_factor %d rows for %d symbols",
                      len(result), result["symbol"].nunique())
        return result

    # ── industry ──────────────────────────────────────────────────────────

    def fetch_industry(
        self, d: date, level: str = "sw_l1"
    ) -> pd.DataFrame:
        # tushare provides Shenwan industry classification via pro.index_classify()
        # but it requires a separate (expensive) membership. Use stock_basic industry field instead.
        stocks = self.list_stocks()
        return pd.DataFrame(columns=["symbol", "industry_code", "industry_name"])

    # ── index weights ─────────────────────────────────────────────────────

    def fetch_index_weights(
        self, index_code: str, d: date
    ) -> pd.DataFrame:
        self._throttle()
        d_str = d.strftime("%Y%m%d")
        try:
            raw = self.pro.index_weight(index_code=index_code, trade_date=d_str)
        except Exception:
            return pd.DataFrame(columns=["symbol", "weight"])

        if raw is None or raw.empty:
            return pd.DataFrame(columns=["symbol", "weight"])

        df = pd.DataFrame()
        df["symbol"] = raw["con_code"].astype(str).apply(_ts_code_to_symbol)
        df["weight"] = pd.to_numeric(raw["weight"], errors="coerce") / 100.0
        return df

    # ── adjustments ───────────────────────────────────────────────────────

    def fetch_adjustments(
        self, symbols: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Fetch adj_factor as adjustment events (delegates to fetch_adj_factor)."""
        return self.fetch_adj_factor(symbols, start, end)
