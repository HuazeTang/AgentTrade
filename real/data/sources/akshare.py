"""Akshare data source implementation (free, no token required)."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime

# `requests` reads macOS system proxy (e.g. Clash/V2Ray on 127.0.0.1:7897)
# via urllib.request.getproxies() at import time.  Monkey-patching that
# function is too late if any dependency already pulled in `requests`.
# Setting NO_PROXY=* works because `requests.utils.should_bypass_proxies`
# reads the env var at call time.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

import numpy as np
import pandas as pd

from config.settings import PRICE_LIMITS
from data.sources.base import DataSource

logger = logging.getLogger(__name__)

# Board mapping: akshare codes to internal labels
_BOARD_MAP = {
    "主板": "main_board",
    "创业板": "chinext",
    "科创板": "star_market",
    "北交所": "beijing",
}


class AkshareSource(DataSource):
    """Data source backed by akshare (free, community-maintained)."""

    name = "akshare"

    def __init__(self, rate_limit: float = 0.5):
        self._rate_limit = rate_limit
        self._stock_cache: pd.DataFrame | None = None
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    # ── stock list ──────────────────────────────────────────────────────

    def list_stocks(self) -> pd.DataFrame:
        if self._stock_cache is not None:
            return self._stock_cache.copy()

        import akshare as ak

        self._throttle()
        raw = ak.stock_zh_a_spot_em()
        df = pd.DataFrame()
        df["symbol"] = raw["代码"].astype(str).str.zfill(6)
        df["name"] = raw["名称"].astype(str)
        # Approximate board from 6-digit code
        df["board"] = df["symbol"].apply(_infer_board)
        df["is_st"] = raw["名称"].str.contains("ST|\\*ST", case=True, na=False)
        # list_date / delist_date not available from spot_em; leave NaN
        df["list_date"] = pd.NaT
        df["delist_date"] = pd.NaT
        self._stock_cache = df.copy()
        return df

    # ── daily OHLCV ─────────────────────────────────────────────────────

    def fetch_daily(
        self,
        symbols: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        import akshare as ak

        frames: list[pd.DataFrame] = []
        period = "daily"

        for sym in symbols:
            self._throttle()
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=sym, period=period,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",  # 前复权
                )
            except Exception:
                logger.debug("akshare fetch failed for %s", sym)
                continue
            if raw is None or raw.empty:
                continue

            raw = raw.rename(columns={
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
            })
            raw["trade_date"] = pd.to_datetime(raw["trade_date"])
            raw["symbol"] = sym
            raw["pre_close"] = raw["close"].shift(1)
            raw["adj_factor"] = 1.0  # qfq already adjusted
            raw["is_suspended"] = raw["volume"] <= 0
            raw["is_st"] = False
            raw["price_limit_frac"] = PRICE_LIMITS["main_board"]
            raw["board"] = "main_board"
            raw["market_cap"] = np.nan
            raw["tradable_shares"] = np.nan

            keep = [
                "trade_date", "symbol", "open", "high", "low", "close",
                "pre_close", "volume", "amount", "adj_factor", "turnover",
                "is_suspended", "is_st", "price_limit_frac", "board",
                "market_cap", "tradable_shares",
            ]
            frames.append(raw[keep])

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.set_index(["trade_date", "symbol"]).sort_index()
        return result

    # ── industry ────────────────────────────────────────────────────────

    def fetch_industry(
        self, d: date, level: str = "sw_l1"
    ) -> pd.DataFrame:
        import akshare as ak

        self._throttle()
        try:
            raw = ak.stock_board_industry_name_em()
            df = pd.DataFrame()
            df["symbol"] = raw["代码"].astype(str).str.zfill(6)
            df["industry_name"] = raw["板块名称"].astype(str)
            df["industry_code"] = df["industry_name"]
        except Exception:
            try:
                raw = ak.stock_board_concept_name_em()
                df = pd.DataFrame()
                df["symbol"] = raw["代码"].astype(str).str.zfill(6)
                df["industry_name"] = raw["板块名称"].astype(str)
                df["industry_code"] = df["industry_name"]
            except Exception:
                return pd.DataFrame(columns=["symbol", "industry_code", "industry_name"])
        return df

    # ── index weights ───────────────────────────────────────────────────

    def fetch_index_weights(
        self, index_code: str, d: date
    ) -> pd.DataFrame:
        import akshare as ak

        self._throttle()
        try:
            raw = ak.index_stock_cons_weight_csindex(index=index_code)
            df = pd.DataFrame()
            df["symbol"] = raw["成分券代码"].astype(str).str.zfill(6)
            df["weight"] = pd.to_numeric(raw["权重"], errors="coerce") / 100.0
        except Exception:
            return pd.DataFrame(columns=["symbol", "weight"])
        return df

    # ── adjustments ─────────────────────────────────────────────────────

    def fetch_adjustments(
        self, symbols: list[str], start: date, end: date
    ) -> pd.DataFrame:
        # akshare does not have a direct QFQ factor endpoint;
        # the qfq-adjusted prices already embed adjustments.
        return pd.DataFrame(columns=["symbol", "ex_date", "adj_factor", "event_type"])


def _infer_board(code: str) -> str:
    """Infer board from 6-digit stock code prefix."""
    code = str(code).zfill(6)
    if code.startswith("688"):
        return "star_market"
    elif code.startswith("300") or code.startswith("301"):
        return "chinext"
    elif code.startswith("8") or code.startswith("4"):
        return "beijing"
    else:
        return "main_board"
