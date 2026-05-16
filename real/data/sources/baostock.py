"""Baostock data source – free, no token, uses raw TCP socket (bypasses HTTP proxy)."""

from __future__ import annotations

import logging
import socket
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import PRICE_LIMITS
from data.sources.base import DataSource

logger = logging.getLogger(__name__)

# Board inference from 6-digit code prefix
_BOARD_MAP = {
    "60": "main_board",
    "00": "main_board",
    "688": "star_market",
    "300": "chinext",
    "301": "chinext",
    "8": "beijing",
    "4": "beijing",
}

# Available K-line fields from baostock
_BS_DAILY_FIELDS = "date,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST"


def _to_bs_code(symbol: str) -> str:
    """'600000' -> 'sh.600000', '000001' -> 'sz.000001'"""
    symbol = str(symbol).strip().zfill(6)
    prefix = "sh" if symbol.startswith("6") else "sz"
    return f"{prefix}.{symbol}"


def _from_bs_code(bs_code: str) -> str:
    """'sh.600000' -> '600000'"""
    return bs_code.split(".")[-1]


def _infer_board(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("688"):
        return "star_market"
    if code.startswith("300") or code.startswith("301"):
        return "chinext"
    if code.startswith("8") or code.startswith("4"):
        return "beijing"
    return "main_board"


class BaoStockSource(DataSource):
    """Data source backed by baostock raw TCP protocol."""

    name = "baostock"

    def __init__(self, rate_limit: float = 0.2):
        self._rate_limit = rate_limit
        self._stock_cache: pd.DataFrame | None = None
        self._last_call = 0.0
        self._logged_in = False

    def _login(self) -> None:
        if self._logged_in:
            return
        import baostock as bs
        import baostock.common.context as conx

        socket.setdefaulttimeout(10)
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")

        # Aggressively shorten the real socket timeout so a silent
        # server cannot hold the recv loop open for too long.
        if hasattr(conx, "default_socket"):
            sock_obj = getattr(conx, "default_socket")
            if sock_obj is not None:
                try:
                    sock_obj.settimeout(5)
                except Exception:
                    pass

        self._logged_in = True

    def _reconnect(self) -> None:
        """Force a fresh TCP connection after a socket error."""
        import baostock as bs
        import baostock.common.context as conx

        try:
            if hasattr(conx, "default_socket"):
                sock_obj = getattr(conx, "default_socket")
                if sock_obj is not None:
                    try:
                        sock_obj.close()
                    except Exception:
                        pass
                setattr(conx, "default_socket", None)
        except Exception:
            pass
        try:
            bs.logout()
        except Exception:
            pass
        self._logged_in = False
        self._login()

    def _logout(self) -> None:
        if not self._logged_in:
            return
        import baostock as bs

        bs.logout()
        self._logged_in = False

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    def _bs_query_to_df(self, rs) -> pd.DataFrame:
        """Convert baostock ResultData to DataFrame."""
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame(columns=rs.fields)
        return pd.DataFrame(rows, columns=rs.fields)

    def list_stocks(self) -> pd.DataFrame:
        if self._stock_cache is not None:
            return self._stock_cache.copy()

        import baostock as bs

        self._login()
        self._throttle()

        today = date.today()
        # Try up to 5 days back to find the most recent trading day
        for offset in range(5):
            query_date = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            rs = bs.query_all_stock(day=query_date)
            df = self._bs_query_to_df(rs)
            if not df.empty:
                break

        if df.empty:
            return df

        # Filter A-shares only (exclude sz.399xxx indices)
        df = df[df["code"].str.match(r"^(sh\.6|sz\.00|sz\.300|sz\.301|bj\.)")].copy()

        result = pd.DataFrame()
        result["symbol"] = df["code"].apply(_from_bs_code)
        result["name"] = df["code_name"].astype(str)
        result["board"] = result["symbol"].apply(_infer_board)
        result["is_st"] = result["name"].str.contains("ST|\\*ST", case=True, na=False)
        result["list_date"] = pd.NaT
        result["delist_date"] = pd.NaT

        self._stock_cache = result.copy()
        return result

    def fetch_daily(
        self,
        symbols: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        import baostock as bs

        self._login()

        frames: list[pd.DataFrame] = []
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        SYMBOL_DEADLINE = 60  # max seconds for a single symbol
        RECONNECT_EVERY = 200  # fresh TCP connection periodically

        for i, sym in enumerate(symbols):
            # Periodic reconnect to avoid server-side rate limiting
            if i > 0 and i % RECONNECT_EVERY == 0:
                logger.info("Periodic reconnect after %d symbols", i)
                self._reconnect()

            sym_t0 = time.time()
            self._throttle()
            bs_code = _to_bs_code(sym)
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    _BS_DAILY_FIELDS,
                    start_date=start_str,
                    end_date=end_str,
                    frequency="d",
                    adjustflag="2",  # 前复权
                )
            except Exception:
                logger.debug("baostock query failed for %s", sym)
                continue

            if rs is None or rs.error_code != "0":
                logger.debug("baostock query returned error for %s: %s", sym, rs.error_code if rs else "None")
                self._reconnect()
                continue

            rows = []
            try:
                while rs.next():
                    rows.append(rs.get_row_data())
                    if time.time() - sym_t0 > SYMBOL_DEADLINE:
                        logger.warning("symbol %s exceeded %ds deadline, skipping", sym, SYMBOL_DEADLINE)
                        rows = []
                        break
            except (socket.timeout, OSError, TimeoutError, Exception) as e:
                logger.warning("baostock error for %s: %s", sym, e)
                self._reconnect()
                continue

            elapsed = time.time() - sym_t0
            if elapsed > 15:
                logger.debug("symbol %s took %.0fs", sym, elapsed)

            if not rows:
                continue

            raw = pd.DataFrame(rows, columns=rs.fields)

            raw = raw.rename(columns={
                "date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "preclose": "pre_close",
                "volume": "volume",
                "amount": "amount",
                "turn": "turnover",
            })

            # Baostock returns strings; convert numerics
            for col in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover"]:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

            raw["trade_date"] = pd.to_datetime(raw["trade_date"])
            raw["symbol"] = sym
            raw["adj_factor"] = 1.0  # qfq already adjusted
            raw["is_suspended"] = raw.get("tradestatus", "1") == "0"
            raw["is_st"] = raw.get("isST", "0") == "1"
            raw["price_limit_frac"] = raw["symbol"].apply(
                lambda s: PRICE_LIMITS.get(_infer_board(s), PRICE_LIMITS["main_board"])
            )
            # Apply ST 5% override
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

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.set_index(["trade_date", "symbol"]).sort_index()
        return result

    def fetch_industry(
        self, d: date, level: str = "sw_l1"
    ) -> pd.DataFrame:
        import baostock as bs

        self._login()

        # baostock only provides CSRC (证监会) classification, not Shenwan.
        stocks_df = self.list_stocks()
        if stocks_df.empty:
            return pd.DataFrame(columns=["symbol", "industry_code", "industry_name"])

        rows: list[dict] = []
        symbols = stocks_df["symbol"].tolist()

        for sym in symbols:
            self._throttle()
            bs_code = _to_bs_code(sym)
            try:
                rs = bs.query_stock_industry(bs_code)
            except Exception:
                continue
            if rs.error_code != "0":
                continue
            while rs.next():
                row_data = rs.get_row_data()
                # Fields: updateDate, code, code_name, industry, industryClassification
                if len(row_data) >= 4:
                    rows.append({
                        "symbol": sym,
                        "industry_code": row_data[3],
                        "industry_name": row_data[3],
                    })
                break  # take first classification only

        if not rows:
            return pd.DataFrame(columns=["symbol", "industry_code", "industry_name"])

        return pd.DataFrame(rows)

    def fetch_index_weights(
        self, index_code: str, d: date
    ) -> pd.DataFrame:
        import baostock as bs

        self._login()
        self._throttle()

        index_map = {
            "000300.SH": bs.query_hs300_stocks,
            "000016.SH": bs.query_sz50_stocks,
            "000905.SH": bs.query_zz500_stocks,
        }

        fn = index_map.get(index_code)
        if fn is None:
            logger.warning("Index %s not supported by baostock", index_code)
            return pd.DataFrame(columns=["symbol", "weight"])

        try:
            rs = fn()
            df = self._bs_query_to_df(rs)
        except Exception:
            return pd.DataFrame(columns=["symbol", "weight"])

        if df.empty:
            return pd.DataFrame(columns=["symbol", "weight"])

        # baostock returns constituents without weights
        # Fields: updateDate / tradeDate, code, code_name
        code_col = "code" if "code" in df.columns else df.columns[1]
        result = pd.DataFrame()
        result["symbol"] = df[code_col].astype(str).apply(_from_bs_code)
        n = len(result)
        result["weight"] = 1.0 / n if n > 0 else 0.0  # equal weight fallback

        return result

    def fetch_adjustments(
        self, symbols: list[str], start: date, end: date
    ) -> pd.DataFrame:
        import baostock as bs

        self._login()

        frames: list[pd.DataFrame] = []
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        for sym in symbols:
            self._throttle()
            bs_code = _to_bs_code(sym)
            try:
                rs = bs.query_adjust_factor(
                    code=bs_code, start_date=start_str, end_date=end_str
                )
            except Exception:
                continue

            rows = []
            while (rs.error_code == "0") and rs.next():
                row_data = rs.get_row_data()
                rows.append(row_data)

            if not rows:
                continue

            raw = pd.DataFrame(
                rows,
                columns=["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor", "adjustFactor"],
            )
            raw["symbol"] = sym
            raw["ex_date"] = pd.to_datetime(raw["dividOperateDate"])
            raw["adj_factor"] = pd.to_numeric(raw["adjustFactor"], errors="coerce")
            raw["event_type"] = "dividend"
            frames.append(raw[["symbol", "ex_date", "adj_factor", "event_type"]])

        if not frames:
            return pd.DataFrame(columns=["symbol", "ex_date", "adj_factor", "event_type"])

        return pd.concat(frames, ignore_index=True)

    def close(self) -> None:
        self._logout()

    def __del__(self) -> None:
        try:
            self._logout()
        except Exception:
            pass
