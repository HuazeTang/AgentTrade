"""A-share trading calendar."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

# Built-in holiday set for China. This is maintained manually or
# can be refreshed from akshare. Values are date strings "YYYY-MM-DD".
_CHINA_HOLIDAYS: set[str] = set()
_TRADING_DAYS_CACHE: pd.DatetimeIndex | None = None


def _build_default_calendar(start_year: int = 2010, end_year: int = 2030) -> pd.DatetimeIndex:
    """Generate trading days by excluding weekends only.
    Specific holidays can be added via `add_holidays`.
    """
    dates = pd.bdate_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="C",
        weekmask="Mon Tue Wed Thu Fri",
    )
    return dates


def _default_calendar() -> pd.DatetimeIndex:
    global _TRADING_DAYS_CACHE
    if _TRADING_DAYS_CACHE is None:
        _TRADING_DAYS_CACHE = _build_default_calendar()
    return _TRADING_DAYS_CACHE


def add_holidays(*date_strs: str) -> None:
    """Register additional non-trading days (e.g., national holidays)."""
    _CHINA_HOLIDAYS.update(date_strs)
    global _TRADING_DAYS_CACHE
    _TRADING_DAYS_CACHE = None


def load_holidays_from_akshare() -> int:
    """Try to load China holiday calendar from akshare.
    Returns number of holidays loaded, or 0 on failure.
    """
    try:
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        # akshare returns all dates; trading days are those with trade_status=1
        # We only need to identify non-trading weekdays as holidays
        all_dates = pd.to_datetime(df["trade_date"])
        trading = set(
            all_dates[df.get("trade_status", 1) == 1].strftime("%Y-%m-%d")
        )
        # Generate all weekdays and mark non-trading ones as holidays
        full_range = pd.bdate_range(
            start=all_dates.min(), end=all_dates.max(), freq="C"
        )
        holidays = {
            d.strftime("%Y-%m-%d") for d in full_range if d.strftime("%Y-%m-%d") not in trading
        }
        add_holidays(*holidays)
        return len(holidays)
    except Exception:
        return 0


def get_trading_days(
    start: date | str, end: date | str
) -> pd.DatetimeIndex:
    """Return trading days in [start, end] (inclusive)."""
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    cal = _default_calendar()
    mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
    days = cal[mask]
    if _CHINA_HOLIDAYS:
        holidays_ts = {pd.Timestamp(h) for h in _CHINA_HOLIDAYS}
        days = days[~days.isin(holidays_ts)]
    return days


def is_trading_day(d: date | str | pd.Timestamp) -> bool:
    """Check if a given date is a trading day."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    ts = pd.Timestamp(d)
    cal = _default_calendar()
    if ts not in cal:
        return False
    if ts.strftime("%Y-%m-%d") in _CHINA_HOLIDAYS:
        return False
    return True


def prev_trading_day(d: date | str | pd.Timestamp) -> pd.Timestamp:
    """Return the most recent trading day <= d."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    ts = pd.Timestamp(d)
    days = get_trading_days(ts - timedelta(days=10), ts)
    return days[-1]


def next_trading_day(d: date | str | pd.Timestamp) -> pd.Timestamp:
    """Return the earliest trading day >= d."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    ts = pd.Timestamp(d)
    days = get_trading_days(ts, ts + timedelta(days=10))
    return days[0]


def shift_trading_day(
    d: pd.Timestamp, n: int
) -> pd.Timestamp:
    """Shift by n trading days. n > 0 forward, n < 0 backward."""
    if n == 0:
        return d
    cal = _default_calendar()
    loc = cal.get_loc(d)
    if isinstance(loc, slice):
        loc = loc.start
    new_loc = loc + n
    if new_loc < 0 or new_loc >= len(cal):
        raise ValueError(f"Cannot shift {n} days from {d}: out of calendar range")
    return cal[new_loc]
