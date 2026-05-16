"""Stock universe filtering for A-share backtesting."""

from __future__ import annotations

import pandas as pd

from config.settings import MIN_LISTING_DAYS


class UniverseFilter:
    """Filters the tradeable universe each day."""

    def __init__(
        self,
        exclude_st: bool = True,
        exclude_suspended: bool = True,
        exclude_limit: bool = True,
        min_listing_days: int = MIN_LISTING_DAYS,
    ):
        self.exclude_st = exclude_st
        self.exclude_suspended = exclude_suspended
        self.exclude_limit = exclude_limit
        self.min_listing_days = min_listing_days

    def filter(
        self,
        date: pd.Timestamp,
        symbols: list[str],
        info: pd.DataFrame,
        yesterday_data: pd.DataFrame,
    ) -> list[str]:
        """Return tradeable symbols for a given date.

        Args:
            date: Current trading date.
            symbols: All available symbols.
            info: DataFrame indexed by symbol with 'list_date', 'delist_date',
                  'is_st', 'board'.
            yesterday_data: Previous day's data, multi-indexed (trade_date, symbol).
        """
        universe = list(symbols)

        if self.exclude_st and not info.empty and "is_st" in info.columns:
            st_mask = info["is_st"].fillna(False)
            universe = [s for s in universe if s not in info.index or not st_mask.get(s, False)]

        if self.exclude_suspended and not yesterday_data.empty:
            if "is_suspended" in yesterday_data.columns:
                # Get yesterday's suspension status per symbol
                if isinstance(yesterday_data.index, pd.MultiIndex):
                    yest = yesterday_data.xs(
                        date - pd.Timedelta(days=1), level="trade_date", drop_level=False
                    )
                else:
                    yest = yesterday_data
                suspended = set(yest[yest.get("is_suspended", False)].index.get_level_values("symbol"))
                universe = [s for s in universe if s not in suspended]

        if self.exclude_limit and not yesterday_data.empty:
            if isinstance(yesterday_data.index, pd.MultiIndex):
                yest = yesterday_data.xs(
                    date - pd.Timedelta(days=1), level="trade_date", drop_level=False
                )
            else:
                yest = yesterday_data
            if "close" in yest.columns and "pre_close" in yest.columns:
                limit_pct = yest.get("price_limit_frac", 0.10)
                change = (yest["close"] - yest["pre_close"]) / yest["pre_close"].abs()
                limit_up = set(yest[change >= limit_pct - 0.001].index.get_level_values("symbol"))
                limit_down = set(yest[change <= -limit_pct + 0.001].index.get_level_values("symbol"))
                universe = [s for s in universe if s not in limit_up and s not in limit_down]

        return universe
