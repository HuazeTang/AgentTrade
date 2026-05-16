"""Abstract base class for market data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataSource(ABC):
    """Abstract data source for A-share market data.

    All fetch methods return DataFrames.  Date-indexed data uses
    a consistent column named `trade_date` (datetime64[ns]).
    """

    @abstractmethod
    def list_stocks(self) -> pd.DataFrame:
        """Return all A-share stocks with metadata.

        Returns columns: symbol, name, list_date, delist_date, board, is_st.
        """
        ...

    @abstractmethod
    def fetch_daily(
        self,
        symbols: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV + metadata.

        Returns DataFrame with columns:
          trade_date, symbol, open, high, low, close, pre_close,
          volume, amount, adj_factor, turnover, is_suspended, is_st,
          price_limit_frac, board.

        trade_date is datetime64[ns]; symbol is str.
        """
        ...

    @abstractmethod
    def fetch_industry(
        self, d: date, level: str = "sw_l1"
    ) -> pd.DataFrame:
        """Fetch industry classification for stocks on a given date.

        Returns columns: symbol, industry_code, industry_name.
        level: "sw_l1" for Shenwan Level 1, "sw_l2" for Level 2, etc.
        """
        ...

    @abstractmethod
    def fetch_index_weights(
        self, index_code: str, d: date
    ) -> pd.DataFrame:
        """Fetch index constituent weights.

        Returns columns: symbol, weight.
        """
        ...

    @abstractmethod
    def fetch_adjustments(
        self, symbols: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Fetch dividend/split adjustment factors.

        Returns columns: symbol, ex_date, adj_factor, event_type.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
