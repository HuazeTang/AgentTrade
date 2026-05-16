"""Abstract base class and metadata for all factors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FactorMeta:
    name: str
    category: str  # momentum, value, quality, volatility, size, liquidity, growth
    description: str = ""
    version: str = "1.0.0"
    params: dict = field(default_factory=dict)
    lookback_days: int = 0
    lag_days: int = 1  # days to shift to avoid look-ahead bias


class Factor(ABC):
    """Abstract base class for all factors.

    Factors operate on a DataFrame multi-indexed by (trade_date, symbol)
    and return a Series with the same multi-index.

    Subclasses must set the `meta` class attribute.
    """

    meta: FactorMeta

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute factor values.

        Args:
            data: Multi-indexed DataFrame with index (trade_date, symbol).
                  Must contain at minimum the fields listed in required_fields.

        Returns:
            Series with same multi-index (trade_date, symbol), dtype float.
        """
        ...

    @property
    def dependencies(self) -> list[str]:
        """Names of other factors this factor depends on. Default: none."""
        return []

    @property
    def required_fields(self) -> list[str]:
        """Raw data fields this factor needs. Override in subclass."""
        return ["close", "volume"]
