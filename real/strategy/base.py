"""Abstract strategy base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract strategy that generates target portfolio weights.

    A strategy receives all available data as of a given decision date
    (with proper lag to avoid look-ahead bias) and outputs target weights
    as a fraction of total portfolio equity.
    """

    @abstractmethod
    def generate_weights(
        self,
        date: pd.Timestamp,
        universe: list[str],
        data: pd.DataFrame,
        prices: pd.Series,
        current_positions: dict[str, float],
        cash: float,
    ) -> pd.Series:
        """Generate target weight per stock.

        Args:
            date: Current decision date.
            universe: List of tradeable symbols.
            data: DataFrame indexed by symbol with factor/signal values.
            prices: Series indexed by symbol with latest close price.
            current_positions: {symbol: shares} currently held.
            cash: Available cash.

        Returns:
            Series indexed by symbol with target weight values
            as fraction of total equity. Positive = long.
            Symbols not in the result receive zero weight.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def allow_short(self) -> bool:
        """Whether this strategy allows short selling."""
        return False
