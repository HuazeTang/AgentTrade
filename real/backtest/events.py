"""Event types for the event-driven backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.types import Fill, Order, Side


@dataclass
class MarketOpen:
    """Fired at the start of each trading day."""
    date: pd.Timestamp


@dataclass
class DataReady:
    """Fired when data for a given date is loaded and available."""
    date: pd.Timestamp
    data: pd.DataFrame         # factor/signal values, (symbol) index
    prices: pd.Series          # symbol -> close price (previous day)


@dataclass
class SignalGenerated:
    """Fired when the strategy produces target weights."""
    date: pd.Timestamp
    weights: pd.Series         # symbol -> target weight


@dataclass
class OrdersGenerated:
    """Fired when target weights are converted to orders."""
    date: pd.Timestamp
    orders: list[Order]


@dataclass
class OrderFilled:
    """Fired when an order is fully or partially filled."""
    date: pd.Timestamp
    fill: Fill


@dataclass
class DayEnd:
    """Fired at the end of each trading day."""
    date: pd.Timestamp
    portfolio: dict            # snapshot of portfolio state
