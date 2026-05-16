"""Portfolio accounting: track cash, positions, P&L, turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import Fill, Side


class PortfolioAccountant:
    """Tracks portfolio state across backtest days."""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}   # symbol -> shares
        self.avg_costs: dict[str, float] = {}    # symbol -> avg cost
        self.equity_history: list[dict] = []
        self.turnover_history: list[dict] = []

    def apply_fills(self, fills: list[Fill], today: pd.Timestamp) -> None:
        """Update cash and positions from executed fills."""
        for f in fills:
            if f.side == Side.BUY:
                self.cash -= f.price * f.quantity + f.commission + f.stamp_tax + f.transfer_fee
                old_qty = self.positions.get(f.symbol, 0)
                old_cost = self.avg_costs.get(f.symbol, 0.0)
                new_qty = old_qty + f.quantity
                if new_qty > 0:
                    self.avg_costs[f.symbol] = (
                        (old_cost * old_qty + f.price * f.quantity) / new_qty
                    )
                self.positions[f.symbol] = new_qty
            else:
                self.cash += f.price * f.quantity - f.commission - f.stamp_tax - f.transfer_fee
                old_qty = self.positions.get(f.symbol, 0)
                self.positions[f.symbol] = max(0, old_qty - f.quantity)
                if self.positions[f.symbol] <= 0:
                    self.positions.pop(f.symbol, None)
                    self.avg_costs.pop(f.symbol, None)

    def mark_to_market(
        self, prices: pd.Series, today: pd.Timestamp
    ) -> tuple[float, float]:
        """Compute portfolio equity and daily return.

        Args:
            prices: Series symbol -> close price.
            today: Current trading date.

        Returns:
            (equity, daily_return) where return is relative to previous day.
        """
        position_value = 0.0
        for sym, qty in self.positions.items():
            if sym in prices.index and qty > 0:
                position_value += qty * float(prices[sym])

        equity = self.cash + position_value

        prev_equity = self.initial_cash
        if self.equity_history:
            prev_equity = self.equity_history[-1]["equity"]

        daily_return = (equity / prev_equity - 1) if prev_equity > 0 else 0.0

        self.equity_history.append({
            "trade_date": today,
            "equity": equity,
            "cash": self.cash,
            "position_value": position_value,
            "daily_return": daily_return,
        })

        return equity, daily_return

    def compute_turnover(
        self, fills: list[Fill], today: pd.Timestamp
    ) -> float:
        """Compute daily turnover ratio."""
        if not self.equity_history:
            return 0.0

        prev_equity = self.equity_history[-1]["equity"]
        if prev_equity <= 0:
            return 0.0

        buy_value = sum(f.price * f.quantity for f in fills if f.side == Side.BUY)
        sell_value = sum(f.price * f.quantity for f in fills if f.side == Side.SELL)
        turnover = min(buy_value, sell_value) / prev_equity
        self.turnover_history.append({
            "trade_date": today,
            "turnover": turnover,
        })
        return turnover

    def to_equity_series(self) -> pd.Series:
        """Return equity curve as Series indexed by date."""
        return pd.Series(
            {e["trade_date"]: e["equity"] for e in self.equity_history}
        ).sort_index()

    def to_return_series(self) -> pd.Series:
        """Return daily return as Series indexed by date."""
        return pd.Series(
            {e["trade_date"]: e["daily_return"] for e in self.equity_history}
        ).sort_index()

    def to_turnover_series(self) -> pd.Series:
        """Return turnover as Series indexed by date."""
        return pd.Series(
            {t["trade_date"]: t["turnover"] for t in self.turnover_history}
        ).sort_index()

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.avg_costs.clear()
        self.equity_history.clear()
        self.turnover_history.clear()
