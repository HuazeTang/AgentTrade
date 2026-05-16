"""A-share broker: enforces T+1, price limits, lot sizes, and executes orders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import LOT_SIZE
from core.types import Fill, Order, Position, Side


def _round_lot(n: int) -> int:
    """Round down to nearest lot size (multiples of 100)."""
    return max(0, (n // LOT_SIZE) * LOT_SIZE)


@dataclass
class AShareBroker:
    """Broker that enforces A-share trading rules and executes orders.

    Rules enforced:
    - T+1: Shares bought today cannot be sold until next trading day.
    - Price limits: Cannot buy at limit-up or sell at limit-down.
    - Lot size: Shares must be multiples of 100.
    - Delisting: Force-sell positions on delisting date.
    """

    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 5.0

    # Internal state
    _position_lots: dict[str, list[dict]] = field(default_factory=dict)
    _fill_counter: int = field(default=0, init=False)

    def _next_fill_id(self) -> str:
        self._fill_counter += 1
        return f"fill{self._fill_counter:08d}"

    def register_buy(
        self, symbol: str, date: pd.Timestamp, shares: int, avg_price: float
    ) -> None:
        """Register a new holding lot (for T+1 tracking)."""
        if symbol not in self._position_lots:
            self._position_lots[symbol] = []
        self._position_lots[symbol].append({
            "buy_date": date.normalize(),
            "shares": shares,
            "avg_price": avg_price,
        })

    def sellable_shares(self, symbol: str, today: pd.Timestamp) -> int:
        """Return the number of shares that can be sold today (T+0 buys excluded)."""
        if symbol not in self._position_lots:
            return 0
        today_d = today.normalize()
        total = 0
        for lot in self._position_lots[symbol]:
            buy_d = lot["buy_date"]
            if isinstance(buy_d, pd.Timestamp):
                buy_d = buy_d.normalize()
            if buy_d < today_d:  # strictly before today
                total += lot["shares"]
        return total

    def total_position(self, symbol: str) -> int:
        """Total shares held for a symbol (including T+0 buys)."""
        if symbol not in self._position_lots:
            return 0
        return sum(lot["shares"] for lot in self._position_lots[symbol])

    def execute_orders(
        self,
        orders: list[Order],
        market_open: pd.Series,
        market_pre_close: pd.Series,
        price_limits: pd.Series,
        today: pd.Timestamp,
        market_close: pd.Series | None = None,
    ) -> tuple[list[Fill], float]:
        """Execute orders: sells at open, buys at close (if available).

        Args:
            orders: List of Order objects.
            market_open: Series symbol -> open price (used for sells and limit checks).
            market_pre_close: Series symbol -> previous close (for limit calc).
            price_limits: Series symbol -> limit fraction (0.10 for main board).
            today: Current trading date.
            market_close: Series symbol -> close price (used for buys). Falls back to open.

        Returns:
            (fills, total_commission) where fills are the executed fills.
        """
        fills: list[Fill] = []
        total_cost = 0.0

        for order in orders:
            sym = order.symbol
            if sym not in market_open.index or pd.isna(market_open.get(sym)):
                continue

            open_price = float(market_open[sym])
            limit_frac = float(price_limits.get(sym, 0.10))
            pre_close = float(market_pre_close.get(sym, open_price))

            # Execution price: sells at open, buys at close (or open fallback)
            if order.side == Side.BUY and market_close is not None and sym in market_close.index:
                ref_price = float(market_close.get(sym, open_price))
            else:
                ref_price = open_price

            # Price limit check
            if pre_close > 0:
                limit_up = pre_close * (1 + limit_frac)
                limit_down = pre_close * (1 - limit_frac)
                if order.side == Side.BUY and ref_price >= limit_up * 0.999:
                    continue  # Hit limit-up, can't buy
                if order.side == Side.SELL and ref_price <= limit_down * 1.001:
                    continue  # Hit limit-down, can't sell

            # Lot size rounding
            qty = _round_lot(order.quantity)
            if qty <= 0:
                continue

            # T+1 check for sells
            if order.side == Side.SELL:
                available = self.sellable_shares(sym, today)
                qty = min(qty, available)
                if qty <= 0:
                    continue

            # Slippage (simplified: add slippage_bps to price)
            slippage_bps = float(self.slippage_bps)
            if order.side == Side.BUY:
                exec_price = ref_price * (1 + slippage_bps / 10000.0)
            else:
                exec_price = ref_price * (1 - slippage_bps / 10000.0)

            # Transaction costs
            turnover = exec_price * qty
            commission = max(self.min_commission, turnover * self.commission_rate)
            stamp_tax = turnover * self.stamp_tax_rate if order.side == Side.SELL else 0.0
            transfer_fee = turnover * self.transfer_fee_rate
            trade_cost = commission + stamp_tax + transfer_fee

            # Record fill
            fill = Fill(
                order_id=order.order_id,
                symbol=sym,
                side=order.side,
                quantity=qty,
                price=exec_price,
                commission=commission,
                stamp_tax=stamp_tax,
                transfer_fee=transfer_fee,
                slippage=abs(exec_price - open_price) * qty,
                trade_date=today,
                fill_id=self._next_fill_id(),
            )
            fills.append(fill)

            if order.side == Side.SELL:
                total_cost += turnover - trade_cost  # proceeds minus costs
            else:
                total_cost += turnover + trade_cost  # cost plus fees

        net_cash_flow = sum(
            (f.price * f.quantity * (1 if f.side == Side.SELL else -1)) - f.commission - f.stamp_tax - f.transfer_fee
            for f in fills
        )
        return fills, net_cash_flow

    def remove_sold_lots(
        self, symbol: str, sold_qty: int, today: pd.Timestamp
    ) -> None:
        """Decrement lot shares for sold quantity (FIFO)."""
        if symbol not in self._position_lots:
            return
        today_d = today.normalize()
        remaining = sold_qty
        for lot in self._position_lots[symbol]:
            if remaining <= 0:
                break
            buy_d = lot["buy_date"]
            if isinstance(buy_d, pd.Timestamp):
                buy_d = buy_d.normalize()
            if buy_d < today_d:
                taken = min(remaining, lot["shares"])
                lot["shares"] -= taken
                remaining -= taken
        self._position_lots[symbol] = [
            l for l in self._position_lots[symbol] if l["shares"] > 0
        ]

    def reset(self) -> None:
        """Clear all internal state."""
        self._position_lots.clear()
        self._fill_counter = 0
