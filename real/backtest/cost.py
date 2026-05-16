"""A-share transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass

from core.types import Side


@dataclass(frozen=True)
class TransactionCost:
    """A-share transaction cost calculator.

    Costs:
    - Commission (佣金): 0.025% per side, min 5 CNY
    - Stamp tax (印花税): 0.1% on sell side only
    - Transfer fee (过户费): 0.001% per side
    """

    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00001

    def compute(
        self, side: Side, price: float, quantity: int
    ) -> tuple[float, float, float]:
        """Compute costs for a trade.

        Returns:
            (commission, stamp_tax, transfer_fee) all in CNY.
        """
        turnover = price * quantity
        commission = max(self.min_commission, turnover * self.commission_rate)
        stamp_tax = turnover * self.stamp_tax_rate if side == Side.SELL else 0.0
        transfer_fee = turnover * self.transfer_fee_rate
        return commission, stamp_tax, transfer_fee

    def total_cost(self, side: Side, price: float, quantity: int) -> float:
        """Total cost in CNY."""
        c, s, t = self.compute(side, price, quantity)
        return c + s + t
