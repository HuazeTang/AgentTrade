"""Slippage models for order execution simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.types import Side


class SlippageModel(ABC):
    """Abstract slippage model."""

    @abstractmethod
    def apply(
        self, base_price: float, side: Side, volume_ratio: float = 0.0
    ) -> float:
        """Return the executed price after slippage.

        Args:
            base_price: Reference price (e.g., open price).
            side: Buy or sell.
            volume_ratio: Order volume / average daily volume (0 to 1).
        """
        ...


class NoSlippage(SlippageModel):
    """Execute exactly at the reference price (testing only)."""

    def apply(self, base_price: float, side: Side, volume_ratio: float = 0.0) -> float:
        return base_price


class FixedBpsSlippage(SlippageModel):
    """Add/subtract a fixed basis point spread."""

    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def apply(self, base_price: float, side: Side, volume_ratio: float = 0.0) -> float:
        pct = self.bps / 10000.0
        if side == Side.BUY:
            return base_price * (1 + pct)
        else:
            return base_price * (1 - pct)


class OpenSlippage(SlippageModel):
    """Execute at next-day open (no additional slippage beyond open price)."""

    def apply(self, base_price: float, side: Side, volume_ratio: float = 0.0) -> float:
        return base_price


class VolumeWeightedSlippage(SlippageModel):
    """Add slippage proportional to order volume relative to average daily volume."""

    def __init__(self, impact_coef: float = 10.0):
        """
        Args:
            impact_coef: Basis points of slippage per 1% of ADV.
        """
        self.impact_coef = impact_coef

    def apply(self, base_price: float, side: Side, volume_ratio: float = 0.0) -> float:
        bps = self.impact_coef * volume_ratio * 100  # volume_ratio * 100 = % of ADV
        pct = bps / 10000.0
        if side == Side.BUY:
            return base_price * (1 + pct)
        else:
            return base_price * (1 - pct)
