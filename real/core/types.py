"""Shared data types for the backtesting system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd


# ── Enums ────────────────────────────────────────────────────────────────────


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class PriceLimit(float, Enum):
    """A-share price limit regimes, as decimal fraction."""

    MAIN_BOARD = 0.10
    STAR_MARKET = 0.20
    CHINEXT = 0.20
    BEIJING = 0.30
    ST_STOCK = 0.05


# ── Core data structures ─────────────────────────────────────────────────────


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    date: pd.Timestamp | None = None
    order_id: str = ""


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    slippage: float
    trade_date: pd.Timestamp
    fill_id: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    buy_date: pd.Timestamp | None = None


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position]
    equity: float = 0.0
    date: pd.Timestamp | None = None


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_cash: float = 1_000_000.0
    benchmark: str = "000300.SH"       # CSI 300
    commission_rate: float = 0.00025    # 0.025% 佣金
    min_commission: float = 5.0         # 最低佣金
    stamp_tax_rate: float = 0.001       # 0.1% 印花税 (卖出)
    transfer_fee_rate: float = 0.00001  # 0.001% 过户费
    slippage_bps: float = 5.0           # 5 bps 滑点
    max_position_pct: float = 0.10      # 单票最大 10%
    rebalance_freq: str = "monthly"


@dataclass
class BacktestResult:
    """Container for completed backtest results."""

    daily_returns: pd.Series         # date -> daily portfolio return
    benchmark_returns: pd.Series     # date -> benchmark daily return
    equity_curve: pd.Series          # date -> portfolio equity
    turnover: pd.Series              # date -> daily turnover
    positions: pd.DataFrame          # date x symbol -> weight
    fills: pd.DataFrame              # all fills during backtest
    factor_exposures: pd.DataFrame | None = None

    @property
    def cumulative_return(self) -> float:
        return float((1 + self.daily_returns).prod() - 1)

    @property
    def annualized_return(self) -> float:
        n_years = len(self.daily_returns) / 252
        return float((1 + self.cumulative_return) ** (1 / max(n_years, 0.01)) - 1)

    @property
    def annualized_volatility(self) -> float:
        return float(self.daily_returns.std() * (252**0.5))

    @property
    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility
        if vol == 0:
            return 0.0
        return float(self.annualized_return / vol)

    @property
    def max_drawdown(self) -> float:
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        return float(drawdown.min())

    @property
    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown)
        if mdd == 0:
            return 0.0
        return float(self.annualized_return / mdd)

    @property
    def excess_return(self) -> float:
        return float(self.cumulative_return - (1 + self.benchmark_returns).prod() + 1)

    @property
    def information_ratio(self) -> float:
        excess = self.daily_returns - self.benchmark_returns
        vol = excess.std() * (252**0.5)
        if vol == 0:
            return 0.0
        return float(excess.mean() * 252 / vol)
