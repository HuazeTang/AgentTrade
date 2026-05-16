"""Event-driven backtest engine for A-share markets.

Daily cycle:
1. Advance clock → next trading day
2. Filter universe (suspensions, ST, price limits)
3. Load lagged data for decision date
4. Strategy generates target weights
5. Convert weights → orders
6. Execute at open price (with T+1 enforcement)
7. Apply transaction costs
8. End-of-day mark-to-market
9. Record portfolio state
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from core.types import BacktestConfig, BacktestResult, Fill, Order, OrderType, Side
from data.calendar import get_trading_days
from data.cache import read_daily
from backtest.accounting import PortfolioAccountant
from backtest.broker import AShareBroker
from backtest.universe import UniverseFilter
from strategy.base import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Orchestrates a full A-share backtest."""

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Strategy,
        data_source=None,
    ):
        self.config = config
        self.strategy = strategy
        self.data_source = data_source
        self.broker = AShareBroker(
            commission_rate=config.commission_rate,
            min_commission=config.min_commission,
            stamp_tax_rate=config.stamp_tax_rate,
            transfer_fee_rate=config.transfer_fee_rate,
            slippage_bps=config.slippage_bps,
        )
        self.accountant = PortfolioAccountant(initial_cash=config.initial_cash)
        self.universe_filter = UniverseFilter()

        # Working state
        self._trading_days: pd.DatetimeIndex | None = None
        self._daily_cache: pd.DataFrame | None = None
        self._all_fills: list[dict] = []
        self._all_positions: list[dict] = []
        self._all_returns: list[dict] = []
        self._all_benchmark_returns: list[dict] = []

    def run(self) -> BacktestResult:
        """Execute the full backtest."""
        self._trading_days = get_trading_days(
            self.config.start_date, self.config.end_date
        )
        if len(self._trading_days) == 0:
            raise ValueError("No trading days in the specified date range")

        logger.info(
            "Running backtest: %s → %s, %d trading days",
            self.config.start_date, self.config.end_date,
            len(self._trading_days),
        )

        # Preload daily data for the entire period
        self._daily_cache = read_daily(
            self.config.start_date, self.config.end_date
        )

        symbols = self._get_symbols()
        if not symbols:
            raise ValueError("No symbols found in data cache. Run data ingest first.")

        logger.info("Universe: %d symbols", len(symbols))

        for i, today in enumerate(self._trading_days):
            if i % 50 == 0:
                logger.debug("Processing %s (%d/%d)", today.date(), i + 1, len(self._trading_days))

            self._process_day(today, symbols)

        return self._build_result()

    def _get_symbols(self) -> list[str]:
        if self._daily_cache is not None and not self._daily_cache.empty:
            if isinstance(self._daily_cache.index, pd.MultiIndex):
                return sorted(self._daily_cache.index.get_level_values("symbol").unique().tolist())
            elif "symbol" in self._daily_cache.columns:
                return sorted(self._daily_cache["symbol"].unique().tolist())
        return []

    def _process_day(self, today: pd.Timestamp, all_symbols: list[str]) -> None:
        # 1. Universe filter
        yesterday = today - pd.Timedelta(days=1)
        yest_data = pd.DataFrame()
        if self._daily_cache is not None and not self._daily_cache.empty:
            try:
                yest_data = self._daily_cache.xs(yesterday, level="trade_date")
            except KeyError:
                pass

        universe = self.universe_filter.filter(
            today, all_symbols, pd.DataFrame(), yest_data
        )
        if not universe:
            self._record_day(today, [], pd.Series(), pd.Series(), pd.Series())
            return

        # 2. Get latest data for today's decisions
        today_data = self._get_day_data(today, universe)
        prices = self._get_prices(today, today_data)
        pre_close = self._get_prices(yesterday, yest_data) if not yest_data.empty else prices

        # 3. Strategy generates weights
        current_pos = {
            sym: qty for sym, qty in self.accountant.positions.items()
        }
        weights = self.strategy.generate_weights(
            date=today,
            universe=universe,
            data=today_data,
            prices=prices,
            current_positions=current_pos,
            cash=self.accountant.cash,
        )

        # 4. Convert weights to orders
        orders = self._weights_to_orders(
            weights, today, prices, universe
        )

        # 5. Execute orders through broker
        open_prices = self._get_open_prices(today, universe)
        if open_prices.empty:
            open_prices = prices

        fills, net_cash_flow = self.broker.execute_orders(
            orders,
            open_prices,
            pre_close,
            pd.Series(0.10, index=universe),  # default limit 10%
            today,
        )

        # Update T+1 lot tracking
        for f in fills:
            if f.side == Side.BUY:
                self.broker.register_buy(f.symbol, today, f.quantity, f.price)
            else:
                self.broker.remove_sold_lots(f.symbol, f.quantity, today)

        # 6. Accounting
        self.accountant.apply_fills(fills, today)
        self.accountant.mark_to_market(prices, today)
        self.accountant.compute_turnover(fills, today)

        # 7. Record
        self._record_day(today, fills, weights, prices, open_prices)

    def _get_day_data(
        self, today: pd.Timestamp, universe: list[str]
    ) -> pd.DataFrame:
        """Get data available on `today` for decision-making.

        Uses previous day's close data to avoid look-ahead.
        """
        if self._daily_cache is None or self._daily_cache.empty:
            return pd.DataFrame(index=pd.Index(universe, name="symbol"))

        yesterday = today - pd.Timedelta(days=1)
        try:
            data = self._daily_cache.xs(yesterday, level="trade_date")
        except KeyError:
            return pd.DataFrame(index=pd.Index(universe, name="symbol"))

        # Filter to universe
        data = data[data.index.isin(universe)]
        return data

    def _get_prices(
        self, d: pd.Timestamp, data: pd.DataFrame
    ) -> pd.Series:
        """Extract close prices from data."""
        if data.empty or "close" not in data.columns:
            return pd.Series(dtype=float)
        return data["close"]

    def _get_open_prices(
        self, d: pd.Timestamp, universe: list[str]
    ) -> pd.Series:
        """Get open prices for today's trading."""
        if self._daily_cache is None or self._daily_cache.empty:
            return pd.Series(dtype=float)
        try:
            data = self._daily_cache.xs(d, level="trade_date")
        except KeyError:
            return pd.Series(dtype=float)
        if "open" not in data.columns:
            return data.get("close", pd.Series(dtype=float))
        open_prices = data["open"]
        return open_prices[open_prices.index.isin(universe)]

    def _weights_to_orders(
        self,
        weights: pd.Series,
        today: pd.Timestamp,
        prices: pd.Series,
        universe: list[str],
    ) -> list[Order]:
        """Convert target weights to a list of Order objects."""
        if weights.empty:
            return []

        equity = self.accountant.cash
        for sym, qty in self.accountant.positions.items():
            if sym in prices.index:
                equity += qty * float(prices[sym])

        orders: list[Order] = []
        for sym, w in weights.items():
            if sym not in universe:
                continue
            if sym not in prices.index or pd.isna(prices[sym]) or prices[sym] <= 0:
                continue

            target_value = equity * w
            price = float(prices[sym])
            current_shares = self.accountant.positions.get(sym, 0)
            current_value = current_shares * price
            diff_value = target_value - current_value

            if abs(diff_value) < price * 100:  # Skip tiny adjustments (< 1 lot)
                continue

            if diff_value > 0:
                # Buy
                qty = int(diff_value / price)
                qty = (qty // 100) * 100
                if qty > 0:
                    orders.append(Order(
                        symbol=sym,
                        side=Side.BUY,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        date=today,
                        order_id=f"order_{today.date()}_{sym}_buy",
                    ))
            else:
                # Sell
                qty = int(-diff_value / price)
                qty = (qty // 100) * 100
                qty = min(qty, current_shares)
                if qty > 0:
                    orders.append(Order(
                        symbol=sym,
                        side=Side.SELL,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        date=today,
                        order_id=f"order_{today.date()}_{sym}_sell",
                    ))

        return orders

    def _record_day(
        self,
        today: pd.Timestamp,
        fills: list[Fill],
        weights: pd.Series,
        prices: pd.Series,
        open_prices: pd.Series,
    ) -> None:
        """Record daily state."""
        for f in fills:
            self._all_fills.append({
                "trade_date": f.trade_date,
                "fill_id": f.fill_id,
                "order_id": f.order_id,
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "commission": f.commission,
                "stamp_tax": f.stamp_tax,
                "transfer_fee": f.transfer_fee,
                "slippage": f.slippage,
            })

        if not weights.empty:
            for sym, w in weights.items():
                self._all_positions.append({
                    "trade_date": today,
                    "symbol": sym,
                    "weight": w,
                })

        # Benchmark return
        bm_ret = 0.0
        self._all_benchmark_returns.append({
            "trade_date": today,
            "benchmark_return": bm_ret,
        })

    def _build_result(self) -> BacktestResult:
        """Assemble BacktestResult from recorded history."""
        equity = self.accountant.to_equity_series()
        returns = self.accountant.to_return_series()
        turnover = self.accountant.to_turnover_series()

        bm_ret = pd.Series(
            {r["trade_date"]: r["benchmark_return"] for r in self._all_benchmark_returns}
        ).sort_index()

        positions = pd.DataFrame(self._all_positions) if self._all_positions else pd.DataFrame(
            columns=["trade_date", "symbol", "weight"]
        )

        fills = pd.DataFrame(self._all_fills) if self._all_fills else pd.DataFrame(
            columns=["trade_date", "fill_id", "symbol", "side", "quantity", "price"]
        )

        return BacktestResult(
            daily_returns=returns,
            benchmark_returns=bm_ret,
            equity_curve=equity,
            turnover=turnover,
            positions=positions,
            fills=fills,
        )
