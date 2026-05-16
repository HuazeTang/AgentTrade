#!/usr/bin/env python3
"""Day-by-day Trading Simulation with unified, realistic constraints.

Same as the research report:
  - 500-stock industry-stratified pool
  - IC_IR-calibrated dynamic factor weights
  - Real trading costs, T+1, lot-size rounding, position limits
  - Weekly rebalancing (default)

Two decision modes:
  - factor  – pure signal-driven (top-N by composite)
  - llm     – LLM agent makes buy/sell decisions
  - compare – run both side-by-side
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# config.chart_style must be imported before matplotlib.pyplot
import config.chart_style  # noqa: F401 — CJK fonts + Agg backend + rcParams
from config.chart_style import COLORS as CHART_COLORS, get_cjk_font_name, get_mpl_rc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from agent.llm_client import LLMClient, create_default_client
from backtest.accounting import PortfolioAccountant
from backtest.broker import AShareBroker
from backtest.universe import UniverseFilter
from config.settings import DATA_DIR, PRICE_LIMITS, LOT_SIZE
from core.types import Fill, Order, OrderType, Side
from data.cache import read_daily
from data.calendar import get_trading_days
from data.sources.baostock import _infer_board
import factor.factors as _  # register all factors
from factor.engine import FactorEngine
from factor.validation import compute_rank_ic
from discovery.gp import GPEngine, GPConfig
from discovery.validate import FactorValidator, orthogonal_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_sim")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_PERIOD   = (date(2025, 10, 1), date(2026, 4, 30))   # out-of-sample
TRAIN_PERIOD     = (date(2024, 8, 30), date(2025, 10, 16))   # IC calibration
FORWARD_PERIODS  = 5        # 5-day forward return for IC
INITIAL_CASH     = 10_000.0
MAX_POSITIONS    = 1
MAX_POSITION_PCT = 0.95
SELL_RANK_LIMIT  = 5     # sell only if rank drops below this (vs MAX_POSITIONS=1 for buys)
SYMBOL_COUNT     = 500       # same as report stock pool
JOURNAL_DIR      = Path(__file__).resolve().parent / "data" / "results"

# Market drawdown circuit breaker: skip new buys when market is falling
# to avoid catching a falling knife. Existing positions ride through.
MKT_DD_THRESHOLD = 1.0    # disabled — found counterproductive in bull periods

# Take-profit threshold: lock in gains before blow-off reversals
TAKE_PROFIT_PCT  = 0.25   # sell if position profit > 25%

BASELINE_FACTORS = [
    # Momentum — multi-timeframe (1m through 12m)
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m1m",
    # Reversal — entry timing
    "reversal_5d", "reversal_10d",
    # Volatility — risk sizing
    "volatility_20d", "volatility_60d",
    "beta_60d",
    # Liquidity
    "turnover_20d",
    # Trend-following (主升浪 identification)
    "trend_efficiency_20d",    # smooth trend vs choppy
    "ma_trend_5_20",          # price vs MA20 (uptrend anchor)
    "donchian_pct_20d",       # Donchian channel position (breakout)
    "up_days_ratio_20d",      # persistent buying pressure
    "ma_cross_5_20",          # MA5 vs MA20 alignment
    # Leader identification (龙头/连板)
    "limit_up_freq_20d",      # 涨停频率 — hallmark of short-term leaders
    "relative_strength_10d",  # excess return vs sector — separates leaders from followers
    "volume_surge_5d",        # volume confirmation of breakout
    "close_position_5d",      # intraday strength — closes near high = bullish
    # Volume-price integration (量价综合)
    "vol_weighted_mom_5d",    # volume-confirmed momentum
    "money_flow_ratio_20d",   # money flow direction (net buying vs selling)
    "vwap_delta_5d",          # deviation from VWAP (buyers vs sellers control)
    "vol_price_div_5d",       # price-volume divergence (weakening/strengthening trend)
    # Risk / drawdown control (回撤控制)
    "downside_vol_20d",       # downside-only volatility (not total vol)
    "max_dd_20d",             # maximum drawdown from 20-day peak
    "risk_adj_mom_20d",       # risk-adjusted momentum (Sharpe-like)
    "dd_recovery_5d",         # bounce strength from 20-day low
    # Market risk (大盘风险)
    "market_dd_beta_20d",     # market drawdown × stock beta — high-beta punished when market falls
]
# Set factors you want to temporarily exclude here
DISABLED_FACTORS: set[str] = {
    "downside_vol_20d", "max_dd_20d", "risk_adj_mom_20d", "dd_recovery_5d",
}
REBALANCE_FREQ   = "weekly"  # "daily", "weekly", "monthly"


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an A-share quantitative trader managing a 1,000,000 CNY portfolio.

## Trading Rules
- A-share main board stocks only (60xxxx, 00xxxx)
- Lot size: multiples of 100 shares
- T+1: shares bought today CANNOT be sold today
- Max 10 positions simultaneously
- Max 15% of equity per position
- Commission: 0.025% (min 5 CNY), Stamp tax: 0.1% (sell only)
- Price limits: ±10% for main board

## Your Task
Given today's market data and your current portfolio, decide:
1. Which stocks to BUY (and how many shares)
2. Which stocks to SELL (and how many shares)

## Decision Framework
- Focus on stocks with strong positive factor signals (high composite score)
- Cut losing positions if the thesis is broken (stock shows clear weakness)
- Take profits on positions with >15% gain if signals deteriorate
- Keep some cash reserve (~10-20%) for new opportunities
- Prioritize stocks with: low volatility, good liquidity, reasonable turnover

## Output Format
Respond ONLY with a JSON object:
{
  "reasoning": "brief 2-3 sentence explanation of your strategy today",
  "buys": [
    {"symbol": "600000", "shares": 1000, "reason": "strong reversal signal, low vol"}
  ],
  "sells": [
    {"symbol": "000001", "shares": 500, "reason": "momentum weakening, take profit"}
  ]
}

If no action is needed, return empty arrays. Be decisive but prudent."""


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AgentSimulation:
    """Day-by-day LLM agent trading simulation."""

    def __init__(
        self,
        start: date,
        end: date,
        initial_cash: float = INITIAL_CASH,
        mode: str = "llm",       # "llm", "factor", "heuristic"
        model: str | None = None,
        output_dir: Path | None = None,
        use_gp: bool = False,
        gp_population: int = 200,
        gp_generations: int = 25,
        gp_early_stop: int = 10,
    ):
        self.start = start
        self.end = end
        self.initial_cash = initial_cash
        self.mode = mode
        self.output_dir = output_dir or JOURNAL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.broker = AShareBroker(
            commission_rate=0.00025,
            min_commission=5.0,
            stamp_tax_rate=0.001,
            transfer_fee_rate=0.00001,
            slippage_bps=5.0,
        )
        self.accountant = PortfolioAccountant(initial_cash=initial_cash)
        self.universe_filter = UniverseFilter()

        # LLM
        self.use_llm = (mode == "llm")
        self.use_gp = use_gp
        self.gp_population = gp_population
        self.gp_generations = gp_generations
        self.gp_early_stop = gp_early_stop
        self.llm = create_default_client() if self.use_llm else None
        if self.use_llm and model:
            self.llm.model = model
        if self.use_llm and not (self.llm and self.llm.configured):
            logger.warning("No LLM API key found. Set DEEPSEEK_API_KEY in .env.")
            logger.warning("Falling back to factor strategy.")
            self.mode = "factor"
            self.use_llm = False

        # State
        self._daily_cache: pd.DataFrame | None = None
        self._trading_days: pd.DatetimeIndex | None = None
        self._journal: list[dict] = []
        self._decisions: list[dict] = []
        self._fill_count = 0

    # ── Main Entry Point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full simulation with unified, realistic constraints."""
        self._trading_days = get_trading_days(self.start, self.end)
        logger.info("Simulation: %s → %s, %d trading days",
                     self.start, self.end, len(self._trading_days))

        # 1. Load full data range (training + trading) for IC calibration
        load_start = TRAIN_PERIOD[0]
        load_end = self.end
        self._daily_cache = read_daily(load_start, load_end)
        all_syms = sorted(
            self._daily_cache.index.get_level_values("symbol").unique().tolist()
        )
        logger.info("Loaded %d rows, %d symbols from cache (%s → %s)",
                     len(self._daily_cache), len(all_syms), load_start, load_end)

        # 2. Generate industry-stratified 500-stock pool
        symbols = self._generate_stock_pool(all_syms)
        if len(symbols) < 50:
            logger.error("Stock pool too small (%d stocks), aborting", len(symbols))
            return {"error": "stock pool too small"}

        # Filter cache to stock pool
        pool_mask = self._daily_cache.index.get_level_values("symbol").isin(symbols)
        self._daily_cache = self._daily_cache[pool_mask]
        logger.info("Filtered to %d pool stocks: %d rows", len(symbols), len(self._daily_cache))

        # 3. Add derived features (needed for GP terminals)
        self._daily_cache = self._add_derived_features(self._daily_cache)

        # 4. Determine factor list: baseline + optionally GP-discovered
        active_factors = [f for f in BASELINE_FACTORS if f not in DISABLED_FACTORS]

        if self.use_gp:
            gp_factors = self._run_gp_discovery(self._daily_cache)
            if gp_factors:
                active_factors.extend(gp_factors)
                logger.info("GP discovered %d new factors. Total: %d",
                             len(gp_factors), len(active_factors))
        else:
            gp_factors = []

        # 5. Compute all factors via FactorEngine
        engine = FactorEngine()
        self._factor_df = engine.compute(active_factors, self._daily_cache)
        logger.info("FactorEngine computed %d factors: %d rows",
                     len(active_factors), len(self._factor_df))

        # 5.5 Shift factors by 1 day: D's factor uses D-1's close data.
        # Sells at D open use D-1 info = correct. Buys at D close also use D-1 info
        # (1-day conservative lag, but avoids look-ahead).
        shifted = self._factor_df.unstack().shift(1).stack()
        self._factor_df = shifted.reorder_levels(["trade_date", "symbol"]).sort_index()

        # 6. Calibrate factor weights on training period (IC_IR-weighted)
        train_dates = get_trading_days(TRAIN_PERIOD[0], TRAIN_PERIOD[1])
        self._factor_weights = self._calibrate_factor_weights(self._factor_df, train_dates)
        logger.info("Calibrated factor weights: %s",
                     {k: f"{v:.3f}" for k, v in self._factor_weights.items()})

        # 6.5 Compute market drawdown for circuit breaker
        self._market_dd = self._compute_market_drawdown()
        logger.info("Market drawdown computed: %d days", len(self._market_dd))

        # 7. Day-by-day loop with weekly rebalancing
        trading_day_index = 0
        for i, today in enumerate(self._trading_days):
            td = today.date()
            is_rebalance = self._is_rebalance_day(trading_day_index)

            if i % 20 == 0:
                logger.info("Day %d/%d: %s %s",
                             i + 1, len(self._trading_days), td,
                             "[rebalance]" if is_rebalance else "")

            try:
                self._process_day(today, symbols, trading_day_index, is_rebalance)
                trading_day_index += 1
            except Exception as e:
                logger.error("Error on %s: %s", td, e, exc_info=True)
                self._journal.append({
                    "date": td.isoformat(),
                    "error": str(e),
                    "cash": self.accountant.cash,
                    "equity": self._equity,
                    "positions": [{"symbol": s, "shares": q, "avg_cost": 0} for s, q in self.accountant.positions.items() if q > 0],
                })

        return self._finalize()

    # ── Daily Processing ────────────────────────────────────────────────────

    def _process_day(
        self,
        today: pd.Timestamp,
        symbols: list[str],
        day_index: int,
        is_rebalance: bool,
    ) -> None:
        """Process a single trading day with weekly rebalancing."""
        td = today.date()

        # 1. Universe filter
        yest_data = self._get_data(today - pd.Timedelta(days=1))
        universe = self.universe_filter.filter(today, symbols, pd.DataFrame(), yest_data)
        if len(universe) < 10:
            self._record_day(td, [], "market closed or too few stocks")
            return

        # 2. Get today's prices
        today_data = self._get_data(today)
        prices_today = today_data["close"] if not today_data.empty else pd.Series(dtype=float)
        open_prices = today_data["open"] if not today_data.empty and "open" in today_data.columns else prices_today

        if prices_today.empty:
            self._record_day(td, [], "no price data")
            return

        # 3. Pre-close and price limits
        pre_close = yest_data["close"] if not yest_data.empty else prices_today
        price_limits = pd.Series(
            [PRICE_LIMITS.get(_infer_board(s), PRICE_LIMITS["main_board"]) for s in universe],
            index=universe,
        )

        if not is_rebalance:
            self.accountant.mark_to_market(prices_today, today)
            self._record_day(td, [], "non-rebalance day")
            return

        # 4. Compute factor composite for today and build candidates
        candidates = self._get_candidates(td, universe, yest_data)

        # 5. Get trading decision based on mode
        if self.mode == "llm" and self.llm and self.llm.configured:
            decision = self._llm_decide(td, candidates, yest_data)
        elif self.mode == "factor":
            decision = self._factor_decide(td, candidates, yest_data)
        else:
            decision = self._heuristic_decide(td, candidates, yest_data)

        self._decisions.append(decision)

        # 6. Convert decisions to orders
        orders = self._decisions_to_orders(decision, td, prices_today, universe)

        # 7. Execute orders at open
        fills, net_cash = self.broker.execute_orders(
            orders, open_prices, pre_close, price_limits, today,
        )

        # 8. Update T+1 tracking
        for f in fills:
            if f.side == Side.BUY:
                self.broker.register_buy(f.symbol, today, f.quantity, f.price)
            else:
                self.broker.remove_sold_lots(f.symbol, f.quantity, today)

        # 9. Accounting
        self.accountant.apply_fills(fills, today)
        self.accountant.mark_to_market(prices_today, today)

        # 10. Journal entry
        self._record_day(td, fills, decision.get("reasoning", ""))

    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def _equity(self) -> float:
        """Current portfolio equity."""
        if self.accountant.equity_history:
            return self.accountant.equity_history[-1]["equity"]
        return self.initial_cash

    # ── Data Access ─────────────────────────────────────────────────────────

    def _get_data(self, d: pd.Timestamp) -> pd.DataFrame:
        """Get daily data for a specific date."""
        if self._daily_cache is None or self._daily_cache.empty:
            return pd.DataFrame()
        try:
            return self._daily_cache.xs(d, level="trade_date")
        except KeyError:
            return pd.DataFrame()

    # ── Rebalance Schedule ──────────────────────────────────────────────────

    def _is_rebalance_day(self, day_index: int) -> bool:
        """Check if today is a rebalance day based on REBALANCE_FREQ."""
        if REBALANCE_FREQ == "daily":
            return True
        if REBALANCE_FREQ == "weekly":
            return day_index % 5 == 0
        if REBALANCE_FREQ == "monthly":
            return day_index % 20 == 0
        return True

    # ── Stock Pool ──────────────────────────────────────────────────────────

    def _generate_stock_pool(self, all_syms: list[str]) -> list[str]:
        """Industry-stratified 500-stock pool, using cache metadata (no live API call)."""
        from data.industry import build_industry_map

        # Derive ST status and board from cache (per-symbol, use last available day)
        sym_meta = self._daily_cache.groupby(level="symbol")[["is_st", "board"]].last()

        # Filter: non-ST, non-BJ (main_board, chinext, star_market)
        valid_mask = ~sym_meta["is_st"].fillna(False)
        valid_mask &= sym_meta["board"].isin(["main_board", "chinext", "star_market"])
        valid_syms = set(sym_meta[valid_mask].index.tolist())

        # Candidate pool = cached ∩ valid
        candidate_syms = sorted(s for s in all_syms if s in valid_syms)
        logger.info("Stock pool candidates: %d (from %d cached, %d valid — from cache)",
                     len(candidate_syms), len(all_syms), len(valid_syms))

        if len(candidate_syms) < SYMBOL_COUNT:
            logger.warning("Only %d valid candidates, using all", len(candidate_syms))
            return candidate_syms[:SYMBOL_COUNT]

        # Get Shenwan L1 industry classification
        industry_map = build_industry_map()

        # Compute avg daily amount for liquidity ranking (training period only)
        train_mask = self._daily_cache.index.get_level_values("trade_date").isin(
            get_trading_days(TRAIN_PERIOD[0], TRAIN_PERIOD[1])
        )
        train_data = self._daily_cache[train_mask]
        train_filtered = train_data[train_data.index.get_level_values("symbol").isin(candidate_syms)]
        _avg_amt = train_filtered.groupby(level="symbol")["amount"].mean().sort_values(ascending=False)

        # Map symbol → industry
        _symbol_industry = {sym: industry_map.get(sym, "综合") for sym in _avg_amt.index}

        # Build per-industry candidate lists sorted by liquidity
        from collections import defaultdict
        _ind_candidates: dict[str, list[str]] = defaultdict(list)
        for sym in _avg_amt.index:
            ind = _symbol_industry.get(sym, "综合")
            _ind_candidates[ind].append(sym)

        # Per-industry allocation proportional to market representation
        _ind_total_stocks = {ind: len(syms) for ind, syms in _ind_candidates.items()}
        _total_repr = sum(_ind_total_stocks.values())
        _ind_alloc = {}
        for ind, n_stocks in _ind_total_stocks.items():
            _ind_alloc[ind] = max(3, int(SYMBOL_COUNT * n_stocks / _total_repr))

        # Adjust to hit exactly SYMBOL_COUNT
        _alloc_total = sum(_ind_alloc.values())
        while _alloc_total > SYMBOL_COUNT:
            max_ind = max(_ind_alloc, key=lambda k: _ind_alloc[k] - 3)
            if _ind_alloc[max_ind] > 3:
                _ind_alloc[max_ind] -= 1
                _alloc_total -= 1
            else:
                break
        while _alloc_total < SYMBOL_COUNT:
            min_ind = min(_ind_alloc, key=lambda k: _ind_alloc[k])
            _ind_alloc[min_ind] += 1
            _alloc_total += 1

        # Select stocks
        symbols: list[str] = []
        industry_counts: dict[str, int] = {}
        for ind, alloc in sorted(_ind_alloc.items(), key=lambda x: -x[1]):
            n_avail = len(_ind_candidates[ind])
            n_pick = min(alloc, n_avail)
            picked = _ind_candidates[ind][:n_pick]
            symbols.extend(picked)
            industry_counts[ind] = n_pick

        logger.info("Stock pool: %d stocks, %d industries", len(symbols), len(industry_counts))
        logger.info("Top industries: %s",
                     dict(sorted(industry_counts.items(), key=lambda x: -x[1])[:5]))
        return symbols

    # ── Factor Calibration ──────────────────────────────────────────────────

    def _calibrate_factor_weights(
        self,
        factor_df: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
    ) -> dict[str, float]:
        """Calibrate factor weights via IC_IR on training period.

        Weight = |IC_mean| / sum(|IC_mean|), floored at 0.005 per factor.
        Works with all factor columns in factor_df.
        """
        factor_names = list(factor_df.columns)

        if len(train_dates) < 20:
            logger.warning("Too few training dates (%d), using equal weights", len(train_dates))
            n = max(len(factor_names), 1)
            return {f: 1.0 / n for f in factor_names}

        # Compute forward returns on training period
        train_mask = self._daily_cache.index.get_level_values("trade_date").isin(train_dates)
        train_cache = self._daily_cache[train_mask]

        close = train_cache["close"].unstack()
        fwd_ret = close.pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()
        fwd_ret.name = "fwd_ret"

        weights = {}
        for fname in factor_names:
            if fname not in factor_df.columns:
                continue
            factor_vals = factor_df[fname]
            common_idx = factor_vals.dropna().index.intersection(fwd_ret.dropna().index)
            if len(common_idx) < 50:
                logger.warning("Factor %s: too few IC samples (%d)", fname, len(common_idx))
                continue

            ic = compute_rank_ic(factor_vals.loc[common_idx], fwd_ret.loc[common_idx])
            weights[fname] = max(abs(ic.mean()), 0.005)

        total_w = sum(weights.values())
        if total_w <= 0:
            n = max(len(factor_names), 1)
            return {f: 1.0 / n for f in factor_names}

        weights = {k: v / total_w for k, v in weights.items()}
        return weights

    def _compute_market_drawdown(self) -> pd.Series:
        """Compute equal-weighted market drawdown, shifted by 1 day.

        Value at D reflects D-1 close. Used for buy/sell decisions at D open,
        so must not contain D's close data.
        """
        close = self._daily_cache["close"].unstack()
        mkt_ret = close.pct_change().mean(axis=1)
        mkt_cum = (1 + mkt_ret).cumprod()
        mkt_peak = mkt_cum.expanding().max()
        dd = mkt_cum / mkt_peak - 1
        return dd.shift(1).dropna()

    # ── Derived Features (for GP terminals) ─────────────────────────────────

    @staticmethod
    def _add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add derived price/volume features needed by GP factor terminals.

        Same as the report Phase 2 derived columns.
        """
        if data.empty:
            return data

        _unstacked = data[["close", "volume", "amount", "high", "low"]].unstack()

        # Momentum
        for days in [5, 20, 60]:
            series = _unstacked["close"].pct_change(days).stack()
            series.name = f"ret_{days}d"
            data = data.join(series, how="left")

        # Volatility
        for days in [20, 60]:
            series = _unstacked["close"].pct_change().rolling(
                days, min_periods=max(1, days // 2)
            ).std().stack()
            series.name = f"vol_{days}d"
            data = data.join(series, how="left")

        # HL ratio
        series = ((_unstacked["high"] - _unstacked["low"]) / _unstacked["close"]).stack()
        series.name = "hl_ratio"
        data = data.join(series, how="left")

        # Amihud illiquidity
        daily_ret = _unstacked["close"].pct_change()
        series = (daily_ret.abs() / _unstacked["amount"].clip(lower=1) * 1e6).stack()
        series.name = "amihud"
        data = data.join(series, how="left")

        # Volume ratio
        series = (_unstacked["volume"] / _unstacked["volume"].rolling(
            20, min_periods=5
        ).mean()).stack()
        series.name = "vol_ratio"
        data = data.join(series, how="left")

        # Limit-up count (last 5 days) — for blow-off top detection
        if "pre_close" in data.columns and "price_limit_frac" in data.columns:
            pre = data["pre_close"].unstack()
            cl = _unstacked["close"]
            lim = data["price_limit_frac"].unstack()
            is_limit_up = cl >= pre * (1 + lim - 0.005)
            series = is_limit_up.rolling(window=5, min_periods=1).sum().stack()
            series.name = "limit_up_5d"
            data = data.join(series, how="left")

        return data

    # ── GP Factor Discovery ─────────────────────────────────────────────────

    def _run_gp_discovery(self, data: pd.DataFrame) -> list[str]:
        """Run GP evolution on training data, return validated new factor names."""
        from discovery.gp import GPEngine, GPConfig
        from discovery.validate import FactorValidator, orthogonal_filter

        # Split data into train/test
        all_dates = sorted(data.index.get_level_values("trade_date").unique())
        split_idx = int(len(all_dates) * 0.67)
        gp_train_dates = all_dates[:split_idx]

        logger.info("GP discovery: training on %s ~ %s (%d days)",
                     gp_train_dates[0].date(), gp_train_dates[-1].date(),
                     len(gp_train_dates))

        # Restrict to training data
        train_mask = data.index.get_level_values("trade_date").isin(gp_train_dates)
        gp_data = data.loc[train_mask]

        # Compute forward returns for GP fitness
        close = gp_data["close"].unstack()
        fwd_ret = close.pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()
        fwd_ret.name = "fwd_ret"

        # Compute baseline factor values on training data for GP
        engine = FactorEngine()
        existing_df = engine.compute(BASELINE_FACTORS, gp_data)

        # Align
        common_idx = existing_df.dropna().index.intersection(fwd_ret.dropna().index)
        gp_fwd = fwd_ret.loc[common_idx]
        gp_existing = existing_df.loc[common_idx]

        if len(gp_fwd) < 100:
            logger.warning("Too few GP training samples (%d), skipping", len(gp_fwd))
            return []

        # GP config
        gp_config = GPConfig(
            population_size=self.gp_population,
            max_generations=self.gp_generations,
            tournament_size=7,
            crossover_prob=0.7,
            mutation_prob=0.5,
            elite_count=10,
            max_depth=6,
            max_complexity=30,
            early_stop_generations=self.gp_early_stop,
            parsimony_penalty=0.001,
            ic_mean_weight=0.25,
            ic_ir_weight=0.35,
            stability_weight=0.25,
            hit_rate_weight=0.15,
        )
        gp = GPEngine(config=gp_config)

        t0 = time.time()
        best_individuals = gp.evolve(
            data=gp_data,
            forward_returns=gp_fwd,
            existing_factors=gp_existing,
        )
        logger.info("GP evolution complete: %.0fs, %d generations",
                     time.time() - t0, gp.generation)

        # Validate top candidates
        validator = FactorValidator()
        max_new = 5
        new_factors: list[str] = []
        validated_values: dict[str, pd.Series] = {}

        hall = sorted(best_individuals, key=lambda x: x.fitness, reverse=True)[:max_new * 2]
        for ind in hall:
            if ind.factor_cls is None or ind.fitness < -100:
                continue
            if len(new_factors) >= max_new:
                break

            try:
                factor_vals = ind.factor_cls().compute(data)
                result = validator.validate(
                    factor_values=factor_vals,
                    forward_returns=fwd_ret,
                    factor_name=ind.factor_name,
                    existing_factors=existing_df,
                )

                if result.passed:
                    new_factors.append(ind.factor_name)
                    existing_df[ind.factor_name] = factor_vals
                    validated_values[ind.factor_name] = factor_vals
                    logger.info("GP factor accepted: %s (fitness=%.4f, IC=%.4f, IR=%.3f)",
                                 ind.factor_name, ind.fitness, ind.ic_mean, ind.ic_ir)
                else:
                    logger.info("GP factor rejected: %s (%s)",
                                 ind.factor_name, ", ".join(result.failures[:2]))
            except Exception as e:
                logger.debug("GP factor error: %s — %s", ind.factor_name, e)

        # Orthogonal filter
        if len(new_factors) > 1:
            ortho_selected = orthogonal_filter(
                validated_values, fwd_ret, min_residual_ir=0.10,
            )
            rejected = set(validated_values) - set(ortho_selected)
            if rejected:
                logger.info("Orthogonal filter dropped: %s", rejected)
            new_factors = [f for f in new_factors if f in ortho_selected]

        # Category diversity: at most 1 per category
        seen_categories: set[str] = set()
        selected: list[str] = []
        for fname in new_factors:
            cat = "other"
            for ind in hall:
                if ind.factor_name == fname and ind.factor_cls is not None:
                    cat = ind.factor_cls.meta.category
                    break
            if cat not in seen_categories or len(selected) < 2:
                selected.append(fname)
                seen_categories.add(cat)
            if len(selected) >= 2:
                break

        # Register selected GP factors in global registry for FactorEngine
        from factor.registry import registry
        for ind in hall:
            if ind.factor_name in selected and ind.factor_cls is not None:
                try:
                    registry.register(ind.factor_cls)
                    logger.debug("Registered GP factor: %s", ind.factor_name)
                except ValueError:
                    logger.debug("GP factor already registered: %s", ind.factor_name)

        # Generate GP evolution report
        self._generate_gp_report(gp, selected)

        return selected

    def _generate_gp_report(
        self, gp: GPEngine, selected: list[str],
    ) -> None:
        """Generate console table and charts showing GP evolution progress."""
        history = gp.generation_history
        if not history:
            return

        chart_dir = Path(self.output_dir) / "gp_report"
        chart_dir.mkdir(parents=True, exist_ok=True)

        # ── Console table ──────────────────────────────────────────────────
        header = (
            f"{'Gen':>4s} {'BestFit':>8s} {'MeanFit':>8s} {'MedianFit':>8s} "
            f"{'BestIC':>7s} {'MeanIC':>7s} {'BestIR':>7s} {'MeanIR':>7s} "
            f"{'Depth':>5s} {'Nodes':>5s} {'Valid':>5s} {'HoF':>4s} {'Stall':>5s} {'Elap':>6s}"
        )
        sep = "-" * len(header)
        print(f"\n{' GP Evolution Progress '.center(len(header), '=')}")
        print(header)
        print(sep)

        for s in history:
            print(
                f"{s.generation:4d} {s.best_fitness:8.4f} {s.mean_fitness:8.4f} {s.median_fitness:8.4f} "
                f"{abs(s.best_ic):7.4f} {s.mean_ic:7.4f} {s.best_ir:7.3f} {s.mean_ir:7.3f} "
                f"{s.best_depth:5d} {s.best_nodes:5d} {s.valid_count:5d} {s.hall_of_fame_size:4d} {s.stall_count:5d} {s.elapsed_seconds:5.0f}s"
            )

        print(sep)
        print(f"Selected factors: {selected}")
        print(f"Total generations: {gp.generation}")

        # ── Charts ─────────────────────────────────────────────────────────
        gens = [s.generation for s in history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("GP Evolution Progress", fontsize=14, fontweight="bold")

        # 1) Fitness evolution
        ax = axes[0, 0]
        ax.plot(gens, [s.best_fitness for s in history], "r-o", label="Best", markersize=5)
        ax.plot(gens, [s.mean_fitness for s in history], "b-s", label="Mean", markersize=5)
        ax.plot(gens, [s.median_fitness for s in history], "g--^", label="Median", markersize=5)
        ax.fill_between(
            gens,
            [s.mean_fitness - s.std_fitness for s in history],
            [s.mean_fitness + s.std_fitness for s in history],
            alpha=0.15, color="b",
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Evolution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2) IC & IR evolution (best)
        ax = axes[0, 1]
        ax.plot(gens, [abs(s.best_ic) for s in history], "r-o", label="Best |IC|", markersize=5)
        ax.plot(gens, [s.mean_ic for s in history], "r--s", label="Mean |IC|", markersize=5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("|IC|", color="r")
        ax.tick_params(axis="y", labelcolor="r")
        ax2 = ax.twinx()
        ax2.plot(gens, [max(0, s.best_ir) for s in history], "b-o", label="Best IR", markersize=5)
        ax2.plot(gens, [max(0, s.mean_ir) for s in history], "b--s", label="Mean IR", markersize=5)
        ax2.set_ylabel("IC IR", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        ax.set_title("IC & IR Evolution")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 3) Complexity
        ax = axes[1, 0]
        ax.plot(gens, [s.best_depth for s in history], "r-o", label="Best Depth", markersize=5)
        ax.plot(gens, [s.mean_depth for s in history], "r--s", label="Mean Depth", markersize=5)
        ax.plot(gens, [s.best_nodes for s in history], "b-o", label="Best Nodes", markersize=5)
        ax.plot(gens, [s.mean_nodes for s in history], "b--s", label="Mean Nodes", markersize=5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Count")
        ax.set_title("Complexity (Depth & Nodes)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4) Population health + HoF
        ax = axes[1, 1]
        ax.bar(gens, [s.valid_count for s in history], color="g", alpha=0.6, label="Valid")
        ax.bar(gens, [s.total_count - s.valid_count for s in history],
               bottom=[s.valid_count for s in history], color="r", alpha=0.4, label="Invalid")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Population")
        ax.set_title("Population Health")
        ax2 = ax.twinx()
        ax2.plot(gens, [s.hall_of_fame_size for s in history], "b-D", label="HoF Size", markersize=5)
        ax2.set_ylabel("Hall of Fame Size", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        report_path = chart_dir / "gp_evolution.png"
        fig.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("GP evolution report saved: %s", report_path)

    # ── Factor Composite ────────────────────────────────────────────────────

    def _compute_day_composite(self, td: date) -> pd.Series:
        """Compute IC_IR-weighted rank-normalized factor composite for a given date."""
        try:
            day_factors = self._factor_df.xs(pd.Timestamp(td), level="trade_date")
        except KeyError:
            try:
                day_factors = self._factor_df.xs(pd.Timestamp(td - timedelta(days=1)), level="trade_date")
            except KeyError:
                return pd.Series(dtype=float)

        composite = pd.Series(0.0, index=day_factors.index)
        for fname, weight in self._factor_weights.items():
            if fname not in day_factors.columns:
                continue
            col = day_factors[fname].dropna()
            if len(col) < 5:
                continue
            ranked = col.rank(pct=True)
            composite = composite.add(ranked * weight, fill_value=0.0)

        composite.name = "composite_score"
        return composite[composite > 0]

    # ── Candidate Selection ─────────────────────────────────────────────────

    def _get_candidates(
        self,
        td: date,
        universe: list[str],
        yest_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the candidate stock list ranked by calibrated factor composite."""
        # Compute composite for today (or yesterday if today not available)
        composite = self._compute_day_composite(td)
        if composite.empty:
            composite = self._compute_day_composite(td - timedelta(days=1))
        if composite.empty:
            return pd.DataFrame()

        # Filter to universe
        valid = composite.index.intersection(universe)
        composite = composite.loc[valid]

        # Take top candidates (enough for ranking + some buffer)
        max_candidates = max(MAX_POSITIONS * 5, 50)
        if len(composite) > max_candidates:
            composite = composite.nlargest(max_candidates)

        # Enrich with recent price data
        rows = []
        for sym in composite.index:
            sym_data = yest_data[yest_data.index == sym] if not yest_data.empty else pd.DataFrame()
            row = {
                "symbol": sym,
                "composite_score": round(float(composite.loc[sym]), 4),
            }
            if not sym_data.empty:
                row["close"] = round(float(sym_data["close"].iloc[0]), 2) if "close" in sym_data.columns else None
                row["pre_close"] = round(float(sym_data["pre_close"].iloc[0]), 2) if "pre_close" in sym_data.columns else None
                if row["close"] and row["pre_close"] and row["pre_close"] > 0:
                    row["change_pct"] = round((row["close"] - row["pre_close"]) / row["pre_close"] * 100, 2)
                else:
                    row["change_pct"] = None
                row["volume"] = int(sym_data["volume"].iloc[0]) if "volume" in sym_data.columns else None
                row["amount"] = round(float(sym_data["amount"].iloc[0]), 0) if "amount" in sym_data.columns else None
                row["turnover"] = round(float(sym_data["turnover"].iloc[0]), 2) if "turnover" in sym_data.columns else None
            rows.append(row)

        return pd.DataFrame(rows)

    # ── LLM Decision ────────────────────────────────────────────────────────

    def _llm_decide(
        self,
        td: date,
        candidates: pd.DataFrame,
        yest_data: pd.DataFrame,
    ) -> dict:
        """Ask the LLM to make trading decisions."""
        # Build portfolio summary
        positions = self._portfolio_summary(yest_data)
        equity = self._equity if self._equity > 0 else self.initial_cash
        cash = self.accountant.cash
        cash_pct = cash / equity * 100 if equity > 0 else 100

        # Build candidates table
        cand_lines = []
        for _, row in candidates.iterrows():
            parts = [
                f"{row['symbol']}",
                f"score={row['composite_score']:.4f}",
            ]
            if row.get("close"):
                parts.append(f"close={row['close']:.2f}")
            if row.get("change_pct") is not None:
                parts.append(f"chg={row['change_pct']:+.2f}%")
            if row.get("turnover") is not None:
                parts.append(f"turn={row['turnover']:.1f}%")
            cand_lines.append(" | ".join(parts))

        candidates_text = "\n".join(cand_lines) if cand_lines else "(no candidates)"

        # Build positions summary
        if positions:
            pos_lines = []
            for p in positions:
                pos_lines.append(
                    f"  {p['symbol']}: {p['shares']}股, "
                    f"成本{p['avg_cost']:.2f}, 现价{p['current_price']:.2f}, "
                    f"盈亏{p['pnl_pct']:+.2f}% (¥{p['pnl_amount']:+.0f}), "
                    f"市值¥{p['market_value']:.0f}"
                )
            pos_text = "\n".join(pos_lines)
        else:
            pos_text = "  (empty portfolio)"

        prompt = f"""## Today: {td}

### Portfolio
Cash: ¥{cash:,.0f} ({cash_pct:.1f}% of equity)
Total Equity: ¥{equity:,.0f}
Positions ({len(positions)}/{MAX_POSITIONS}):
{pos_text}

### Candidate Stocks (ranked by factor composite)
{candidates_text}

### Instructions
Decide your trades for today. Output JSON with "buys" and "sells" arrays."""

        try:
            response = self.llm.chat_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                expected_keys=["buys", "sells"],
            )
            logger.info("LLM decision on %s: buys=%d, sells=%d",
                         td,
                         len(response.get("buys", [])),
                         len(response.get("sells", [])))
            response["date"] = td.isoformat()
            return response
        except Exception as e:
            logger.warning("LLM call failed on %s: %s. Using heuristic.", td, e)
            return self._heuristic_decide(td, candidates, yest_data)

    # ── Factor Strategy (Pure signal-driven) ────────────────────────────────

    def _factor_decide(
        self,
        td: date,
        candidates: pd.DataFrame,
        yest_data: pd.DataFrame,
    ) -> dict:
        """Factor-driven strategy: buys at close, sells at open.

        Uses D's factor composite (shifted = D-1 real data) for both decisions.
        Buys at D close (1-day info lag, safe), sells at D open (no lag).
        """
        positions = self._portfolio_summary(yest_data)
        equity = max(self._equity, self.initial_cash)
        cash = self.accountant.cash

        if candidates.empty:
            return {"date": td.isoformat(), "reasoning": "no candidates", "buys": [], "sells": []}

        # ── Market drawdown circuit breaker ──
        mkt_dd = abs(self._market_dd.get(pd.Timestamp(td), 0.0)) if hasattr(self, '_market_dd') else 0.0
        skip_buys = mkt_dd > MKT_DD_THRESHOLD

        ranked = candidates.sort_values("composite_score", ascending=False)
        buy_top = set(ranked.head(MAX_POSITIONS)["symbol"].tolist())      # top-1 for buying
        sell_top = set(ranked.head(SELL_RANK_LIMIT)["symbol"].tolist())   # top-5 for holding
        held_symbols = {p["symbol"] for p in positions}

        sells = []
        buys = []

        # Take-profit: lock in gains on high-flyers before blow-off reversals
        for p in positions:
            if p.get("pnl_pct", 0) > TAKE_PROFIT_PCT * 100:
                sells.append({
                    "symbol": p["symbol"],
                    "shares": p["shares"],
                    "reason": f"take profit ({p['pnl_pct']:+.1f}% > {TAKE_PROFIT_PCT*100:.0f}%)",
                })

        # Sell if rank drops below SELL_RANK_LIMIT (skip already-sold by take-profit)
        already_sold = {s["symbol"] for s in sells}
        for p in positions:
            if p["symbol"] not in sell_top and p["symbol"] not in already_sold:
                sells.append({
                    "symbol": p["symbol"],
                    "shares": p["shares"],
                    "reason": f"fell out of top-{SELL_RANK_LIMIT} (rank too low)",
                })

        # Market drawdown: skip new buys, let existing positions ride
        if skip_buys:
            return {
                "date": td.isoformat(),
                "reasoning": f"Factor: market DD {mkt_dd:.1%} > {MKT_DD_THRESHOLD:.0%} — no new buys ({len(sells)} sells)",
                "buys": [],
                "sells": sells,
            }

        # Buy top-N stocks not already held
        available_slots = MAX_POSITIONS - len(held_symbols) + len(sells)
        buy_budget = cash + sum(
            s["shares"] * (yest_data.loc[s["symbol"], "close"] if s["symbol"] in yest_data.index else 0)
            for s in sells
            if "close" in yest_data.columns
        )

        per_position_cash = buy_budget * 0.85 / max(available_slots, 1)

        for _, row in ranked.iterrows():
            sym = row["symbol"]
            if sym in held_symbols and sym not in {s["symbol"] for s in sells}:
                continue  # already held
            if len(buys) >= available_slots:
                break
            price = row.get("close", None)
            if not price or price <= 0:
                continue
            max_value = min(equity * MAX_POSITION_PCT, per_position_cash)
            shares = int(max_value / price)
            shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares >= LOT_SIZE:
                buys.append({
                    "symbol": sym,
                    "shares": shares,
                    "reason": f"top-{MAX_POSITIONS} factor composite ({row['composite_score']:.4f})",
                })

        return {
            "date": td.isoformat(),
            "reasoning": f"Factor: buy top-{MAX_POSITIONS} by composite score. {len(buys)} buys, {len(sells)} sells.",
            "buys": buys,
            "sells": sells,
        }

    # ── Heuristic Fallback ──────────────────────────────────────────────────

    def _heuristic_decide(
        self,
        td: date,
        candidates: pd.DataFrame,
        yest_data: pd.DataFrame,
    ) -> dict:
        """Simple heuristic strategy when LLM is unavailable."""
        positions = self._portfolio_summary(yest_data)
        equity = max(self._equity, self.initial_cash)
        cash = self.accountant.cash

        buys = []
        sells = []

        # ── Market drawdown circuit breaker ──
        mkt_dd = abs(self._market_dd.get(pd.Timestamp(td), 0.0)) if hasattr(self, '_market_dd') else 0.0

        # Check existing positions for sells
        for p in positions:
            # Sell if loss > 8% or profit > 20%
            if p["pnl_pct"] < -8.0:
                sells.append({
                    "symbol": p["symbol"],
                    "shares": p["shares"],
                    "reason": f"stop loss ({p['pnl_pct']:+.1f}%)",
                })
            elif p["pnl_pct"] > 20.0:
                sells.append({
                    "symbol": p["symbol"],
                    "shares": p["shares"],
                    "reason": f"take profit ({p['pnl_pct']:+.1f}%)",
                })

        # Market drawdown: skip new buys, let existing positions ride
        if mkt_dd > MKT_DD_THRESHOLD:
            return {
                "date": td.isoformat(),
                "reasoning": f"Heuristic: market DD {mkt_dd:.1%} > {MKT_DD_THRESHOLD:.0%} — no new buys ({len(sells)} sells)",
                "buys": [],
                "sells": sells,
            }

        # Buy top candidates if we have cash and spare slots
        available_slots = MAX_POSITIONS - len(positions) + len(sells)
        if available_slots > 0 and cash > equity * 0.05 and not candidates.empty:
            sorted_cands = candidates.sort_values("composite_score", ascending=False)
            for _, row in sorted_cands.head(available_slots).iterrows():
                sym = row["symbol"]
                if sym in [p["symbol"] for p in positions] or sym in [s["symbol"] for s in sells]:
                    continue
                price = row.get("close", 10.0)
                if not price or price <= 0:
                    continue
                max_value = min(equity * MAX_POSITION_PCT, cash * 0.8 / min(available_slots, 3))
                shares = int(max_value / price)
                shares = (shares // LOT_SIZE) * LOT_SIZE
                if shares >= LOT_SIZE:
                    buys.append({
                        "symbol": sym,
                        "shares": shares,
                        "reason": f"top composite score ({row['composite_score']:.4f})",
                    })

        return {
            "date": td.isoformat(),
            "reasoning": f"Heuristic: {len(buys)} buys, {len(sells)} sells based on factor composite + stop-loss/take-profit rules",
            "buys": buys,
            "sells": sells,
        }

    # ── Portfolio Summary ───────────────────────────────────────────────────

    def _portfolio_summary(self, data: pd.DataFrame) -> list[dict]:
        """Build current portfolio summary with P&L."""
        positions = []
        for sym, shares in self.accountant.positions.items():
            if shares <= 0:
                continue
            current_price = 0.0
            if not data.empty and sym in data.index:
                current_price = float(data.loc[sym, "close"]) if "close" in data.columns else 0.0

            # Compute average cost from broker's lot tracking
            lots = self.broker._position_lots.get(sym, [])
            if lots:
                total_cost = sum(l["shares"] * l["avg_price"] for l in lots)
                total_shares = sum(l["shares"] for l in lots)
                avg_cost = total_cost / total_shares if total_shares > 0 else 0.0
            else:
                avg_cost = current_price

            market_value = shares * current_price
            pnl_amount = (current_price - avg_cost) * shares
            pnl_pct = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0.0

            positions.append({
                "symbol": sym,
                "shares": shares,
                "avg_cost": round(avg_cost, 2),
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 0),
                "pnl_amount": round(pnl_amount, 0),
                "pnl_pct": round(pnl_pct, 2),
            })

        return sorted(positions, key=lambda x: x["pnl_pct"], reverse=True)

    # ── Orders from Decisions ───────────────────────────────────────────────

    def _decisions_to_orders(
        self,
        decision: dict,
        td: date,
        prices: pd.Series,
        universe: list[str],
    ) -> list[Order]:
        """Convert LLM decision JSON to Order objects, enforcing cash limits."""
        orders: list[Order] = []
        today = pd.Timestamp(td)

        # Estimate cash from sells first
        sell_proceeds = 0.0
        for s in decision.get("sells", []):
            sym = s.get("symbol", "").strip()
            shares = int(s.get("shares", 0))
            if sym not in universe or shares <= 0:
                continue
            if sym not in prices.index or pd.isna(prices.get(sym)) or prices.get(sym) <= 0:
                continue
            available = self.broker.sellable_shares(sym, today)
            shares = min(shares, available)
            shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares <= 0:
                continue
            price = float(prices[sym])
            sell_proceeds += shares * price * 0.998  # net of fees

        available_cash = self.accountant.cash + sell_proceeds
        committed_buy_cost = 0.0
        current_positions = len([p for p in self.accountant.positions.values() if p > 0])

        # Parse sells
        for s in decision.get("sells", []):
            sym = s.get("symbol", "").strip()
            shares = int(s.get("shares", 0))
            if sym not in universe or shares <= 0:
                continue
            if sym not in prices.index or pd.isna(prices.get(sym)) or prices.get(sym) <= 0:
                continue
            available = self.broker.sellable_shares(sym, today)
            shares = min(shares, available)
            shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares <= 0:
                continue
            orders.append(Order(
                symbol=sym, side=Side.SELL, quantity=shares,
                order_type=OrderType.MARKET, date=today,
                order_id=f"sim_{td}_{sym}_sell_{self._fill_count}",
            ))
            self._fill_count += 1
            current_positions -= 1

        # Parse buys with cash tracking
        for b in decision.get("buys", []):
            sym = b.get("symbol", "").strip()
            shares = int(b.get("shares", 0))
            if sym not in universe or shares <= 0:
                continue
            if sym not in prices.index or pd.isna(prices.get(sym)) or prices.get(sym) <= 0:
                continue
            shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares <= 0:
                continue

            # Position limit check
            if current_positions >= MAX_POSITIONS and sym not in self.accountant.positions:
                continue

            # Cash limit: max 15% of equity or remaining cash
            price = float(prices[sym])
            max_per_position = self._equity * MAX_POSITION_PCT
            remaining_slots = max(1, MAX_POSITIONS - current_positions)
            max_from_cash = (available_cash - committed_buy_cost) / remaining_slots
            max_value = min(max_per_position, max_from_cash * 1.1)  # slight buffer

            cost = shares * price
            if cost > max_value:
                shares = int(max_value / price)
                shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares <= 0:
                continue

            cost = shares * price
            if committed_buy_cost + cost > available_cash * 0.95:
                # Try smaller size
                remaining = available_cash * 0.95 - committed_buy_cost
                if remaining <= 0:
                    continue
                shares = int(remaining / price)
                shares = (shares // LOT_SIZE) * LOT_SIZE
                if shares <= 0:
                    continue

            orders.append(Order(
                symbol=sym, side=Side.BUY, quantity=shares,
                order_type=OrderType.MARKET, date=today,
                order_id=f"sim_{td}_{sym}_buy_{self._fill_count}",
            ))
            self._fill_count += 1
            committed_buy_cost += shares * price
            current_positions += 1

        return orders

    # ── Journal ─────────────────────────────────────────────────────────────

    def _record_day(
        self,
        td: date,
        fills: list[Fill],
        reasoning: str,
    ) -> None:
        """Record daily journal entry and print trade details to console."""
        positions = []
        for sym, shares in self.accountant.positions.items():
            if shares > 0:
                lots = self.broker._position_lots.get(sym, [])
                avg_cost = sum(l["shares"] * l["avg_price"] for l in lots) / sum(l["shares"] for l in lots) if lots else 0
                positions.append({
                    "symbol": sym,
                    "shares": shares,
                    "avg_cost": round(avg_cost, 2),
                })

        fill_records = []
        for f in fills:
            fill_records.append({
                "symbol": f.symbol,
                "side": f.side.value,
                "shares": f.quantity,
                "price": round(f.price, 2),
                "commission": round(f.commission, 2),
                "stamp_tax": round(f.stamp_tax, 2),
            })

        # Console output: each trade on its own line
        if fills:
            print(f"\n  [{td}] 权益=¥{self._equity:,.0f}  现金=¥{self.accountant.cash:,.0f}  交易={len(fills)}笔")
            for f_rec in fill_records:
                side_tag = "BUY " if f_rec["side"] == "buy" else "SELL"
                arrow = "←" if f_rec["side"] == "buy" else "→"
                cost_str = f"  手续费=¥{f_rec['commission']:.2f}" + (f" 印花税=¥{f_rec['stamp_tax']:.2f}" if f_rec["stamp_tax"] > 0 else "")
                print(f"    {side_tag} {f_rec['symbol']}  {f_rec['shares']}股  @¥{f_rec['price']:.2f}  {arrow} ¥{f_rec['shares'] * f_rec['price']:,.0f}{cost_str}")
            if reasoning:
                print(f"    💭 {reasoning[:120]}")
        elif reasoning:
            print(f"\n  [{td}] 权益=¥{self._equity:,.0f}  无交易  💭 {reasoning[:120]}")

        self._journal.append({
            "date": td.isoformat(),
            "cash": round(self.accountant.cash, 2),
            "equity": round(self._equity, 2),
            "reasoning": reasoning,
            "fills": fill_records,
            "positions": positions,
        })

    # ── K-Line Charts ──────────────────────────────────────────────────────

    def _generate_trade_charts(self, chart_dir: Path) -> Path:
        """Generate K-line charts for all traded stocks with buy/sell markers."""
        import mplfinance as mpf

        # A-share convention: red = up (涨), green = down (跌)
        a_share_style = mpf.make_mpf_style(
            marketcolors=mpf.make_marketcolors(
                up='red', down='green',
                edge='inherit', wick='inherit', volume='inherit',
            ),
            gridstyle='-', gridaxis='horizontal', gridcolor='#e0e0e0',
            rc=get_mpl_rc(),
        )
        BUY_COLOR = "#cc0000"
        SELL_COLOR = "#00aa00"

        # Collect all fills by symbol
        trades_by_symbol: dict[str, list[dict]] = {}
        for entry in self._journal:
            for f_rec in entry.get("fills", []):
                sym = f_rec["symbol"]
                if sym not in trades_by_symbol:
                    trades_by_symbol[sym] = []
                trades_by_symbol[sym].append({
                    "date": pd.Timestamp(entry["date"]),
                    "side": f_rec["side"],
                    "shares": f_rec["shares"],
                    "price": f_rec["price"],
                })

        if not trades_by_symbol:
            logger.info("No trades to chart")
            return chart_dir

        chart_dir.mkdir(parents=True, exist_ok=True)

        all_trade_dates = sorted({
            t["date"] for trades in trades_by_symbol.values() for t in trades
        })
        chart_start = all_trade_dates[0] - pd.Timedelta(days=30)
        chart_end = all_trade_dates[-1] + pd.Timedelta(days=10)

        if self._daily_cache is None:
            return chart_dir

        total_syms = len(trades_by_symbol)
        # 2x2 grid — 4 stocks per page for readability
        n_cols = 2
        n_rows = 2
        per_page = n_cols * n_rows
        syms_list = sorted(trades_by_symbol.keys(), key=lambda s: len(trades_by_symbol[s]), reverse=True)
        total_pages = (total_syms + per_page - 1) // per_page

        logger.info("Generating K-line charts for %d symbols ...", total_syms)

        for page_start in range(0, total_syms, per_page):
            page_syms = syms_list[page_start:page_start + per_page]
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(14, 10),
                squeeze=False,
            )
            fig.suptitle(
                f"Trade Charts — {self.mode.upper()}   ({self.start} → {self.end})",
                fontsize=15, fontweight="bold", y=0.99,
            )
            fig.text(0.50, 0.975, "红涨绿跌 | ▲ 买入 (Buy)   ▼ 卖出 (Sell)",
                     ha="center", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

            for idx, sym in enumerate(page_syms):
                row, col = divmod(idx, n_cols)
                ax = axes[row][col]

                try:
                    sym_data = self._daily_cache.xs(sym, level="symbol")
                    sym_data = sym_data.loc[
                        (sym_data.index >= chart_start) & (sym_data.index <= chart_end)
                    ]
                except KeyError:
                    ax.text(0.5, 0.5, f"{sym}\nNo data", transform=ax.transAxes, ha="center", fontsize=11)
                    ax.set_title(sym, fontsize=12, fontweight="bold")
                    continue

                if sym_data.empty:
                    ax.text(0.5, 0.5, f"{sym}\nNo data in range", transform=ax.transAxes, ha="center", fontsize=11)
                    ax.set_title(sym, fontsize=12, fontweight="bold")
                    continue

                ohlc_cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
                ohlc = sym_data[["open", "high", "low", "close", "volume"]].rename(columns=ohlc_cols)
                ohlc = ohlc.dropna(subset=["Open", "High", "Low", "Close"])

                if ohlc.empty:
                    ax.text(0.5, 0.5, f"{sym}\nNo valid OHLC", transform=ax.transAxes, ha="center", fontsize=11)
                    ax.set_title(sym, fontsize=12, fontweight="bold")
                    continue

                trades = trades_by_symbol[sym]
                buy_dates = [t["date"] for t in trades if t["side"] == "buy"]
                buy_prices = [t["price"] for t in trades if t["side"] == "buy"]
                sell_dates = [t["date"] for t in trades if t["side"] == "sell"]
                sell_prices = [t["price"] for t in trades if t["side"] == "sell"]

                buy_markers = None
                sell_markers = None
                if buy_dates:
                    buy_markers = pd.DataFrame({"price": buy_prices}, index=pd.DatetimeIndex(buy_dates))
                if sell_dates:
                    sell_markers = pd.DataFrame({"price": sell_prices}, index=pd.DatetimeIndex(sell_dates))

                addplots = []
                if buy_markers is not None and not buy_markers.empty:
                    buy_aligned = buy_markers.reindex(ohlc.index)
                    buy_series = buy_aligned["price"].where(buy_aligned["price"].notna(), other=np.nan)
                    addplots.append(mpf.make_addplot(
                        buy_series, type="scatter", marker="^",
                        color=BUY_COLOR, markersize=120, ax=ax,
                    ))
                if sell_markers is not None and not sell_markers.empty:
                    sell_aligned = sell_markers.reindex(ohlc.index)
                    sell_series = sell_aligned["price"].where(sell_aligned["price"].notna(), other=np.nan)
                    addplots.append(mpf.make_addplot(
                        sell_series, type="scatter", marker="v",
                        color=SELL_COLOR, markersize=120, ax=ax,
                    ))

                try:
                    mpf.plot(
                        ohlc, type="candle", style=a_share_style,
                        addplot=addplots if addplots else None,
                        ax=ax, volume=False,
                        show_nontrading=False,
                    )
                except Exception as e:
                    logger.debug("mplfinance error for %s: %s", sym, e)
                    ax.plot(ohlc.index, ohlc["Close"], color="black", linewidth=1.2)
                    if buy_markers is not None:
                        ax.scatter(buy_markers.index, buy_markers["price"],
                                   marker="^", color=BUY_COLOR, s=120, zorder=5)
                    if sell_markers is not None:
                        ax.scatter(sell_markers.index, sell_markers["price"],
                                   marker="v", color=SELL_COLOR, s=120, zorder=5)

                for t in trades:
                    trade_date = t["date"]
                    if trade_date in ohlc.index:
                        y_pos = t["price"]
                        label = f"{'买入' if t['side'] == 'buy' else '卖出'}\n@{t['price']:.2f}"
                        color = BUY_COLOR if t["side"] == "buy" else SELL_COLOR
                        ax.annotate(
                            label,
                            xy=(trade_date, y_pos),
                            xytext=(0, 18 if t["side"] == "buy" else -20),
                            textcoords="offset points",
                            fontsize=7, color=color, fontweight="bold",
                            ha="center",
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
                        )

                buy_count = sum(1 for t in trades if t["side"] == "buy")
                sell_count = len(trades) - buy_count
                ax.set_title(f"{sym}  (买入{buy_count} 卖出{sell_count})", fontsize=12, fontweight="bold")
                ax.set_ylabel("Price", fontsize=9)

            # Hide unused subplots
            for idx in range(len(page_syms), n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            page_num = page_start // per_page + 1
            chart_path = chart_dir / f"charts_p{page_num:02d}.png"
            fig.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Charts page %d/%d: %d symbols", page_num, total_pages, len(page_syms))

        # ── Detail charts for top 10 most-traded stocks ─────────────────
        top_syms = syms_list[:10]
        self._generate_detail_charts(top_syms, trades_by_symbol,
                                     chart_dir, chart_start, chart_end,
                                     a_share_style, BUY_COLOR, SELL_COLOR)

        logger.info("Charts saved to %s (%d symbols total)", chart_dir, total_syms)
        return chart_dir

    def _generate_detail_charts(self, top_syms, trades_by_symbol, chart_dir,
                                 chart_start, chart_end, style, buy_color, sell_color):
        """Generate individual large-format charts for top-traded stocks."""
        import mplfinance as mpf

        for sym in top_syms:
            try:
                sym_data = self._daily_cache.xs(sym, level="symbol")
                sym_data = sym_data.loc[
                    (sym_data.index >= chart_start) & (sym_data.index <= chart_end)
                ]
            except KeyError:
                continue
            if sym_data.empty:
                continue

            ohlc_cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            ohlc = sym_data[["open", "high", "low", "close", "volume"]].rename(columns=ohlc_cols)
            ohlc = ohlc.dropna(subset=["Open", "High", "Low", "Close"])
            if ohlc.empty:
                continue

            # Compute MAs
            ma5 = ohlc["Close"].rolling(5).mean()
            ma20 = ohlc["Close"].rolling(20).mean()

            trades = trades_by_symbol[sym]
            buy_dates = [t["date"] for t in trades if t["side"] == "buy"]
            buy_prices = [t["price"] for t in trades if t["side"] == "buy"]
            sell_dates = [t["date"] for t in trades if t["side"] == "sell"]
            sell_prices = [t["price"] for t in trades if t["side"] == "sell"]

            buy_markers = None
            sell_markers = None
            if buy_dates:
                buy_markers = pd.DataFrame({"price": buy_prices}, index=pd.DatetimeIndex(buy_dates))
            if sell_dates:
                sell_markers = pd.DataFrame({"price": sell_prices}, index=pd.DatetimeIndex(sell_dates))

            addplots = [
                mpf.make_addplot(ma5, color="#ff9800", width=1.2, label="MA5"),
                mpf.make_addplot(ma20, color="#2196f3", width=1.2, label="MA20"),
            ]
            if buy_markers is not None and not buy_markers.empty:
                buy_aligned = buy_markers.reindex(ohlc.index)
                buy_series = buy_aligned["price"].where(buy_aligned["price"].notna(), other=np.nan)
                addplots.append(mpf.make_addplot(
                    buy_series, type="scatter", marker="^",
                    color=buy_color, markersize=150,
                ))
            if sell_markers is not None and not sell_markers.empty:
                sell_aligned = sell_markers.reindex(ohlc.index)
                sell_series = sell_aligned["price"].where(sell_aligned["price"].notna(), other=np.nan)
                addplots.append(mpf.make_addplot(
                    sell_series, type="scatter", marker="v",
                    color=sell_color, markersize=150,
                ))

            fig, axes = mpf.plot(
                ohlc, type="candle", style=style,
                addplot=addplots,
                volume=True,
                show_nontrading=False,
                title=f"{sym} — 交易明细",
                ylabel="Price",
                ylabel_lower="Volume",
                returnfig=True,
                figsize=(14, 8),
            )

            # Annotate trades on the main price axis
            ax_main = axes[0]
            for t in trades:
                trade_date = t["date"]
                if trade_date in ohlc.index:
                    y_pos = t["price"]
                    label = f"{'买入' if t['side'] == 'buy' else '卖出'} @{t['price']:.2f}"
                    color = buy_color if t["side"] == "buy" else sell_color
                    ax_main.annotate(
                        label,
                        xy=(trade_date, y_pos),
                        xytext=(10, 20 if t["side"] == "buy" else -22),
                        textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                    )

            fig.savefig(chart_dir / f"detail_{sym}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Detail chart: %s", sym)

    # ── Benchmark Helpers ────────────────────────────────────────────────────

    def _fetch_benchmark_indices(self) -> dict[str, pd.Series]:
        """Fetch daily close prices for major A-share indices via baostock.

        Returns dict of {label: normalized_price_series} aligned to trading days.
        """
        import baostock as bs

        indices = {
            "沪深300": "sh.000300",
            "上证指数": "sh.000001",
            "中证500": "sh.000905",
        }

        result: dict[str, pd.Series] = {}
        start_str = self.start.isoformat()
        end_str = self.end.isoformat()

        try:
            bs.login()
        except Exception as e:
            logger.warning("Baostock login failed: %s", e)
            return result

        for label, code in indices.items():
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,close",
                    start_date=start_str,
                    end_date=end_str,
                    frequency="d",
                    adjustflag="3",
                )
                if rs.error_code != "0":
                    logger.warning("Benchmark %s query error: %s", label, rs.error_msg)
                    continue

                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    logger.warning("Benchmark %s: no data returned", label)
                    continue

                df = pd.DataFrame(rows, columns=["date", "close"])
                df["close"] = df["close"].astype(float)
                df["date"] = pd.to_datetime(df["date"])
                series = df.set_index("date")["close"].sort_index()

                if len(series) > 1 and series.iloc[0] > 0:
                    series = series / series.iloc[0]
                result[label] = series
                logger.info("Benchmark %s: %d data points (%.4f → %.4f)",
                            label, len(series), series.iloc[0], series.iloc[-1])
            except Exception as e:
                logger.warning("Benchmark %s fetch failed: %s", label, e)

        try:
            bs.logout()
        except Exception:
            pass

        return result

    # ── Equity & Position Chart ──────────────────────────────────────────────

    def _generate_equity_position_chart(self, equity_series: pd.Series, sim_dir: Path) -> Path:
        """Generate equity curve vs benchmarks + position overview chart."""
        chart_path = sim_dir / "equity_position.png"

        # Build daily position summary from journal
        dates = []
        equities = []
        position_counts = []
        position_values = []

        for entry in self._journal:
            dates.append(pd.Timestamp(entry["date"]))
            equities.append(entry["equity"])
            pos = entry.get("positions", [])
            position_counts.append(len(pos))
            position_values.append(entry["equity"] - entry["cash"])

        if len(dates) < 2:
            logger.warning("Not enough data points for equity chart")
            return chart_path

        # Normalize strategy equity to start at 1.0
        norm_equity = pd.Series(equities, index=pd.DatetimeIndex(dates)) / self.initial_cash

        # Fetch benchmark indices
        benchmarks = self._fetch_benchmark_indices()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9),
            gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08},
            sharex=True,
        )

        mode_label = {"factor": "因子策略", "llm": "LLM策略", "compare": "对比策略"}.get(
            self.mode, self.mode.upper())

        # ── Top: Equity Curve vs Benchmarks (normalized) ──────────────────
        # Strategy equity line
        ax1.plot(dates, norm_equity.values, color="#2c3e50", linewidth=2.5,
                 label=f"{mode_label}", zorder=10)
        ax1.fill_between(dates, norm_equity.values, 1.0,
                         where=norm_equity.values >= 1.0, alpha=0.10, color="#27ae60")
        ax1.fill_between(dates, norm_equity.values, 1.0,
                         where=norm_equity.values < 1.0, alpha=0.10, color="#e74c3c")
        ax1.axhline(y=1.0, color="#7f8c8d", linewidth=0.8,
                    linestyle="--", alpha=0.5)

        # Benchmark index curves
        bm_colors = {"沪深300": "#e74c3c", "上证指数": "#3498db", "中证500": "#e67e22"}
        bm_styles = {"沪深300": "-", "上证指数": "--", "中证500": "-."}
        for label, series in benchmarks.items():
            # Align to same date range
            aligned = series.reindex(norm_equity.index).dropna()
            if len(aligned) > 1:
                # Normalize all benchmarks to the same starting point as strategy
                aligned = aligned / aligned.iloc[0]
                color = bm_colors.get(label, "#95a5a6")
                style = bm_styles.get(label, "--")
                ax1.plot(aligned.index, aligned.values, color=color,
                         linewidth=1.3, linestyle=style, alpha=0.75, label=label)

        # Mark peak / final
        final_nav = norm_equity.values[-1]
        ax1.scatter(dates[-1], final_nav, color="#2c3e50", s=60, zorder=12)
        ax1.annotate(
            f"策略 {final_nav:.3f}",
            xy=(dates[-1], final_nav),
            xytext=(15, 0), textcoords="offset points",
            fontsize=9, color="#2c3e50", fontweight="bold", ha="left", va="center",
        )

        # Annotate benchmark final values
        y_offset = -0.025
        for label, series in benchmarks.items():
            aligned = series.reindex(norm_equity.index).dropna()
            if len(aligned) > 1:
                aligned = aligned / aligned.iloc[0]
                bm_final = aligned.values[-1]
                ax1.annotate(
                    f"{label} {bm_final:.3f}",
                    xy=(aligned.index[-1], bm_final),
                    xytext=(15, y_offset), textcoords="offset points",
                    fontsize=8, color=bm_colors.get(label, "#95a5a6"),
                    ha="left", va="top",
                )
                y_offset -= 18

        ax1.set_ylabel("净值 (1.0 = 初始资金)", fontsize=11)
        ax1.set_title(f"{mode_label} — 权益曲线 vs 大盘指数  ({self.start} → {self.end})",
                      fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
        ax1.grid(True, linestyle="--", alpha=0.25)

        # ── Bottom: Position Composition ──────────────────────────────────
        ax2.fill_between(dates, position_values, alpha=0.4, color="#3498db", label="持仓市值")
        ax2.plot(dates, position_values, color="#2980b9", linewidth=1.5, drawstyle="steps-post")
        ax2.set_ylabel("持仓市值 (¥)", fontsize=11, color="#2980b9")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))

        # Second axis: position count
        ax2b = ax2.twinx()
        ax2b.plot(dates, position_counts, color="#e67e22", linewidth=1.5,
                  marker=".", markersize=6, drawstyle="steps-post", label="持仓数")
        ax2b.set_ylabel("持仓数 (只)", fontsize=11, color="#e67e22")
        ax2b.set_ylim(bottom=0)
        if max(position_counts) > 0:
            ax2b.set_ylim(top=max(position_counts) * 1.2)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                   fontsize=9, framealpha=0.9)

        ax2.set_xlabel("日期", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.25)

        fig.tight_layout()
        fig.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Equity+position chart saved to %s", chart_path)
        return chart_path

    # ── Finalize ────────────────────────────────────────────────────────────

    def _finalize(self) -> dict:
        """Compute final metrics and save outputs."""
        equity_series = self.accountant.to_equity_series()
        return_series = self.accountant.to_return_series()

        if len(return_series) < 2:
            logger.warning("Insufficient data for metrics")
            return {"error": "insufficient data"}

        # Compute metrics
        ann_return = float(return_series.mean() * 252)
        ann_vol = float(return_series.std() * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        cum_return = float((1 + return_series).prod() - 1)

        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = float(drawdown.min())

        # Win rate
        win_rate = float((return_series > 0).mean())

        # Trade stats
        total_trades = sum(len(e.get("fills", [])) for e in self._journal)
        buy_trades = sum(
            sum(1 for f in e.get("fills", []) if f["side"] == "buy")
            for e in self._journal
        )
        sell_trades = total_trades - buy_trades

        metrics = {
            "mode": self.mode,
            "simulation_period": f"{self.start} → {self.end}",
            "trading_days": len(self._trading_days),
            "initial_cash": self.initial_cash,
            "final_equity": round(self._equity, 2),
            "cumulative_return": f"{cum_return * 100:.2f}%",
            "annualized_return": f"{ann_return * 100:.2f}%",
            "annualized_volatility": f"{ann_vol * 100:.2f}%",
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": f"{max_dd * 100:.2f}%",
            "win_rate": f"{win_rate * 100:.1f}%",
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "llm_enabled": self.mode == "llm" and (self.llm and self.llm.configured),
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M")

        # Create unified output folder for this run
        sim_dir = self.output_dir / f"sim_{ts}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        charts_subdir = sim_dir / "charts"
        charts_subdir.mkdir(parents=True, exist_ok=True)

        # Move GP report into sim folder if it exists
        gp_report_path = self.output_dir / "gp_report" / "gp_evolution.png"
        if gp_report_path.exists():
            shutil.copy2(gp_report_path, sim_dir / "gp_evolution.png")

        # Save journal into sim folder
        journal_path = sim_dir / "journal.json"
        with open(journal_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "journal": self._journal,
                "decisions": self._decisions,
            }, f, ensure_ascii=False, indent=2)

        # Save markdown report
        md_path = sim_dir / "report.md"
        self._write_markdown_report(metrics, return_series, equity_series, md_path)

        # Generate K-line trade charts into sim/charts/
        chart_dir = self._generate_trade_charts(charts_subdir)
        metrics["chart_dir"] = str(chart_dir)

        # Generate equity + position chart into sim/
        eq_chart_path = self._generate_equity_position_chart(equity_series, sim_dir)
        metrics["equity_chart"] = str(eq_chart_path)

        logger.info("All outputs consolidated in %s", sim_dir)
        metrics["output_dir"] = str(sim_dir)
        return metrics

    def _write_markdown_report(
        self,
        metrics: dict,
        returns: pd.Series,
        equity: pd.Series,
        path: Path,
    ) -> None:
        """Generate a markdown trading journal report."""
        lines = []
        lines.append("# Agent Trading Simulation Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"\n## Summary\n")
        for k, v in metrics.items():
            lines.append(f"| {k} | {v} |")

        lines.append(f"\n## Equity Curve\n")
        lines.append(f"| Date | Equity | Daily Return |")
        lines.append(f"|------|--------|-------------|")
        for d, eq in equity.items():
            ret = returns.get(d, 0.0)
            lines.append(f"| {d.date()} | ¥{eq:,.0f} | {ret * 100:+.3f}% |")

        lines.append(f"\n## Trading Journal\n")
        for entry in self._journal:
            lines.append(f"\n### {entry['date']}")
            lines.append(f"  Equity: ¥{entry['equity']:,.0f} | Cash: ¥{entry['cash']:,.0f}")
            if entry.get("reasoning"):
                lines.append(f"  Strategy: {entry['reasoning']}")
            fills = entry.get("fills", [])
            if fills:
                lines.append(f"\n  | Symbol | Side | Shares | Price | Commission |")
                lines.append(f"  |--------|------|--------|-------|------------|")
                for f in fills:
                    lines.append(f"  | {f['symbol']} | {f['side']} | {f['shares']} | {f['price']} | {f['commission']} |")
            else:
                lines.append(f"  (no trades)")
            positions = entry.get("positions", [])
            if positions:
                lines.append(f"\n  Positions: {len(positions)} holdings")
                for p in positions:
                    lines.append(f"  - {p['symbol']}: {p['shares']}股 @ ¥{p['avg_cost']}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Report saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Trading Simulation")
    parser.add_argument("--start", type=str, default=TRADING_PERIOD[0].isoformat(),
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=TRADING_PERIOD[1].isoformat(),
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=INITIAL_CASH,
                        help="Initial cash")
    parser.add_argument("--mode", type=str, default="llm",
                        choices=["llm", "factor", "compare"],
                        help="Decision mode: llm (AI agent), factor (signal-driven), compare (both)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model to use")
    parser.add_argument("--use-gp", action="store_true",
                        help="Run GP factor discovery and use all validated factors")
    parser.add_argument("--gp-population", type=int, default=200,
                        help="GP population size (default: 200)")
    parser.add_argument("--gp-generations", type=int, default=25,
                        help="GP max generations (default: 25)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    gp_kwargs = dict(
        use_gp=args.use_gp,
        gp_population=args.gp_population,
        gp_generations=args.gp_generations,
    )

    if args.mode == "compare":
        return _run_comparison(start, end, args.cash, args.model, gp_kwargs)

    sim = AgentSimulation(
        start=start, end=end, initial_cash=args.cash,
        mode=args.mode, model=args.model, **gp_kwargs,
    )

    mode_labels = {"llm": "LLM Agent", "factor": "Factor-Driven", "heuristic": "Heuristic"}
    print("=" * 60)
    print(f"  Agent Trading Simulation ({mode_labels.get(args.mode, args.mode)})")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Cash: ¥{args.cash:,.0f}")
    print("=" * 60)

    t0 = time.time()
    metrics = sim.run()

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n  Elapsed: {time.time() - t0:.0f}s")
    print(f"  Output: {JOURNAL_DIR}")


def _run_comparison(start: date, end: date, cash: float, model: str | None, gp_kwargs: dict | None = None) -> None:
    """Run LLM agent and factor strategy side-by-side and compare."""
    gp_kwargs = gp_kwargs or {}

    print("=" * 60)
    print("  Comparison Mode: LLM Agent vs Factor-Driven")
    if gp_kwargs.get("use_gp"):
        print("  GP Factor Discovery: ENABLED")
    print(f"  Period: {start} → {end}")
    print(f"  Cash: ¥{cash:,.0f}")
    print("=" * 60)

    results = {}
    for mode, label in [("factor", "Factor-Driven"), ("llm", "LLM Agent")]:
        print(f"\n{'─' * 40}")
        print(f"  Running: {label}")
        print(f"{'─' * 40}")
        sim = AgentSimulation(
            start=start, end=end, initial_cash=cash,
            mode=mode, model=model, **gp_kwargs,
        )
        t0 = time.time()
        metrics = sim.run()
        elapsed = time.time() - t0
        results[mode] = {
            "label": label,
            "metrics": metrics,
            "elapsed": elapsed,
            "equity_series": sim.accountant.to_equity_series(),
            "return_series": sim.accountant.to_return_series(),
        }

    # Print comparison table
    print("\n" + "=" * 70)
    print("  Comparison Results")
    print("=" * 70)

    key_metrics = [
        ("final_equity", "Final Equity", "¥{:,.0f}"),
        ("cumulative_return", "Cumulative Return", "{}"),
        ("annualized_return", "Annualized Return", "{}"),
        ("annualized_volatility", "Annualized Vol", "{}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
        ("max_drawdown", "Max Drawdown", "{}"),
        ("win_rate", "Win Rate", "{}"),
        ("total_trades", "Total Trades", "{}"),
    ]

    rows = []
    for key, display, _ in key_metrics:
        f_val = results["factor"]["metrics"].get(key, "N/A")
        l_val = results["llm"]["metrics"].get(key, "N/A")

        # Determine winner
        if key in ("final_equity", "cumulative_return", "annualized_return",
                    "sharpe_ratio", "win_rate"):
            # Higher is better
            try:
                f_num = float(str(f_val).rstrip("%"))
                l_num = float(str(l_val).rstrip("%"))
            except (ValueError, AttributeError):
                f_num, l_num = 0, 0
            f_better = f_num > l_num
        elif key in ("annualized_volatility", "max_drawdown", "total_trades"):
            try:
                f_num = float(str(f_val).rstrip("%"))
                l_num = float(str(l_val).rstrip("%"))
            except (ValueError, AttributeError):
                f_num, l_num = 0, 0
            f_better = f_num < l_num
        else:
            f_better = False

        f_mark = " *" if f_better else ""
        l_mark = " *" if not f_better else ""

        rows.append(f"| {display:25s} | {str(f_val):>16s}{f_mark:2s} | {str(l_val):>16s}{l_mark:2s} |")

    print(f"| {'Metric':25s} | {'Factor-Driven':>18s} | {'LLM Agent':>18s} |")
    print(f"|{'-' * 27}|{'-' * 20}|{'-' * 20}|")
    for r in rows:
        print(r)

    print(f"\n  (* = better in category)")
    print(f"  Factor-Driven elapsed: {results['factor']['elapsed']:.0f}s")
    print(f"  LLM Agent elapsed: {results['llm']['elapsed']:.0f}s")

    # Save comparison
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    comp_path = JOURNAL_DIR / f"comparison_{ts}.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump({
            "factor": results["factor"]["metrics"],
            "llm": results["llm"]["metrics"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Comparison saved to: {comp_path}")

    # Save comparison markdown
    md_path = JOURNAL_DIR / f"comparison_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# LLM Agent vs Factor-Driven Comparison\n\n")
        f.write(f"Period: {start} → {end}  \n")
        f.write(f"Initial Cash: ¥{cash:,.0f}  \n\n")
        f.write(f"| {'Metric':25s} | {'Factor-Driven':>18s} | {'LLM Agent':>18s} |\n")
        f.write(f"|{'-' * 27}|{'-' * 20}|{'-' * 20}|\n")
        for r in rows:
            f.write(r + "\n")
        f.write(f"\n\n* = better in category\n")
        # Equity curves
        f.write("\n## Equity Curves\n\n")
        f.write("| Date | Factor Equity | LLM Equity | Factor Return | LLM Return |\n")
        f.write("|------|--------------|-----------|--------------|-----------|\n")
        f_eq = results["factor"]["equity_series"]
        l_eq = results["llm"]["equity_series"]
        f_ret = results["factor"]["return_series"]
        l_ret = results["llm"]["return_series"]
        for d in f_eq.index.union(l_eq.index):
            fe = f_eq.get(d, None)
            le = l_eq.get(d, None)
            fr = f_ret.get(d, None)
            lr = l_ret.get(d, None)
            fe_str = f"¥{fe:,.0f}" if fe is not None else "-"
            le_str = f"¥{le:,.0f}" if le is not None else "-"
            fr_str = f"{fr * 100:+.3f}%" if fr is not None else "-"
            lr_str = f"{lr * 100:+.3f}%" if lr is not None else "-"
            f.write(f"| {d.date()} | {fe_str} | {le_str} | {fr_str} | {lr_str} |\n")

    print(f"  Comparison markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
