"""Parallel backtest runner for GP factor discovery.

Dispatches independent backtest tasks across threads. Each task is read-only
on market data and writes to an isolated PortfolioAccountant, so no locks needed.

Uses ThreadPoolExecutor (not ProcessPoolExecutor) because compiled Factor classes
created via exec() are not pickleable. Thread-level parallelism is sufficient since
pandas/numpy operations release the GIL during backtest loops.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestTask:
    """Immutable specification for one backtest run."""
    label: str
    baseline_names: list[str]
    gp_names: list[str]
    start: date
    end: date
    initial_cash: float
    symbols: list[str]
    strategy_params: dict | None = None  # e.g. {"sell_rank_limit": 7}

    # Set by caller after construction (DataFrames can't be in __init__ for
    # dataclass defaults, but we set them immediately after)
    daily_cache: pd.DataFrame | None = field(default=None, repr=False)
    trading_days: pd.DatetimeIndex | None = field(default=None, repr=False)

    # Pre-computed factor cache: when set, _worker_backtest subsets this
    # DataFrame instead of calling _compute_factor_set_staged from scratch.
    # Must contain baseline columns + all GP columns referenced by gp_names.
    factor_cache: pd.DataFrame | None = field(default=None, repr=False)


def run_parallel_backtests(
    tasks: list[BacktestTask],
    max_workers: int = 4,
) -> dict[str, dict]:
    """Run multiple backtest tasks in parallel via ThreadPoolExecutor.

    Each task is independent: same market data (read-only), isolated
    PortfolioAccountant. Results are deterministic and identical to sequential
    execution.

    Args:
        tasks: List of BacktestTask specifications.
        max_workers: Maximum thread count. Defaults to min(4, cpu_count).

    Returns:
        {task.label: metrics_dict} for each completed task.
    """
    if not tasks:
        return {}

    import os
    max_workers = min(max_workers, os.cpu_count() or 2)

    results: dict[str, dict] = {}
    logger.info("Dispatching %d backtest tasks across %d workers",
                 len(tasks), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_worker_backtest, task): task.label
            for task in tasks
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                results[label] = future.result()
            except Exception as e:
                logger.error("Backtest task [%s] failed: %s", label, e, exc_info=True)
                results[label] = {"error": str(e)}

    return results


def _worker_backtest(task: BacktestTask) -> dict:
    """Run a single backtest task in a worker thread.

    Constructs a minimal AgentSimulation, computes factors, runs the backtest
    loop, and returns performance metrics.

    This must be a top-level function (not a method) for ThreadPoolExecutor compat.
    """
    from run_agent_simulation import AgentSimulation, _calibrate_factor_weights_standalone
    from data.calendar import get_trading_days
    from data.cache import read_daily
    import pandas as pd

    # Load market data (needed since daily_cache can't be passed between threads
    # cleanly due to pandas shared internals)
    if task.daily_cache is not None:
        data = task.daily_cache.copy()
    else:
        data = read_daily(task.start, task.end)

    trading_days = task.trading_days
    if trading_days is None:
        trading_days = get_trading_days(task.start, task.end)

    # Use pre-computed factor cache if available (avoids recomputing baseline +
    # GP factors from scratch for each cumulative backtest)
    if task.factor_cache is not None:
        # Subset: baseline columns + requested GP columns
        all_needed = [c for c in task.baseline_names if c in task.factor_cache.columns]
        all_needed += [c for c in task.gp_names if c in task.factor_cache.columns]
        factor_df = task.factor_cache[all_needed].copy()

        # Shift: D's factor uses D-1 close
        shifted = factor_df.unstack().shift(1).stack()
        factor_df = shifted.reorder_levels(["trade_date", "symbol"]).sort_index()

        from run_agent_simulation import TRAIN_PERIOD
        train_dates = get_trading_days(TRAIN_PERIOD[0], TRAIN_PERIOD[1])
        factor_weights = _calibrate_factor_weights_standalone(
            factor_df, train_dates, data,
        )

        # Create minimal simulation instance for backtest loop
        sim = AgentSimulation(
            mode="factor",
            start=task.start,
            end=task.end,
            initial_cash=task.initial_cash,
        )
        sim._trading_days = trading_days
        sim._daily_cache = data
        sim._trading_day_index = 0
        sim._strategy_params = task.strategy_params or {}
        sim._market_dd = pd.Series(dtype=float)  # minimal stub
    else:
        # Fallback: compute factors from scratch
        sim = AgentSimulation(
            mode="factor",
            start=task.start,
            end=task.end,
            initial_cash=task.initial_cash,
        )
        sim._trading_days = trading_days
        sim._daily_cache = data
        sim._trading_day_index = 0
        sim._strategy_params = task.strategy_params or {}

        factor_df, factor_weights = sim._compute_factor_set_staged(
            task.baseline_names, task.gp_names,
        )

    # Set up symbols
    symbols = task.symbols
    if not symbols:
        symbols = sorted(data.index.get_level_values("symbol").unique().tolist())

    # Run backtest loop
    equity = sim._run_backtest_loop(
        factor_df, factor_weights, symbols, task.label,
    )

    # Compute metrics
    return _metrics_from_equity(equity, task.initial_cash)


def _metrics_from_equity(equity: pd.Series, initial_cash: float) -> dict:
    """Compute performance metrics from equity series. Pure function."""
    if equity.empty:
        return _empty_metrics()

    eq = equity.dropna()
    if len(eq) < 10:
        return _empty_metrics()

    final_eq = eq.iloc[-1]
    cr = (final_eq - initial_cash) / initial_cash
    daily_rets = eq.pct_change().dropna()

    if len(daily_rets) < 5:
        sr = 0.0
    else:
        std = daily_rets.std()
        sr = float(daily_rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())

    days = len(eq)
    ar = float((1 + cr) ** (252 / days) - 1) if days > 0 else 0.0

    wr = float((daily_rets > 0).mean()) if len(daily_rets) > 0 else 0.0

    return {
        "cumulative_return": round(cr, 6),
        "sharpe_ratio": round(sr, 3),
        "max_drawdown": round(mdd, 6),
        "annualized_return": round(ar, 6),
        "win_rate": round(wr, 6),
    }


def _empty_metrics() -> dict:
    return {
        "cumulative_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "annualized_return": 0.0,
        "win_rate": 0.0,
    }
