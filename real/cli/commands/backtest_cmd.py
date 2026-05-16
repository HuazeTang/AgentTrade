"""CLI commands for running backtests."""

from __future__ import annotations

import json
from datetime import date

import click

from core.types import BacktestConfig
from strategy.cross_sectional import CrossSectionalStrategy
from backtest.engine import BacktestEngine
from backtest.recorder import save_backtest_result
from analysis.metrics import full_report


@click.group(name="backtest")
def backtest():
    """Run backtests."""


@backtest.command("run")
@click.option("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
@click.option("--signal", default="momentum_1m", help="Signal/factor column to use")
@click.option("--top-quantile", default=0.2, type=float, help="Fraction of universe to long")
@click.option("--initial-cash", default=1_000_000.0, type=float, help="Initial cash (CNY)")
@click.option("--long-only/--long-short", default=True, help="Long-only or long-short")
@click.option("--n-positions", default=50, type=int, help="Max number of positions")
@click.option("--run-name", default=None, help="Name for this backtest run")
def backtest_run(
    start: str,
    end: str,
    signal: str,
    top_quantile: float,
    initial_cash: float,
    long_only: bool,
    n_positions: int,
    run_name: str | None,
):
    """Run a cross-sectional backtest."""
    config = BacktestConfig(
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end),
        initial_cash=initial_cash,
    )

    strategy = CrossSectionalStrategy(
        signal_col=signal,
        top_quantile=top_quantile,
        long_only=long_only,
        n_positions=n_positions,
    )

    click.echo(f"Running backtest: {start} → {end}")
    click.echo(f"Strategy: {strategy.name} (signal={signal}, long_only={long_only})")

    engine = BacktestEngine(config=config, strategy=strategy)
    result = engine.run()

    # Report
    report = full_report(result.daily_returns, result.equity_curve, result.benchmark_returns)
    click.echo("\n=== Backtest Results ===")
    for k, v in report.items():
        if isinstance(v, float):
            click.echo(f"  {k}: {v:.4f}")
        else:
            click.echo(f"  {k}: {v}")

    # Save
    name = run_name or f"backtest_{start}_{end}"
    path = save_backtest_result(result, name)
    click.echo(f"\nResults saved to: {path}")
