"""CLI commands for analyzing backtest results."""

from __future__ import annotations

import json

import click
import pandas as pd

from config.settings import RESULT_DIR
from analysis.metrics import full_report
from analysis.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns_heatmap,
    plot_turnover,
)


@click.group(name="analyze")
def analyze():
    """Analyze backtest results."""


@analyze.command("metrics")
@click.option("--run-name", required=True, help="Backtest run name")
def metrics_cmd(run_name: str):
    """Print performance metrics for a backtest run."""
    run_dir = RESULT_DIR / run_name
    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}")
        raise SystemExit(1)

    returns = pd.read_parquet(run_dir / "daily_returns.parquet")["return"]
    equity = pd.read_parquet(run_dir / "equity_curve.parquet")["equity"]
    bm_path = run_dir / "benchmark_returns.parquet"
    bm_returns = None
    if bm_path.exists():
        bm_returns = pd.read_parquet(bm_path)["benchmark_return"]

    report = full_report(returns, equity, bm_returns)
    click.echo(json.dumps(report, indent=2, default=str))


@analyze.command("plot")
@click.option("--run-name", required=True, help="Backtest run name")
@click.option("--output", default=None, help="Output file path for the chart (png)")
@click.option("--type", "plot_type", default="equity",
              type=click.Choice(["equity", "drawdown", "monthly", "turnover"]))
def plot_cmd(run_name: str, output: str | None, plot_type: str):
    """Generate plots from backtest results."""
    import matplotlib
    matplotlib.use("Agg")

    run_dir = RESULT_DIR / run_name
    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}")
        raise SystemExit(1)

    returns = pd.read_parquet(run_dir / "daily_returns.parquet")["return"]
    equity = pd.read_parquet(run_dir / "equity_curve.parquet")["equity"]
    turnover = pd.read_parquet(run_dir / "turnover.parquet")["turnover"]

    bm_returns = None
    bm_path = run_dir / "benchmark_returns.parquet"
    if bm_path.exists():
        bm_returns = pd.read_parquet(bm_path)["benchmark_return"]

    if plot_type == "equity":
        fig = plot_equity_curve(equity, bm_returns)
    elif plot_type == "drawdown":
        fig = plot_drawdown(equity)
    elif plot_type == "monthly":
        fig = plot_monthly_returns_heatmap(returns)
    elif plot_type == "turnover":
        fig = plot_turnover(turnover)
    else:
        click.echo(f"Unknown plot type: {plot_type}")
        return

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        click.echo(f"Saved to {output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
