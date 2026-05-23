"""CLI for parameter search."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import click

from param_search.config import (
    ALL_TUNABLE_PARAMS,
    DEFAULT_PARAMS,
    ParameterSpec,
    SearchConfig,
)
from param_search.engine import ResultStore, SearchEngine


@click.group(name="search")
def search_group():
    """Parameter search for strategy optimization."""


@search_group.command("run")
@click.option("--strategy", type=click.Choice(["random", "grid", "sequential"]),
              default="random", help="Search strategy")
@click.option("--iterations", "-n", type=int, default=500,
              help="Number of combinations to evaluate")
@click.option("--workers", "-w", type=int, default=4,
              help="Parallel workers")
@click.option("--resume/--no-resume", default=True,
              help="Resume from existing results file")
@click.option("--output", "-o", type=Path, default=Path("data/results/param_search.json"),
              help="Results output path")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--metric", default="sharpe_ratio",
              help="Metric to optimize (must be a key in the raw metrics dict)")
@click.option("--start", "trading_start", type=str, default=None,
              help="Override trading start date (YYYY-MM-DD)")
@click.option("--end", "trading_end", type=str, default=None,
              help="Override trading end date (YYYY-MM-DD)")
@click.option("--cash", type=float, default=None, help="Override initial cash")
@click.option("--params", "-p", multiple=True,
              help="Limit search to specific params (e.g. -p MAX_POSITIONS -p SELL_RANK_LIMIT)")
def search_run(
    strategy: str,
    iterations: int,
    workers: int,
    resume: bool,
    output: Path,
    seed: int,
    metric: str,
    trading_start: str | None,
    trading_end: str | None,
    cash: float | None,
    params: tuple[str, ...],
) -> None:
    """Run parameter search."""
    # Select parameters
    if params:
        selected = [p for p in ALL_TUNABLE_PARAMS if p.name in set(params)]
        if not selected:
            click.secho(f"No matching params for: {params}", fg="red")
            raise SystemExit(1)
        click.echo(f"Limited to {len(selected)} params: "
                   f"{', '.join(p.name for p in selected)}")
    else:
        selected = list(ALL_TUNABLE_PARAMS)

    ts = date.fromisoformat(trading_start) if trading_start else None
    te = date.fromisoformat(trading_end) if trading_end else None

    config = SearchConfig(
        parameters=selected,
        strategy=strategy,
        n_iterations=iterations,
        n_workers=workers,
        random_seed=seed,
        metric=metric,
        output_path=output,
        trading_start=ts,
        trading_end=te,
        initial_cash=cash,
        resume=resume,
    )

    engine = SearchEngine(config)
    results = engine.run()

    # Show top 5
    top5 = results[:5] if len(results) >= 5 else results
    click.echo(f"\n  Top {len(top5)}:")
    for i, r in enumerate(top5):
        m = r["metrics"]
        click.echo(
            f"  #{i+1}  {metric}={m.get(metric, '?'):.4f}  "
            f"Sharpe={m.get('sharpe_ratio', '?'):.3f}  "
            f"Return={m.get('cumulative_return', 0)*100:.1f}%  "
            f"DD={m.get('max_drawdown', 0)*100:.1f}%  "
            f"Trades={m.get('total_trades', '?')}"
        )
        if click.get_current_context().params.get("verbose", False):
            click.echo(f"       {r['params']}")


@search_group.command("results")
@click.option("--input", "-i", "input_path", type=Path, required=True,
              help="Results JSON file")
@click.option("--top", type=int, default=10, help="Show top N")
@click.option("--metric", default="sharpe_ratio", help="Metric to sort by")
def search_results(input_path: Path, top: int, metric: str) -> None:
    """Display top results from a search."""
    store = ResultStore(input_path)
    n = store.load()
    if n == 0:
        click.secho("No results found.", fg="yellow")
        return

    click.echo(f"{n} results loaded from {input_path}\n")

    items = store.top(metric=metric, n=top)
    click.echo(f"  {'#':<3} {'Sharpe':>7} {'Return':>9} {'Max DD':>8} "
               f"{'WinRate':>8} {'Trades':>7}  {'Params'}")
    click.echo(f"  {'─'*75}")
    for i, r in enumerate(items):
        m = r["metrics"]
        ret = m.get("cumulative_return", 0)
        dd = m.get("max_drawdown", 0)
        sharpe = m.get("sharpe_ratio", 0)
        wr = m.get("win_rate", 0)
        trades = m.get("total_trades", "?")
        p = r.get("params", {})
        # Short param summary
        param_str = ", ".join(f"{k}={v}" for k, v in sorted(p.items())
                              if v != DEFAULT_PARAMS.get(k))
        if not param_str:
            param_str = "(baseline)"
        click.echo(
            f"  {i+1:<3} {sharpe:>7.3f} {ret*100:>8.1f}% "
            f"{dd*100:>7.1f}% {wr*100:>7.1f}% {trades:>7}  {param_str}"
        )


@search_group.command("plot")
@click.option("--input", "-i", "input_path", type=Path, required=True,
              help="Results JSON file")
@click.option("--output", "-o", "output_path", type=Path, default=None,
              help="Output chart path (default: same dir, param_search.png)")
@click.option("--kind", type=click.Choice(["pareto", "parallel", "sensitivity"]),
              default="pareto", help="Chart type")
def search_plot(input_path: Path, output_path: Path | None, kind: str) -> None:
    """Visualize parameter search results."""
    from param_search.viz import plot_pareto, plot_sensitivity, plot_parallel

    store = ResultStore(input_path)
    n = store.load()
    if n == 0:
        click.secho("No results found.", fg="yellow")
        return

    if output_path is None:
        output_path = input_path.with_suffix(".png")

    items = store.top(n=None)
    click.echo(f"Plotting {len(items)} results → {output_path}")

    if kind == "pareto":
        plot_pareto(items, output_path)
    elif kind == "sensitivity":
        plot_sensitivity(items, output_path)
    elif kind == "parallel":
        plot_parallel(items, output_path)


if __name__ == "__main__":
    search_group()
