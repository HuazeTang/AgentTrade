"""CLI commands for factor management."""

from __future__ import annotations

from datetime import date

import click

from data.cache import read_daily
from factor.engine import FactorEngine
from factor.registry import registry
from factor.storage import write_factor

# Import factor implementations to trigger registration
import factor.factors  # noqa


@click.group(name="factor")
def factor():
    """Manage and compute factors."""


@factor.command("list")
def list_factors():
    """List available factors."""
    names = registry.list_all()
    if not names:
        click.echo("No factors registered.")
        return
    click.echo(f"{'Name':<30} {'Category':<15} {'Description'}")
    click.echo("-" * 70)
    for n in names:
        cls = registry.get(n)
        meta = cls.meta
        click.echo(f"{n:<30} {meta.category:<15} {meta.description}")


@factor.command("compute")
@click.option("--factors", "-f", required=True, help="Comma-separated factor names")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--save/--no-save", default=True, help="Save to parquet")
def compute_cmd(factors: str, start: str, end: str, save: bool):
    """Compute factors for a date range."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    factor_names = [f.strip() for f in factors.split(",")]

    # Verify all factors exist
    for name in factor_names:
        try:
            registry.get(name)
        except KeyError:
            click.echo(f"Error: Factor '{name}' not found. Run 'factor list' to see available.")
            raise SystemExit(1)

    # Load data
    click.echo("Loading market data...")
    df = read_daily(start_d, end_d)
    if df.empty:
        click.echo("No market data found. Run 'realtrade data ingest' first.")
        raise SystemExit(1)

    click.echo(f"Computing factors: {', '.join(factor_names)}...")
    engine = FactorEngine()
    result = engine.compute(factor_names, df)

    if save:
        for name in factor_names:
            if name in result.columns:
                ser = result[name].dropna()
                if not ser.empty:
                    path = write_factor(ser, name)
                    click.echo(f"  Saved {name}: {len(ser)} values → {path}")

    click.echo(f"\nFactor matrix: {result.shape[1]} factors, {result.shape[0]} rows")
    click.echo(result.describe().to_string())
