"""CLI commands for data management."""

from __future__ import annotations

from datetime import date

import click

from data.cache import data_summary
from data.pipeline import ingest_daily, ensure_data
from data.quality import generate_report
from data.sources.akshare import AkshareSource


@click.group(name="data")
def data():
    """Manage market data."""


@data.command("ingest")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default=None, help="Comma-separated symbols, or 'all'")
@click.option("--chunk-size", default=30, help="Symbols per fetch batch")
def ingest_cmd(start: str, end: str, symbols: str | None, chunk_size: int):
    """Download and cache daily market data."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    source = AkshareSource()

    if symbols is None or symbols == "all":
        stocks = source.list_stocks()
        sym_list = stocks["symbol"].tolist()
        click.echo(f"Fetching all {len(sym_list)} stocks...")
    else:
        sym_list = [s.strip() for s in symbols.split(",")]

    click.echo(f"Ingesting data from {start} to {end} for {len(sym_list)} symbols...")
    df = ingest_daily(sym_list, start_d, end_d, source=source, chunk_size=chunk_size)
    click.echo(f"Cached {len(df)} rows.")


@data.command("ensure")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default=None, help="Comma-separated symbols")
def ensure_cmd(start: str, end: str, symbols: str | None):
    """Ensure data is cached; fetch only missing."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    source = AkshareSource()
    sym_list = [s.strip() for s in symbols.split(",")] if symbols else []
    if not sym_list:
        stocks = source.list_stocks()
        sym_list = stocks["symbol"].tolist()

    df = ensure_data(sym_list, start_d, end_d, source=source)
    click.echo(f"Available rows: {len(df)}")


@data.command("describe")
def describe_cmd():
    """Show cached data summary."""
    summary = data_summary()
    click.echo(f"Date range: {summary['dates'][0]} ~ {summary['dates'][1]}")
    click.echo(f"Symbols: {summary['symbols']}")
    click.echo(f"Total rows: {summary['rows']}")


@data.command("validate")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
def validate_cmd(start: str, end: str):
    """Run data quality checks."""
    from data.cache import read_daily

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    df = read_daily(start_d, end_d)
    if df.empty:
        click.echo("No data found in cache for the given range.")
        return
    report = generate_report(df)
    click.echo(report)
