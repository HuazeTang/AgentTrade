"""CLI commands for model training."""

from __future__ import annotations

from datetime import date

import click
import pandas as pd

from data.cache import read_daily
from factor.storage import read_factor
from model.models.gbdt import GBDTModel
from model.serialization import save_model
from model.trainer import train_model


@click.group(name="train")
def train():
    """Train alpha prediction models."""


@train.command("run")
@click.option("--factors", required=True, help="Comma-separated factor names to use as features")
@click.option("--start", required=True, help="Train start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="Train end date (YYYY-MM-DD)")
@click.option("--horizon", default=5, help="Forward return horizon (trading days)")
@click.option("--model-name", default=None, help="Name for saved model")
@click.option("--n-estimators", default=200, type=int, help="Number of trees")
@click.option("--max-depth", default=5, type=int, help="Max tree depth")
@click.option("--learning-rate", default=0.05, type=float, help="Learning rate")
def train_run(
    factors: str,
    start: str,
    end: str,
    horizon: int,
    model_name: str | None,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
):
    """Train a GBDT alpha model."""
    factor_names = [f.strip() for f in factors.split(",")]
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    # Load factor data
    click.echo("Loading factors...")
    factor_dfs = []
    for name in factor_names:
        try:
            fdf = read_factor(name, start=start_d, end=end_d)
            factor_dfs.append(fdf)
            click.echo(f"  {name}: {len(fdf)} values")
        except FileNotFoundError:
            click.echo(f"  {name}: not found, skipping")
            continue

    if not factor_dfs:
        click.echo("No factor data found. Run 'realtrade factor compute' first.")
        raise SystemExit(1)

    factor_df = pd.concat(factor_dfs, axis=1)

    # Load prices for target
    click.echo("Loading prices...")
    prices_df = read_daily(start_d, end_d)
    if prices_df.empty:
        click.echo("No price data. Run 'realtrade data ingest' first.")
        raise SystemExit(1)
    prices = prices_df["close"]

    # Train
    click.echo(f"Training GBDT model (horizon={horizon}d)...")
    model = GBDTModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    trained, X, y = train_model(
        model, factor_df, prices, horizon=horizon,
        train_start=start, train_end=end,
    )

    # Save
    name = model_name or f"gbdt_{start}_{end}_h{horizon}"
    path = save_model(trained, name, metadata={
        "factors": factor_names,
        "horizon": horizon,
        "train_start": start,
        "train_end": end,
        "n_features": X.shape[1],
        "n_samples": len(y),
    })
    click.echo(f"Model saved: {path}")

    # Show importance
    imp = trained.feature_importance
    if not imp.empty:
        click.echo("\nTop 10 feature importance:")
        for feat, score in imp.head(10).items():
            click.echo(f"  {feat}: {score:.4f}")
