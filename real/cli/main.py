"""CLI entry point for RealTrade."""

from __future__ import annotations

import click

from cli.commands.data_cmd import data
from cli.commands.factor_cmd import factor
from cli.commands.train_cmd import train
from cli.commands.backtest_cmd import backtest
from cli.commands.analyze_cmd import analyze


@click.group()
def cli():
    """RealTrade -- A-share quantitative research and backtesting platform."""


cli.add_command(data)
cli.add_command(factor)
cli.add_command(train)
cli.add_command(backtest)
cli.add_command(analyze)


if __name__ == "__main__":
    cli()
