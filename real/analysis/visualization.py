"""Visualization for backtest results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot cumulative equity curve with optional benchmark overlay."""
    fig, ax = plt.subplots(figsize=figsize)

    equity_norm = equity / equity.iloc[0]
    ax.plot(equity_norm.index, equity_norm.values, label="Portfolio", linewidth=1.5)

    if benchmark is not None and not benchmark.empty:
        bm_norm = (1 + benchmark).cumprod()
        ax.plot(bm_norm.index, bm_norm.values, label="Benchmark",
                linewidth=1, alpha=0.7, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y:.2f}x"
    ))
    fig.tight_layout()
    return fig


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot drawdown over time."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(drawdown.index, drawdown.values, 0,
                     color="red", alpha=0.3, label="Drawdown")
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y:.0f}%"
    ))
    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot monthly returns heatmap."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, cbar_kws={"label": "Return (%)"},
        ax=ax, linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    fig.tight_layout()
    return fig


def plot_turnover(
    turnover: pd.Series,
    title: str = "Daily Turnover",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot daily turnover."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(turnover.index, turnover.values, alpha=0.5, label="Turnover")
    ax.plot(turnover.index, turnover.values, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover Ratio")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_quantile_returns(
    quantile_df: pd.DataFrame,
    title: str = "Cumulative Returns by Quantile",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot cumulative returns for each signal quantile."""
    fig, ax = plt.subplots(figsize=figsize)

    for q in sorted(quantile_df["quantile"].unique()):
        subset = quantile_df[quantile_df["quantile"] == q].sort_values("trade_date")
        ax.plot(subset["trade_date"], subset["cum_return"].values,
                label=f"Q{q + 1}", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
