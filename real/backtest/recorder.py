"""Record backtest results to disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import RESULT_DIR
from core.types import BacktestResult, Fill


def save_backtest_result(
    result: BacktestResult,
    run_name: str,
) -> Path:
    """Save all backtest result artifacts to a run directory.

    Args:
        result: Completed backtest result.
        run_name: Unique name for this run.

    Returns:
        Path to the run output directory.
    """
    out_dir = RESULT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_frame("equity").to_parquet(out_dir / "equity_curve.parquet")
    result.daily_returns.to_frame("return").to_parquet(out_dir / "daily_returns.parquet")
    result.benchmark_returns.to_frame("benchmark_return").to_parquet(
        out_dir / "benchmark_returns.parquet"
    )
    result.turnover.to_frame("turnover").to_parquet(out_dir / "turnover.parquet")

    if not result.positions.empty:
        result.positions.to_parquet(out_dir / "positions.parquet")

    if not result.fills.empty:
        result.fills.to_parquet(out_dir / "fills.parquet")

    if result.factor_exposures is not None and not result.factor_exposures.empty:
        result.factor_exposures.to_parquet(out_dir / "factor_exposures.parquet")

    # Summary JSON
    import json
    summary = {
        "cumulative_return": result.cumulative_return,
        "annualized_return": result.annualized_return,
        "annualized_volatility": result.annualized_volatility,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "calmar_ratio": result.calmar_ratio,
        "excess_return": result.excess_return,
        "information_ratio": result.information_ratio,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return out_dir
