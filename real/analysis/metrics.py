"""Performance metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    annualization: float = 252,
    risk_free: float = 0.0,
) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / annualization
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(annualization))


def sortino_ratio(
    returns: pd.Series,
    annualization: float = 252,
    risk_free: float = 0.0,
) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    excess = returns - risk_free / annualization
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(annualization))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return (max_drawdown_fraction, peak_date, trough_date)."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    min_idx = drawdown.idxmin()
    mdd = float(drawdown[min_idx])
    peak_idx = peak[peak == peak.loc[min_idx]].index[0]
    return mdd, peak_idx, min_idx


def calmar_ratio(annualized_return: float, mdd: float) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""
    if mdd == 0:
        return 0.0
    return annualized_return / abs(mdd)


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, annualization: float = 252
) -> float:
    """Annualized information ratio."""
    excess = returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(annualization))


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return periods."""
    pos = (returns > 0).sum()
    total = len(returns)
    return float(pos / total) if total > 0 else 0.0


def profit_factor(returns: pd.Series) -> float:
    """Sum of positive returns / abs sum of negative returns."""
    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    return float(pos / neg) if neg > 0 else float("inf")


def annualized_return(returns: pd.Series, annualization: float = 252) -> float:
    """CAGR from daily returns."""
    total = (1 + returns).prod()
    n_years = len(returns) / annualization
    return float(total ** (1 / max(n_years, 0.01)) - 1)


def annualized_volatility(returns: pd.Series, annualization: float = 252) -> float:
    """Annualized volatility."""
    return float(returns.std() * np.sqrt(annualization))


def maximum_daily_return(returns: pd.Series) -> float:
    """Best single-day return."""
    return float(returns.max()) if not returns.empty else 0.0


def minimum_daily_return(returns: pd.Series) -> float:
    """Worst single-day return."""
    return float(returns.min()) if not returns.empty else 0.0


def full_report(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """Generate a comprehensive metrics report."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    mdd, peak_date, trough_date = max_drawdown(equity_curve)

    report = {
        "cumulative_return": float((1 + returns).prod() - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": mdd,
        "max_drawdown_peak": str(peak_date.date()),
        "max_drawdown_trough": str(trough_date.date()),
        "calmar_ratio": calmar_ratio(ann_ret, mdd),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "max_daily_return": maximum_daily_return(returns),
        "min_daily_return": minimum_daily_return(returns),
        "n_days": len(returns),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if not aligned.empty:
            report["information_ratio"] = information_ratio(
                aligned.iloc[:, 0], aligned.iloc[:, 1]
            )
        report["benchmark_cumulative_return"] = float(
            (1 + benchmark_returns).prod() - 1
        )

    return report
