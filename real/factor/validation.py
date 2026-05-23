"""Factor validation: IC analysis, quantile returns, correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    method: str = "pearson",
) -> pd.Series:
    """Compute cross-sectional IC per date.

    Args:
        factor: Multi-indexed (trade_date, symbol) Series of factor values.
        forward_returns: Multi-indexed (trade_date, symbol) Series of forward returns.
        method: "pearson" (default) or "spearman" (rank IC).

    Returns:
        Series indexed by trade_date with IC values.
    """
    df = pd.DataFrame({"factor": factor, "fwd_ret": forward_returns}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    if method == "spearman":
        return df.groupby("trade_date").apply(
            lambda g: g["factor"].corr(g["fwd_ret"], method="spearman")
        ).dropna()
    else:
        return df.groupby("trade_date").apply(
            lambda g: g["factor"].corr(g["fwd_ret"], method="pearson")
        ).dropna()


def compute_rank_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
) -> pd.Series:
    """Convenience: cross-sectional rank IC (Spearman)."""
    return compute_ic(factor, forward_returns, method="spearman")


def ic_summary(ic_series: pd.Series) -> dict:
    """Summarize IC statistics."""
    if ic_series.empty:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "hit_rate": np.nan}
    return {
        "mean": float(ic_series.mean()),
        "std": float(ic_series.std()),
        "ir": float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0,
        "hit_rate": float((ic_series > 0).mean()),
    }


def quantile_returns(
    factor: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute forward returns by factor quantile per date.

    Returns DataFrame with columns: date, quantile, return (mean forward return).
    """
    df = pd.DataFrame({"factor": factor, "fwd_ret": forward_returns}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "quantile", "return"])

    df["quantile"] = df.groupby("trade_date")["factor"].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
    )

    result = df.groupby(["trade_date", "quantile"])["fwd_ret"].mean().reset_index()
    return result


def cross_sectional_range_return(
    forward_returns: pd.Series,
    n_quantiles: int = 10,
) -> pd.Series:
    """Per-date top-minus-bottom decile return spread, sorted by actual outcome.

    Ranks stocks by their *ex-post* forward return each date (not by a factor).
    This gives the maximum achievable winner-loser spread — the benchmark against
    which a factor's top-decile spread is compared.

    Args:
        forward_returns: MultiIndex (trade_date, symbol) Series.
        n_quantiles: Number of buckets (default 10 for deciles).

    Returns:
        Series indexed by trade_date with the cross-sectional spread.
    """
    fwd = forward_returns.dropna()
    if fwd.empty:
        return pd.Series(dtype=float)

    ret_name = fwd.name if fwd.name else "fwd_ret"
    df = fwd.reset_index()

    def _range_per_date(g):
        vals = g[ret_name].dropna()
        if len(vals) < n_quantiles:
            return np.nan
        try:
            labels = pd.qcut(vals, n_quantiles, labels=False, duplicates="drop")
            top_mask = labels == labels.max()
            bot_mask = labels == labels.min()
            return float(vals[top_mask].mean() - vals[bot_mask].mean())
        except (ValueError, IndexError):
            return np.nan

    result = df.groupby("trade_date").apply(_range_per_date).dropna()
    result.name = "cs_range_return"
    return result


def quantile_cumulative_returns(
    factor: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute cumulative returns per quantile over time.

    Returns DataFrame with columns: trade_date, quantile, cum_return.
    """
    qr = quantile_returns(factor, forward_returns, n_quantiles)
    if qr.empty:
        return pd.DataFrame(columns=["trade_date", "quantile", "cum_return"])

    qr["cum_return"] = qr.groupby("quantile")["return"].transform(
        lambda x: (1 + x).cumprod()
    )
    return qr


def factor_correlation(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise factor correlation matrix using pooled cross-section."""
    return factor_df.corr()


def factor_auto_correlation(factor: pd.Series) -> float:
    """Compute average rank autocorrelation of a factor (1-day lag)."""
    df = factor.unstack()  # date x symbol
    ranks = df.rank(axis=1)
    corrs = []
    for i in range(1, len(ranks)):
        corrs.append(ranks.iloc[i].corr(ranks.iloc[i - 1], method="spearman"))
    return float(np.mean(corrs)) if corrs else np.nan
