"""Performance attribution: factor IC, quantile P&L, sector breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd


def factor_icc(returns: pd.DataFrame, factor_exposures: pd.DataFrame) -> pd.Series:
    """Compute Information Coefficient per factor.

    Args:
        returns: DataFrame (date x symbol) of forward returns.
        factor_exposures: DataFrame (date x symbol) of factor values.

    Returns:
        Series mapping factor name to mean IC.
    """
    common_factors = factor_exposures.columns.intersection(returns.columns)
    ic_values: dict[str, list[float]] = {f: [] for f in common_factors}

    for d in returns.index.intersection(factor_exposures.index):
        for f in common_factors:
            ret_row = returns.loc[d].dropna()
            fac_row = factor_exposures.loc[d, f].dropna()
            common = ret_row.index.intersection(fac_row.index)
            if len(common) >= 10:
                ic = ret_row[common].corr(fac_row[common], method="spearman")
                if not np.isnan(ic):
                    ic_values[f].append(ic)

    return pd.Series({f: np.mean(v) for f, v in ic_values.items()}).sort_values(
        ascending=False
    )


def quantile_pnl_decomposition(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute cumulative P&L for each signal quantile.

    Returns DataFrame with columns: date, quantile, cum_return.
    """
    results: list[dict] = []

    for d in signals.index:
        if d not in returns.index:
            continue
        sig = signals.loc[d].dropna()
        ret = returns.loc[d].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < n_quantiles * 2:
            continue
        try:
            quantiles = pd.qcut(sig[common], n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        for q in range(n_quantiles):
            mask = quantiles == q
            if mask.any():
                avg_ret = ret[common][mask].mean()
                results.append({"trade_date": d, "quantile": q, "return": avg_ret})

    if not results:
        return pd.DataFrame(columns=["trade_date", "quantile", "return"])

    df = pd.DataFrame(results)
    df["cum_return"] = df.groupby("quantile")["return"].transform(
        lambda x: (1 + x).cumprod()
    )
    return df


def sector_attribution(
    returns: pd.Series,
    sectors: pd.Series,
    weights: pd.Series,
) -> pd.DataFrame:
    """Attribute returns to sectors.

    Args:
        returns: Series indexed by symbol with period returns.
        sectors: Series indexed by symbol with sector labels.
        weights: Series indexed by symbol with portfolio weights.

    Returns:
        DataFrame with columns: sector, weight, contribution.
    """
    df = pd.DataFrame({
        "return": returns,
        "sector": sectors,
        "weight": weights,
    }).dropna()

    if df.empty:
        return pd.DataFrame(columns=["sector", "weight", "contribution"])

    df["contribution"] = df["weight"] * df["return"]
    return df.groupby("sector")[["weight", "contribution"]].sum().reset_index()
