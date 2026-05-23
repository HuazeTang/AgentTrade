"""Pure factor evaluation via factor mimicking portfolios.

Evaluates factors in a "sterile" cross-sectional environment:
no transaction costs, no T+1 delay, no position limits, no stop-loss.
This isolates factor predictive power from trading rule implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("pure_factor")


@dataclass
class PureFactorMetrics:
    """Output of pure factor evaluation."""

    # IC metrics
    ic_mean: float = np.nan
    ic_std: float = np.nan
    ic_ir: float = np.nan
    ic_hit_rate: float = np.nan

    # Factor mimicking portfolio metrics (annualized)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    cumulative_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    mean_daily_return: float = 0.0
    win_rate: float = 0.0

    # Walk-forward
    wf_sharpe: float = np.nan
    wf_sharpe_std: float = np.nan
    wf_passed: bool = False

    # IC stability
    ic_dispersion: float = np.nan  # std of IC across sub-periods

    # Tail capture metrics
    top_decile_spread_capture: float = np.nan
    market_upside_tail_capture: float = np.nan
    market_downside_tail_capture: float = np.nan
    market_tail_win_rate: float = np.nan
    upside_concentration: float = np.nan
    downside_concentration: float = np.nan

    def to_dict(self) -> dict:
        def _fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, float):
                return round(v, 6)
            return v

        return {
            "ic_mean": _fmt(self.ic_mean),
            "ic_std": _fmt(self.ic_std),
            "ic_ir": _fmt(self.ic_ir),
            "ic_hit_rate": _fmt(self.ic_hit_rate),
            "sharpe_ratio": _fmt(self.sharpe_ratio),
            "max_drawdown": _fmt(self.max_drawdown),
            "cumulative_return": _fmt(self.cumulative_return),
            "annualized_return": _fmt(self.annualized_return),
            "volatility": _fmt(self.volatility),
            "win_rate": _fmt(self.win_rate),
            "wf_sharpe": _fmt(self.wf_sharpe),
            "wf_sharpe_std": _fmt(self.wf_sharpe_std),
            "wf_passed": self.wf_passed,
            "ic_dispersion": _fmt(self.ic_dispersion),
            "top_decile_spread_capture": _fmt(self.top_decile_spread_capture),
            "market_upside_tail_capture": _fmt(self.market_upside_tail_capture),
            "market_downside_tail_capture": _fmt(self.market_downside_tail_capture),
            "market_tail_win_rate": _fmt(self.market_tail_win_rate),
            "upside_concentration": _fmt(self.upside_concentration),
            "downside_concentration": _fmt(self.downside_concentration),
        }


class FactorMimickingPortfolio:
    """Pure factor evaluation via cross-sectional factor mimicking portfolios.

    Standardizes factor values cross-sectionally each period, then uses
    standardized scores as portfolio weights. No trading frictions.

    Parameters
    ----------
    total_leverage : float
        Gross exposure (sum of absolute weights).
    rebalance_freq : str
        "daily", "weekly", or "monthly".
    long_only : bool
        If True, zero out negative weights (long-only portfolio).
    use_ranks : bool
        If True, use rank-percentile normalization (robust to outliers).
        If False, use z-score normalization.
    """

    def __init__(
        self,
        total_leverage: float = 1.0,
        rebalance_freq: str = "daily",
        long_only: bool = False,
        use_ranks: bool = True,
    ):
        self.total_leverage = total_leverage
        self.rebalance_freq = rebalance_freq
        self.long_only = long_only
        self.use_ranks = use_ranks
        self._trading_days_per_year = 252

    # ── weight computation ──────────────────────────────────────────────

    def compute_weights(
        self,
        factor_values: pd.Series,
    ) -> pd.DataFrame:
        """Compute mimicking portfolio weights for each date.

        Args:
            factor_values: MultiIndex (trade_date, symbol) Series.

        Returns:
            DataFrame (index=trade_date, columns=symbol) of weights.
        """
        if isinstance(factor_values.index, pd.MultiIndex):
            df = factor_values.unstack()  # rows=dates, cols=symbols
        else:
            df = factor_values.to_frame().T

        if self.use_ranks:
            ranked = df.rank(axis=1, pct=True)
            z = 2.0 * (ranked - 0.5)  # [-1, 1] uniform-like
        else:
            mu = df.mean(axis=1)
            sigma = df.std(axis=1).clip(lower=1e-10)
            z = df.sub(mu, axis=0).div(sigma, axis=0)

        # Zero out weights when cross-sectional dispersion is zero (all factor
        # values identical).  Tied ranks produce z ≈ 0.0003 instead of 0 for
        # constant rows, creating phantom equal-weight market exposure.
        # Binary / rare-event factors are especially affected (93%+ constant days).
        constant_rows = df.nunique(axis=1) <= 1
        if constant_rows.any():
            z.loc[constant_rows] = 0.0

        if self.long_only:
            z = z.clip(lower=0.0)

        # Normalize: sum(abs(weights)) = total_leverage
        abs_sum = z.abs().sum(axis=1).clip(lower=1e-10)
        weights = z.div(abs_sum, axis=0) * self.total_leverage

        return weights

    # ── return computation ──────────────────────────────────────────────

    def compute_return_stream(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """Compute daily portfolio returns.

        Portfolio return at date d = sum(weight_d[s] * forward_return_{d}[s]).
        Caller must ensure factor_values use only information known at date d
        (i.e., already shifted/lagged).

        Args:
            factor_values: MultiIndex (trade_date, symbol) Series.
            forward_returns: MultiIndex (trade_date, symbol) Series.

        Returns:
            Series (index=trade_date) of daily portfolio returns.
        """
        weights = self.compute_weights(factor_values)
        if isinstance(forward_returns.index, pd.MultiIndex):
            ret_df = forward_returns.unstack()
        else:
            ret_df = forward_returns.to_frame().T

        common_dates = weights.index.intersection(ret_df.index)
        common_symbols = weights.columns.intersection(ret_df.columns)

        w = weights.loc[common_dates, common_symbols]
        r = ret_df.loc[common_dates, common_symbols]

        daily_ret = (w * r).sum(axis=1)
        daily_ret.name = "fmp_return"
        return daily_ret

    # ── full evaluation ─────────────────────────────────────────────────

    def evaluate(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
    ) -> PureFactorMetrics:
        """Full evaluation: IC + factor mimicking portfolio metrics.

        Args:
            factor_values: MultiIndex (trade_date, symbol) Series.
            forward_returns: MultiIndex (trade_date, symbol) Series.

        Returns:
            PureFactorMetrics with all fields populated.
        """
        # IC metrics
        ic_mean, ic_std, ic_ir, ic_hit_rate = 0.0, 0.0, 0.0, 0.0
        try:
            aligned = pd.DataFrame({
                "factor": factor_values,
                "fwd_ret": forward_returns,
            }).dropna()

            if len(aligned) > 50:
                daily_ic = aligned.groupby("trade_date").apply(
                    lambda g: g["factor"].corr(g["fwd_ret"], method="spearman")
                    if len(g) >= 10 else np.nan
                )
                daily_ic = daily_ic.dropna()
                if len(daily_ic) > 5:
                    ic_mean = daily_ic.mean()
                    ic_std = daily_ic.std()
                    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
                    ic_hit_rate = (daily_ic > 0).mean()
        except Exception:
            pass

        # Factor mimicking portfolio metrics
        daily_ret = self.compute_return_stream(factor_values, forward_returns)
        daily_ret = daily_ret.dropna()

        if len(daily_ret) < 20:
            return PureFactorMetrics(
                ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
                ic_hit_rate=ic_hit_rate,
            )

        cum_ret = (1.0 + daily_ret).prod() - 1.0
        mean_daily = daily_ret.mean()
        daily_vol = daily_ret.std()
        ann_ret = mean_daily * self._trading_days_per_year
        ann_vol = daily_vol * np.sqrt(self._trading_days_per_year)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
        win_rate = (daily_ret > 0).mean()

        # Max drawdown
        cum_series = (1.0 + daily_ret).cumprod()
        running_max = cum_series.expanding().max()
        drawdowns = (cum_series - running_max) / running_max
        max_dd = drawdowns.min()

        # IC stability
        ic_dispersion = np.nan
        try:
            daily_ic_series = daily_ret  # reuse if IC not available
            if 'daily_ic' in dir() and len(daily_ic) > 20:
                n_splits = min(5, len(daily_ic) // 40)
                if n_splits >= 2:
                    splits = np.array_split(daily_ic.values, n_splits)
                    sub_means = [s.mean() for s in splits if len(s) > 10]
                    if len(sub_means) >= 2:
                        ic_dispersion = float(np.std(sub_means))
        except Exception:
            pass

        # ── Metric 1: Top-Decile Spread Capture ──────────────────────────
        top_decile_spread_capture = np.nan
        try:
            from factor.validation import quantile_returns, cross_sectional_range_return

            qr = quantile_returns(factor_values, forward_returns, n_quantiles=10)
            if not qr.empty:
                top_q = qr[qr["quantile"] == qr["quantile"].max()]
                bot_q = qr[qr["quantile"] == qr["quantile"].min()]
                ret_col = "fwd_ret" if "fwd_ret" in qr.columns else "return"
                spread_by_date = pd.merge(
                    top_q[["trade_date", ret_col]],
                    bot_q[["trade_date", ret_col]],
                    on="trade_date", suffixes=("_top", "_bot"),
                )
                spread_by_date["spread"] = (
                    spread_by_date[f"{ret_col}_top"] - spread_by_date[f"{ret_col}_bot"]
                )
                factor_mean_spread = spread_by_date["spread"].mean()

                cs_range = cross_sectional_range_return(forward_returns, n_quantiles=10)
                cs_mean_range = cs_range.mean()

                if (not np.isnan(factor_mean_spread)
                        and not np.isnan(cs_mean_range)
                        and abs(cs_mean_range) > 1e-10):
                    top_decile_spread_capture = factor_mean_spread / cs_mean_range
        except Exception:
            pass

        # ── Metric 2: Market Tail-Day Capture ────────────────────────────
        market_upside_tail_capture = np.nan
        market_downside_tail_capture = np.nan
        market_tail_win_rate = np.nan
        try:
            fwd_unstacked = forward_returns.unstack()
            market_daily = fwd_unstacked.mean(axis=1).dropna()

            if len(market_daily) >= 30:
                n_tail = max(1, int(len(market_daily) * 0.10))
                sorted_mkt = market_daily.sort_values()
                up_threshold = sorted_mkt.iloc[-n_tail]
                down_threshold = sorted_mkt.iloc[n_tail - 1]

                up_tail_dates = set(market_daily[market_daily >= up_threshold].index)
                down_tail_dates = set(market_daily[market_daily <= down_threshold].index)

                common_dates = daily_ret.index.intersection(market_daily.index)
                fmp_aligned = daily_ret.loc[common_dates]
                mkt_aligned = market_daily.loc[common_dates]

                # Upside capture
                up_fmp = fmp_aligned[fmp_aligned.index.isin(up_tail_dates)]
                up_mkt = mkt_aligned[mkt_aligned.index.isin(up_tail_dates)]
                if len(up_fmp) >= 3 and abs(up_mkt.mean()) > 1e-10:
                    market_upside_tail_capture = up_fmp.mean() / up_mkt.mean()

                # Downside capture
                down_fmp = fmp_aligned[fmp_aligned.index.isin(down_tail_dates)]
                down_mkt = mkt_aligned[mkt_aligned.index.isin(down_tail_dates)]
                if len(down_fmp) >= 3 and abs(down_mkt.mean()) > 1e-10:
                    market_downside_tail_capture = down_fmp.mean() / down_mkt.mean()

                # Win rate on all tail days
                all_tail_dates = up_tail_dates | down_tail_dates
                tail_fmp = fmp_aligned[fmp_aligned.index.isin(all_tail_dates)]
                if len(tail_fmp) >= 5:
                    market_tail_win_rate = (tail_fmp > 0).mean()
        except Exception:
            pass

        # ── Metric 3: Return Concentration in Tails ──────────────────────
        upside_concentration = np.nan
        downside_concentration = np.nan
        try:
            if len(daily_ret) >= 30:
                sorted_ret = daily_ret.sort_values()
                n_tail = max(1, int(len(sorted_ret) * 0.10))

                top_ret = sorted_ret.iloc[-n_tail:]
                pos_ret = daily_ret[daily_ret > 0]
                total_positive = pos_ret.sum()
                if total_positive > 1e-10:
                    upside_concentration = top_ret[top_ret > 0].sum() / total_positive

                bot_ret = sorted_ret.iloc[:n_tail]
                neg_ret = daily_ret[daily_ret < 0]
                total_negative = abs(neg_ret.sum())
                if total_negative > 1e-10:
                    downside_concentration = (
                        abs(bot_ret[bot_ret < 0].sum()) / total_negative
                    )
        except Exception:
            pass

        return PureFactorMetrics(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_hit_rate=ic_hit_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            volatility=ann_vol,
            mean_daily_return=mean_daily,
            win_rate=win_rate,
            ic_dispersion=ic_dispersion,
            top_decile_spread_capture=top_decile_spread_capture,
            market_upside_tail_capture=market_upside_tail_capture,
            market_downside_tail_capture=market_downside_tail_capture,
            market_tail_win_rate=market_tail_win_rate,
            upside_concentration=upside_concentration,
            downside_concentration=downside_concentration,
        )

    # ── multi-factor composite ──────────────────────────────────────────

    def compute_composite_weights(
        self,
        factor_weights: dict[str, float],
        factor_values: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Combine multiple factors into composite mimicking portfolio weights.

        Each factor is cross-sectionally standardized, then weighted by
        factor_weights and summed to form a composite score. The composite
        score is then converted to portfolio weights.

        Args:
            factor_weights: {factor_name: weight} mapping.
            factor_values: {factor_name: MultiIndex Series} mapping.

        Returns:
            DataFrame (index=trade_date, columns=symbol) of weights.
        """
        if not factor_weights or not factor_values:
            raise ValueError("factor_weights and factor_values must be non-empty")

        total_w = sum(abs(w) for w in factor_weights.values())
        if total_w < 1e-10:
            raise ValueError("Sum of absolute factor weights is zero")

        # Build composite z-score
        composite = None
        common_idx = None
        for name, weight in factor_weights.items():
            if name not in factor_values:
                continue
            fv = factor_values[name]
            df = fv.unstack() if isinstance(fv.index, pd.MultiIndex) else fv.to_frame().T

            if self.use_ranks:
                ranked = df.rank(axis=1, pct=True)
                z = 2.0 * (ranked - 0.5)
            else:
                mu = df.mean(axis=1)
                sigma = df.std(axis=1).clip(lower=1e-10)
                z = df.sub(mu, axis=0).div(sigma, axis=0)

            if common_idx is None:
                common_idx = z.index
                composite = z * (weight / total_w)
            else:
                common_idx = common_idx.intersection(z.index)
                composite = composite.reindex(common_idx) + z.reindex(common_idx) * (weight / total_w)

        if composite is None:
            raise ValueError("No valid factor values found")

        composite = composite.loc[common_idx]

        if self.long_only:
            composite = composite.clip(lower=0.0)

        abs_sum = composite.abs().sum(axis=1).clip(lower=1e-10)
        weights = composite.div(abs_sum, axis=0) * self.total_leverage

        return weights

    def evaluate_composite(
        self,
        factor_weights: dict[str, float],
        factor_values: dict[str, pd.Series],
        forward_returns: pd.Series,
    ) -> PureFactorMetrics:
        """Evaluate a weighted composite of multiple factors.

        Args:
            factor_weights: {factor_name: weight}.
            factor_values: {factor_name: MultiIndex Series}.
            forward_returns: MultiIndex (trade_date, symbol) Series.

        Returns:
            PureFactorMetrics.
        """
        weights = self.compute_composite_weights(factor_weights, factor_values)
        if isinstance(forward_returns.index, pd.MultiIndex):
            ret_df = forward_returns.unstack()
        else:
            ret_df = forward_returns.to_frame().T

        common_dates = weights.index.intersection(ret_df.index)
        common_symbols = weights.columns.intersection(ret_df.columns)
        w = weights.loc[common_dates, common_symbols]
        r = ret_df.loc[common_dates, common_symbols]
        daily_ret = (w * r).sum(axis=1)

        return self._metrics_from_returns(daily_ret)

    def _metrics_from_returns(self, daily_ret: pd.Series) -> PureFactorMetrics:
        """Compute portfolio metrics from a daily return series."""
        daily_ret = daily_ret.dropna()
        if len(daily_ret) < 20:
            return PureFactorMetrics()

        cum_ret = (1.0 + daily_ret).prod() - 1.0
        mean_daily = daily_ret.mean()
        daily_vol = daily_ret.std()
        ann_ret = mean_daily * self._trading_days_per_year
        ann_vol = daily_vol * np.sqrt(self._trading_days_per_year)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
        win_rate = (daily_ret > 0).mean()

        cum_series = (1.0 + daily_ret).cumprod()
        running_max = cum_series.expanding().max()
        drawdowns = (cum_series - running_max) / running_max
        max_dd = drawdowns.min()

        # Return concentration in tails
        upside_conc = np.nan
        downside_conc = np.nan
        try:
            if len(daily_ret) >= 30:
                sorted_ret = daily_ret.sort_values()
                n_tail = max(1, int(len(sorted_ret) * 0.10))

                top_ret = sorted_ret.iloc[-n_tail:]
                pos_sum = daily_ret[daily_ret > 0].sum()
                if pos_sum > 1e-10:
                    upside_conc = top_ret[top_ret > 0].sum() / pos_sum

                bot_ret = sorted_ret.iloc[:n_tail]
                neg_sum = abs(daily_ret[daily_ret < 0].sum())
                if neg_sum > 1e-10:
                    downside_conc = abs(bot_ret[bot_ret < 0].sum()) / neg_sum
        except Exception:
            pass

        return PureFactorMetrics(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            volatility=ann_vol,
            mean_daily_return=mean_daily,
            win_rate=win_rate,
            upside_concentration=upside_conc,
            downside_concentration=downside_conc,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    window_size: int = 252,
    step_size: int = 63,
    min_windows: int = 3,
    portfolio: FactorMimickingPortfolio | None = None,
) -> dict:
    """Rolling-window walk-forward validation.

    For each window:
      1. Train: compute IC metrics on training period
      2. Test: construct factor mimicking portfolio on test period, compute Sharpe.

    Args:
        factor_values: MultiIndex (trade_date, symbol) Series.
        forward_returns: MultiIndex (trade_date, symbol) Series.
        window_size: Trading days in each training window.
        step_size: Trading days to advance each fold.
        min_windows: Minimum number of complete windows required.
        portfolio: FactorMimickingPortfolio instance (created if None).

    Returns:
        dict with keys: train_ic_means, test_sharpes, mean_test_sharpe,
        sharpe_std, min_test_sharpe, all_positive, passed, n_windows.
    """
    if portfolio is None:
        portfolio = FactorMimickingPortfolio()

    dates = sorted(factor_values.index.get_level_values("trade_date").unique())
    n_dates = len(dates)

    if n_dates < window_size + step_size:
        logger.warning("WF: not enough dates (%d < %d)", n_dates, window_size + step_size)
        return {
            "train_ic_means": [], "test_sharpes": [],
            "mean_test_sharpe": np.nan, "sharpe_std": np.nan,
            "min_test_sharpe": np.nan, "all_positive": False,
            "passed": False, "n_windows": 0,
        }

    train_ic_means: list[float] = []
    test_sharpes: list[float] = []

    start = 0
    while start + window_size + step_size <= n_dates:
        train_end = start + window_size
        test_end = train_end + step_size

        train_dates = dates[start:train_end]
        test_dates = dates[train_end:test_end]

        train_mask = factor_values.index.get_level_values("trade_date").isin(train_dates)
        test_mask = factor_values.index.get_level_values("trade_date").isin(test_dates)

        fv_train = factor_values.loc[train_mask]
        fr_train = forward_returns.loc[forward_returns.index.get_level_values("trade_date").isin(train_dates)]

        # Train: compute IC
        aligned = pd.DataFrame({"factor": fv_train, "fwd_ret": fr_train}).dropna()
        if len(aligned) > 50:
            daily_ic = aligned.groupby("trade_date").apply(
                lambda g: g["factor"].corr(g["fwd_ret"], method="spearman")
                if len(g) >= 10 else np.nan
            ).dropna()
            if len(daily_ic) > 5:
                train_ic_means.append(daily_ic.mean())
            else:
                train_ic_means.append(0.0)
        else:
            train_ic_means.append(0.0)

        # Test: factor mimicking portfolio
        fv_test = factor_values.loc[test_mask]
        fr_test = forward_returns.loc[forward_returns.index.get_level_values("trade_date").isin(test_dates)]

        try:
            daily_ret = portfolio.compute_return_stream(fv_test, fr_test).dropna()
            if len(daily_ret) > 10:
                ann_vol = daily_ret.std() * np.sqrt(252)
                ann_ret = daily_ret.mean() * 252
                sr = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
                test_sharpes.append(sr)
            else:
                test_sharpes.append(0.0)
        except Exception:
            test_sharpes.append(0.0)

        start += step_size

    n_windows = len(test_sharpes)
    if n_windows < min_windows:
        return {
            "train_ic_means": train_ic_means,
            "test_sharpes": test_sharpes,
            "mean_test_sharpe": float(np.mean(test_sharpes)) if test_sharpes else np.nan,
            "sharpe_std": float(np.std(test_sharpes)) if test_sharpes else np.nan,
            "min_test_sharpe": float(np.min(test_sharpes)) if test_sharpes else np.nan,
            "all_positive": all(s > 0 for s in test_sharpes) if test_sharpes else False,
            "passed": False,
            "n_windows": n_windows,
        }

    mean_sr = float(np.mean(test_sharpes))
    std_sr = float(np.std(test_sharpes))
    min_sr = float(np.min(test_sharpes))
    all_pos = all(s > 0 for s in test_sharpes)

    passed = mean_sr > 0.0 and min_sr > -0.5

    return {
        "train_ic_means": train_ic_means,
        "test_sharpes": test_sharpes,
        "mean_test_sharpe": mean_sr,
        "sharpe_std": std_sr,
        "min_test_sharpe": min_sr,
        "all_positive": all_pos,
        "passed": passed,
        "n_windows": n_windows,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IC Stability
# ═══════════════════════════════════════════════════════════════════════════════

def ic_stability_metrics(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_regimes: int = 5,
) -> dict:
    """IC stability across equal-duration sub-periods.

    Args:
        factor_values: MultiIndex (trade_date, symbol) Series.
        forward_returns: MultiIndex (trade_date, symbol) Series.
        n_regimes: Number of equal-duration sub-periods.

    Returns:
        dict with: regime_ics, regime_hit_rates, ic_dispersion,
        worst_regime_ic, best_regime_ic.
    """
    aligned = pd.DataFrame({
        "factor": factor_values,
        "fwd_ret": forward_returns,
    }).dropna()

    if len(aligned) < 100:
        return {"regime_ics": [], "ic_dispersion": np.nan}

    daily_ic = aligned.groupby("trade_date").apply(
        lambda g: g["factor"].corr(g["fwd_ret"], method="spearman")
        if len(g) >= 10 else np.nan
    ).dropna()

    if len(daily_ic) < n_regimes * 10:
        return {"regime_ics": [], "ic_dispersion": np.nan}

    splits = np.array_split(daily_ic.values, n_regimes)
    regime_ics = []
    regime_hit_rates = []
    for s in splits:
        if len(s) > 5:
            regime_ics.append(float(np.mean(s)))
            regime_hit_rates.append(float((s > 0).mean()))

    return {
        "regime_ics": regime_ics,
        "regime_hit_rates": regime_hit_rates,
        "ic_dispersion": float(np.std(regime_ics)) if regime_ics else np.nan,
        "worst_regime_ic": float(np.min(regime_ics)) if regime_ics else np.nan,
        "best_regime_ic": float(np.max(regime_ics)) if regime_ics else np.nan,
    }
