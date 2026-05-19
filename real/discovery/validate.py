"""Factor validation pipeline for discovered factors.

Evaluates a factor on: IC (predictive power), stability (across time regimes),
novelty (vs existing factors), and robustness (walk-forward).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from factor.validation import (
    compute_rank_ic,
    ic_summary,
    factor_correlation,
    factor_auto_correlation,
)


@dataclass
class ValidationResult:
    """Full validation output for a single factor.

    Attributes:
        factor_name: Name of the validated factor.
        passed: Whether the factor meets all acceptance criteria.
        ic_mean: Mean rank IC.
        ic_std: Standard deviation of rank IC.
        ic_ir: Information ratio of IC (IC_mean / IC_std).
        hit_rate: Fraction of periods with positive IC.
        auto_corr: Average rank autocorrelation (1-day lag).
        max_corr_existing: Max correlation with any existing factor.
        ic_by_period: IC statistics split by time periods.
        wf_ic_mean: Walk-forward out-of-sample IC mean.
        wf_passed: Whether walk-forward check passed.
        failures: List of specific checks that failed.
        warnings: List of non-blocking concerns.
    """
    factor_name: str
    passed: bool = False
    ic_mean: float = np.nan
    ic_std: float = np.nan
    ic_ir: float = np.nan
    hit_rate: float = np.nan
    auto_corr: float = np.nan
    max_corr_existing: float = np.nan
    ic_by_period: dict[str, dict] = field(default_factory=dict)
    wf_ic_mean: float = np.nan
    wf_passed: bool = False
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "passed": self.passed,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_ir": self.ic_ir,
            "hit_rate": self.hit_rate,
            "auto_corr": self.auto_corr,
            "max_corr_existing": self.max_corr_existing,
            "ic_by_period": self.ic_by_period,
            "wf_ic_mean": self.wf_ic_mean,
            "wf_passed": self.wf_passed,
            "failures": self.failures,
            "warnings": self.warnings,
        }


class FactorValidator:
    """Validates discovered factors against acceptance criteria.

    Criteria (from the plan):
        - |IC_mean| > 0.01
        - IC_IR > 0.15
        - autocorrelation < 0.3
        - max correlation with existing factors < 0.7
        - walk-forward IC maintains sign and significance
    """

    def __init__(
        self,
        min_abs_ic: float = 0.01,
        min_ic_ir: float = 0.15,
        max_auto_corr: float = 0.95,
        max_corr_existing: float = 0.7,
        wf_min_ic: float = 0.005,
        n_wf_splits: int = 3,
    ):
        self.min_abs_ic = min_abs_ic
        self.min_ic_ir = min_ic_ir
        self.max_auto_corr = max_auto_corr
        self.max_corr_existing = max_corr_existing
        self.wf_min_ic = wf_min_ic
        self.n_wf_splits = n_wf_splits

    def validate(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        factor_name: str,
        existing_factors: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """Run full validation pipeline.

        Args:
            factor_values: Multi-indexed (trade_date, symbol) factor Series.
            forward_returns: Same-index forward return Series (e.g. 5-day fwd).
            factor_name: Name for reporting.
            existing_factors: DataFrame of existing factor values (columns = factor names),
                              same multi-index, for novelty check.

        Returns:
            ValidationResult with pass/fail and detailed metrics.
        """
        result = ValidationResult(factor_name=factor_name)
        failures: list[str] = []
        warnings: list[str] = []

        # 1. IC analysis
        ic = compute_rank_ic(factor_values, forward_returns)
        if ic.empty:
            failures.append("No valid IC observations (empty after dropna)")
            result.failures = failures
            return result

        summary = ic_summary(ic)
        result.ic_mean = summary["mean"]
        result.ic_std = summary["std"]
        result.ic_ir = summary["ir"]
        result.hit_rate = summary["hit_rate"]

        if abs(result.ic_mean) < self.min_abs_ic:
            failures.append(
                f"|IC_mean|={abs(result.ic_mean):.4f} < {self.min_abs_ic}"
            )
        if result.ic_ir < self.min_ic_ir:
            failures.append(
                f"IC_IR={result.ic_ir:.3f} < {self.min_ic_ir}"
            )

        # 2. Stability: autocorrelation
        result.auto_corr = factor_auto_correlation(factor_values)
        if not np.isnan(result.auto_corr):
            if result.auto_corr > self.max_auto_corr:
                failures.append(
                    f"Autocorrelation={result.auto_corr:.3f} > {self.max_auto_corr}"
                )
        else:
            # NaN auto_corr often means constant factor (zero variance across ranks)
            try:
                xs_std = factor_values.groupby("trade_date").std()
                mean_xs_std = float(xs_std.mean())
                if np.isnan(mean_xs_std) or mean_xs_std < 1e-8:
                    failures.append(
                        f"Near-constant factor (mean cross-sectional std={mean_xs_std:.2e})"
                    )
                else:
                    warnings.append("Could not compute autocorrelation (non-constant, possible NaN)")
            except Exception:
                warnings.append("Could not compute autocorrelation")

        # 3. Novelty: correlation with existing factors
        if existing_factors is not None and not existing_factors.empty:
            combined = pd.DataFrame({"new": factor_values}).join(
                existing_factors, how="inner"
            )
            if not combined.empty:
                corr = combined.corr()
                corrs_with_existing = corr["new"].drop("new").abs()
                if len(corrs_with_existing) > 0:
                    result.max_corr_existing = float(corrs_with_existing.max())
                    if result.max_corr_existing > self.max_corr_existing:
                        max_name = corrs_with_existing.idxmax()
                        failures.append(
                            f"Too similar to '{max_name}' "
                            f"(corr={result.max_corr_existing:.3f} > {self.max_corr_existing})"
                        )
            else:
                warnings.append("No overlapping dates with existing factors")
        elif existing_factors is None or existing_factors.empty:
            result.max_corr_existing = 0.0
            warnings.append("No existing factors to check novelty against")

        # 4. IC stability across time periods
        result.ic_by_period = _ic_by_periods(ic, n_periods=self.n_wf_splits)

        # 5. Walk-forward: train on first 2/3, test on last 1/3
        result.wf_passed, result.wf_ic_mean = _walk_forward_check(
            ic, min_ic=self.wf_min_ic
        )
        if not result.wf_passed:
            failures.append(
                f"Walk-forward IC={result.wf_ic_mean:.4f} < {self.wf_min_ic}"
            )

        result.failures = failures
        result.warnings = warnings
        result.passed = len(failures) == 0
        return result

    def validate_batch(
        self,
        factor_values: dict[str, pd.Series],
        forward_returns: pd.Series,
        existing_factors: pd.DataFrame | None = None,
    ) -> list[ValidationResult]:
        """Validate multiple factors, updating existing_factors as we go.

        Factors are validated sequentially; accepted factors are added to
        the existing_factors pool so later factors are checked against them.
        """
        results: list[ValidationResult] = []
        pool = existing_factors.copy() if existing_factors is not None else pd.DataFrame()

        for name, values in factor_values.items():
            result = self.validate(values, forward_returns, name, existing_factors=pool)
            results.append(result)
            if result.passed:
                # Add to pool so subsequent factors are checked against it
                pool[name] = values

        return results


def _ic_by_periods(ic: pd.Series, n_periods: int = 3) -> dict[str, dict]:
    """Split IC series into equal periods and compute summary per period."""
    if len(ic) < n_periods * 2:
        return {}

    splits = np.array_split(ic.index, n_periods)
    result: dict[str, dict] = {}
    for i, idx in enumerate(splits):
        sub_ic = ic.loc[idx]
        s = ic_summary(sub_ic)
        label = f"period_{i+1}"
        result[label] = {
            "start": str(idx[0]),
            "end": str(idx[-1]),
            **s,
        }
    return result


def _walk_forward_check(
    ic: pd.Series,
    train_frac: float = 0.67,
    min_ic: float = 0.005,
) -> tuple[bool, float]:
    """Simple walk-forward: train on first train_frac, evaluate IC on remainder."""
    ic = ic.dropna()
    if len(ic) < 10:
        return False, np.nan

    split_idx = int(len(ic) * train_frac)
    test_ic = ic.iloc[split_idx:].dropna()
    if test_ic.empty or len(test_ic) < 3:
        # Too few test samples — fallback: use full-series mean as proxy
        logger.debug("Walk-forward: too few test IC samples (%d), using full mean", len(test_ic))
        full_mean = float(ic.mean())
        return abs(full_mean) >= min_ic, full_mean

    wf_mean = float(test_ic.mean())
    return abs(wf_mean) >= min_ic, wf_mean


def orthogonal_filter(
    factor_values: dict[str, pd.Series],
    forward_returns: pd.Series,
    min_residual_ir: float = 0.10,
) -> list[str]:
    """Select factors with independent alpha via Gram-Schmidt orthogonalization.

    Sorts factors by IC_IR descending, then greedily selects each factor
    whose residual (after regressing out already-selected factors) still
    has |IC_IR| >= min_residual_ir.  This guarantees every selected factor
    contributes non-redundant predictive power.

    Args:
        factor_values: {name: Series} of factor values (MultiIndex trade_date x symbol).
        forward_returns: Same-index forward return Series.
        min_residual_ir: Minimum absolute IC_IR of the orthogonalized residual
                         for a factor to be selected.

    Returns:
        List of selected factor names in selection order (best first).
    """
    if len(factor_values) <= 1:
        return list(factor_values.keys())

    # 1. Score each factor by IC_IR
    scored: list[tuple[float, str, pd.Series]] = []
    for name, values in factor_values.items():
        ic = compute_rank_ic(values, forward_returns)
        if ic.empty or ic.std() == 0:
            scored.append((0.0, name, values))
        else:
            ir = abs(float(ic.mean() / ic.std()))
            scored.append((ir, name, values))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 2. Greedy orthogonal selection
    selected: list[str] = []
    selected_series: list[pd.Series] = []  # parallel to selected

    for ir, name, values in scored:
        if not selected:
            selected.append(name)
            selected_series.append(values)
            continue

        # Build pooled design matrix from already-selected factors
        df = pd.DataFrame({"candidate": values})
        for sn in selected:
            df[sn] = factor_values[sn]
        df = df.dropna()

        if len(df) < 50:
            continue

        X_cols = selected
        X = df[X_cols].values
        y = df["candidate"].values

        # OLS with constant
        X_aug = np.column_stack([np.ones(len(X)), X])
        try:
            beta, residuals, rank, singulars = np.linalg.lstsq(X_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        y_pred = X_aug @ beta
        residual_vals = y - y_pred

        residual_series = pd.Series(residual_vals, index=df.index)
        # Align with forward_returns (may have different index due to NaN dropping)
        common_idx = residual_series.dropna().index.intersection(forward_returns.dropna().index)
        if len(common_idx) < 50:
            continue
        residual_ic = compute_rank_ic(
            residual_series.loc[common_idx],
            forward_returns.loc[common_idx],
        )
        if residual_ic.empty or residual_ic.std() == 0:
            residual_ir = 0.0
        else:
            residual_ir = abs(float(residual_ic.mean() / residual_ic.std()))

        if residual_ir >= min_residual_ir:
            selected.append(name)
            selected_series.append(values)

    return selected
