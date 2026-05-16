"""Double Machine Learning (Chernozhukov et al. 2018) for causal inference.

Implements DML with K-fold cross-fitting and GBDT nuisance functions
to estimate average treatment effects (ATE) and conditional average
treatment effects (CATE) in the presence of high-dimensional confounders.

Ref: Chernozhukov, Chetverikov, Demirer, et al. (2018)
     "Double/debiased machine learning for treatment and structural parameters"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DMLResult:
    """Result of a DML estimation.

    Attributes:
        ate: Average treatment effect.
        ate_se: Standard error of ATE.
        ate_t_stat: t-statistic for ATE significance.
        ate_p_value: p-value for ATE against H₀: θ=0.
        cate: Series of conditional ATE per sample (if CATE estimated).
        nuisance_scores: Cross-validated R² for nuisance models.
        n_obs: Number of observations used.
    """
    ate: float
    ate_se: float
    ate_t_stat: float
    ate_p_value: float
    cate: pd.Series | None = None
    nuisance_scores: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0

    @property
    def significant(self, alpha: float = 0.05) -> bool:
        return self.ate_p_value < alpha

    def summary(self) -> str:
        lines = [
            f"ATE: {self.ate:.6f} (SE={self.ate_se:.6f})",
            f"t-stat: {self.ate_t_stat:.3f}, p-value: {self.ate_p_value:.4f}",
            f"Significant at 5%: {self.significant}",
            f"N: {self.n_obs}",
        ]
        if self.nuisance_scores:
            lines.append(f"Nuisance R²: {self.nuisance_scores}")
        return "\n".join(lines)


class DoubleML:
    """Double ML estimator with GBDT nuisance functions.

    Supports binary and continuous treatment. Uses K-fold cross-fitting
    to avoid overfitting bias from the nuisance models.

    Usage:
        dml = DoubleML(n_folds=5)
        result = dml.estimate_ate(Y=returns, T=treatment, X=confounders)
        print(result.summary())
    """

    def __init__(
        self,
        n_folds: int = 5,
        nuisance_model: str = "gbdt",
        gbdt_params: dict | None = None,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.nuisance_model = nuisance_model
        self.gbdt_params = gbdt_params or {
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_samples_leaf": 20,
            "random_state": random_state,
        }
        self.random_state = random_state

    # ── Public API ──────────────────────────────────────────────────────────

    def estimate_ate(
        self,
        Y: pd.Series,
        T: pd.Series,
        X: pd.DataFrame,
        T_binary: bool | None = None,
    ) -> DMLResult:
        """Estimate average treatment effect.

        Args:
            Y: Outcome variable (e.g., forward returns).
            T: Treatment variable (binary or continuous).
            X: Confounders / control variables.
            T_binary: Whether T is binary. Auto-detected if None.

        Returns:
            DMLResult with ATE and inference statistics.
        """
        Y, T, X = self._align(Y, T, X)
        if len(Y) < 50:
            return DMLResult(ate=np.nan, ate_se=np.nan, ate_t_stat=np.nan,
                             ate_p_value=1.0, n_obs=len(Y))

        if T_binary is None:
            T_binary = self._is_binary(T)

        # Cross-fitting: out-of-fold nuisance predictions
        Y_residuals, T_residuals = self._cross_fit(Y, T, X, T_binary)

        # Orthogonalized regression: Y_resid ~ T_resid
        theta, se = self._final_regression(Y_residuals, T_residuals)

        t_stat = theta / se if se > 0 else 0.0
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return DMLResult(
            ate=float(theta),
            ate_se=float(se),
            ate_t_stat=float(t_stat),
            ate_p_value=float(p_value),
            n_obs=len(Y),
        )

    def estimate_cate(
        self,
        Y: pd.Series,
        T: pd.Series,
        X: pd.DataFrame,
        T_binary: bool | None = None,
    ) -> DMLResult:
        """Estimate conditional average treatment effects.

        Returns per-sample CATE values in addition to ATE statistics.
        The CATE is estimated by fitting a final GBDT on the orthogonalized
        residuals: θ̂(x) = f(Y_resid / T_resid ~ X).
        """
        Y, T, X = self._align(Y, T, X)
        if len(Y) < 50:
            return DMLResult(ate=np.nan, ate_se=np.nan, ate_t_stat=np.nan,
                             ate_p_value=1.0, n_obs=len(Y))

        if T_binary is None:
            T_binary = self._is_binary(T)

        Y_residuals, T_residuals = self._cross_fit(Y, T, X, T_binary)

        # ATE from orthogonalized regression
        theta, se = self._final_regression(Y_residuals, T_residuals)

        # CATE: fit treatment effect as function of X
        # Use the doubly-robust score: ψ = (Y - m₀(X)) - θ * (T - e(X))
        # But simplified: pseudo-outcome = Y_resid / T_resid (clipped)
        pseudo_outcome = Y_residuals / (T_residuals + 1e-10)
        pseudo_outcome = pseudo_outcome.clip(
            lower=pseudo_outcome.quantile(0.01),
            upper=pseudo_outcome.quantile(0.99),
        )

        cate_model = self._make_regressor()
        try:
            cate_model.fit(X, pseudo_outcome)
            cate_values = cate_model.predict(X)
        except Exception:
            cate_values = np.full(len(Y), theta)

        cate_series = pd.Series(cate_values, index=Y.index, name="cate")

        t_stat = theta / se if se > 0 else 0.0
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return DMLResult(
            ate=float(theta),
            ate_se=float(se),
            ate_t_stat=float(t_stat),
            ate_p_value=float(p_value),
            cate=cate_series,
            n_obs=len(Y),
        )

    # ── Cross-fitting ───────────────────────────────────────────────────────

    def _cross_fit(
        self,
        Y: pd.Series,
        T: pd.Series,
        X: pd.DataFrame,
        T_binary: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """K-fold cross-fitting of nuisance functions.

        Returns:
            (Y_residuals, T_residuals) as numpy arrays.
        """
        n = len(Y)
        fold_ids = np.arange(n) % self.n_folds
        np.random.seed(self.random_state)
        np.random.shuffle(fold_ids)

        Y_resid = np.zeros(n)
        T_resid = np.zeros(n)

        for k in range(self.n_folds):
            train_idx = fold_ids != k
            test_idx = fold_ids == k

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            T_train = T.iloc[train_idx]

            # --- Outcome model: m₀(X) = E[Y | X] (ignoring T) ---
            m_model = self._make_regressor()
            try:
                m_model.fit(X_train, Y_train)
                Y_pred = m_model.predict(X_test)
            except Exception:
                Y_pred = np.zeros(test_idx.sum())

            Y_resid[test_idx] = Y_test.values - Y_pred

            # --- Propensity model: e(X) = P(T=1 | X) or E[T | X] ---
            if T_binary:
                e_model = self._make_classifier()
                try:
                    e_model.fit(X_train, T_train)
                    T_pred = e_model.predict_proba(X_test)[:, 1]
                except Exception:
                    T_pred = np.full(test_idx.sum(), T_train.mean())
            else:
                e_model = self._make_regressor()
                try:
                    e_model.fit(X_train, T_train)
                    T_pred = e_model.predict(X_test)
                except Exception:
                    T_pred = np.full(test_idx.sum(), T_train.mean())

            T_resid[test_idx] = T.iloc[test_idx].values - T_pred

        return Y_resid, T_resid

    def _final_regression(
        self, Y_resid: np.ndarray, T_resid: np.ndarray
    ) -> tuple[float, float]:
        """OLS of Y_resid on T_resid (no intercept because residuals are
        mean-zero by construction).
        """
        # theta = (T_resid' Y_resid) / (T_resid' T_resid)
        mask = np.isfinite(Y_resid) & np.isfinite(T_resid)
        Y_r = Y_resid[mask]
        T_r = T_resid[mask]

        if len(T_r) < 2:
            return 0.0, np.inf

        # Include intercept for robustness
        X_design = np.column_stack([np.ones(len(T_r)), T_r])
        try:
            beta = np.linalg.lstsq(X_design, Y_r, rcond=None)[0]
            theta = beta[1]
            residuals = Y_r - X_design @ beta
            # Heteroskedasticity-robust (HC1) standard error
            n = len(residuals)
            se = np.sqrt(
                np.sum((X_design[:, 1] * residuals) ** 2)
                / (np.sum(X_design[:, 1] ** 2) ** 2)
                * n / max(n - 2, 1)
            )
        except np.linalg.LinAlgError:
            return 0.0, np.inf

        return float(theta), float(se)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _align(
        Y: pd.Series, T: pd.Series, X: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Align Y, T, X on common non-null index."""
        common = Y.dropna().index.intersection(T.dropna().index).intersection(
            X.dropna().index
        )
        return Y.loc[common], T.loc[common], X.loc[common]

    @staticmethod
    def _is_binary(T: pd.Series) -> bool:
        return set(T.unique()) <= {0, 1, 0.0, 1.0, True, False}

    def _make_regressor(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**self.gbdt_params)

    def _make_classifier(self):
        from sklearn.ensemble import GradientBoostingClassifier
        params = {k: v for k, v in self.gbdt_params.items()
                  if k != "n_estimators"}
        params["n_estimators"] = self.gbdt_params.get("n_estimators", 100)
        return GradientBoostingClassifier(**params)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return float(0.5 * (1 + np.math.erf(x / np.sqrt(2))))
