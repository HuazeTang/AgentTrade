"""DML-based causal strategy.

Uses Double Machine Learning to estimate heterogeneous treatment effects
from a continuous or binary treatment, then ranks stocks by CATE.

Example treatments:
- "Is this stock in the top quantile of momentum?" → binary treatment
- "How large was the earnings surprise?" → continuous treatment (dose-response)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from causal.dml import DoubleML, DMLResult
from causal.treatment import ContinuousTreatment, ThresholdTreatment
from strategy.base import Strategy

logger = logging.getLogger(__name__)


class DMLCausalStrategy(Strategy):
    """Strategy using Double ML to estimate treatment effects for ranking.

    At each refit cycle:
    1. Compute treatment T from a factor threshold or continuous exposure
    2. Compute forward returns Y
    3. Fit DML: Y_resid = θ * T_resid + ε, with cross-fitting
    4. Estimate CATE per stock
    5. Go long on positive CATE, short on negative CATE
    """

    name = "dml_causal"

    def __init__(
        self,
        treatment_col: str = "",
        treatment_quantile: float = 0.7,
        confounders: list[str] | None = None,
        forward_periods: int = 5,
        top_n: int = 20,
        long_only: bool = False,
        refit_freq: str = "W-MON",
        dml_folds: int = 5,
        min_history: int = 120,
    ):
        self.treatment_col = treatment_col
        self.treatment_quantile = treatment_quantile
        self.confounders = confounders or []
        self.forward_periods = forward_periods
        self.top_n = top_n
        self.long_only = long_only
        self.refit_freq = refit_freq
        self.dml_folds = dml_folds
        self.min_history = min_history

        # Rolling state
        self._history_Y: list[pd.Series] = []
        self._history_T: list[pd.Series] = []
        self._history_X: list[pd.DataFrame] = []
        self._cate: pd.Series | None = None
        self._last_result: DMLResult | None = None
        self._last_refit_date: pd.Timestamp | None = None

    @property
    def allow_short(self) -> bool:
        return not self.long_only

    def generate_weights(
        self,
        date: pd.Timestamp,
        universe: list[str],
        data: pd.DataFrame,
        prices: pd.Series,
        current_positions: dict[str, float],
        cash: float,
    ) -> pd.Series:
        # Collect this date's data into history
        self._collect_history(date, data)

        # Refit if needed
        if self._should_refit(date):
            self._refit()

        # Use latest CATE for ranking
        if self._cate is None:
            return pd.Series(dtype=float)

        cate = self._cate.reindex(universe).dropna()
        if cate.empty:
            return pd.Series(dtype=float)

        cate = cate.sort_values(ascending=False)
        long_n = min(self.top_n, len(cate))

        long_syms = cate.head(long_n).index.tolist()
        short_syms = [] if self.long_only else cate.tail(long_n).index.tolist()

        weights = pd.Series(0.0, index=data.index)
        if long_syms:
            weights.loc[long_syms] = 1.0 / len(long_syms)
        if short_syms:
            weights.loc[short_syms] = -1.0 / len(short_syms)

        total = weights.abs().sum()
        if total > 0:
            weights = weights / total
        return weights[weights != 0]

    def _collect_history(self, date: pd.Timestamp, data: pd.DataFrame) -> None:
        """Accumulate one period of (Y, T, X) for causal estimation."""
        if self.treatment_col and self.treatment_col in data.columns:
            # Continuous treatment
            T = data[self.treatment_col].copy()
            # Binarize at quantile for binary DML
            threshold = T.quantile(self.treatment_quantile)
            T_binary = (T > threshold).astype(int)
        else:
            return

        if "close" in data.columns:
            # Approximate forward return (simplified for single-date data)
            Y = data["close"].pct_change().shift(-self.forward_periods)
        else:
            return

        X_cols = self.confounders or [c for c in data.columns
                                      if c not in ("close", self.treatment_col)]
        X = data[X_cols].select_dtypes(include=[np.number])

        self._history_Y.append(Y.dropna())
        self._history_T.append(T_binary.dropna())
        self._history_X.append(X.dropna())

        # Keep rolling window
        max_hist = max(self.min_history * 2, 500)
        if len(self._history_Y) > max_hist:
            self._history_Y = self._history_Y[-max_hist:]
            self._history_T = self._history_T[-max_hist:]
            self._history_X = self._history_X[-max_hist:]

    def _should_refit(self, date: pd.Timestamp) -> bool:
        if self._last_refit_date is None:
            return True
        if self.refit_freq == "daily":
            return True
        if self.refit_freq == "W-MON" and date.dayofweek == 0:
            return True
        return False

    def _refit(self) -> None:
        """Refit DML on accumulated history."""
        if len(self._history_Y) < self.min_history:
            return

        try:
            Y = pd.concat(self._history_Y)
            T = pd.concat(self._history_T)
            X = pd.concat(self._history_X)

            # Align
            common = Y.index.intersection(T.index).intersection(X.index)
            Y, T, X = Y.loc[common], T.loc[common], X.loc[common]

            dml = DoubleML(n_folds=self.dml_folds)
            result = dml.estimate_cate(Y, T, X)

            self._last_result = result
            if result.cate is not None:
                self._cate = result.cate
            self._last_refit_date = pd.Timestamp.now()
            logger.info(
                "DML refit: ATE=%.6f (p=%.4f), CATE std=%.4f, n=%d",
                result.ate, result.ate_p_value,
                result.cate.std() if result.cate is not None else 0,
                result.n_obs,
            )
        except Exception as e:
            logger.warning("DML refit failed: %s", e)

    @property
    def last_result(self) -> DMLResult | None:
        return self._last_result
