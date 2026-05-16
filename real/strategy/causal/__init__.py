"""Causal inference strategies.

Uses Double ML, PSM, and event studies to estimate treatment effects
and construct long-short portfolios around causal signals.

Unlike traditional factor strategies that rely on correlation (IC),
causal strategies aim to identify true treatment effects by controlling
for confounders.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategy.base import Strategy
from strategy.causal.dml_strategy import DMLCausalStrategy
from strategy.causal.event_strategy import EventStudyStrategy

logger = logging.getLogger(__name__)


class CausalStrategy(Strategy):
    """Strategy that constructs portfolios based on estimated treatment effects.

    Generic framework: plug in any Treatment + causal method combination.

    The strategy:
    1. Computes treatment assignments T from data at decision time
    2. Estimates conditional average treatment effects (CATE) using DML
       trained on a rolling lookback of (Y, T, X)
    3. Goes long on stocks with positive CATE, short on negative CATE

    Refits the causal model periodically (default: weekly).
    """

    name = "causal"

    def __init__(
        self,
        treatment,              # causal.Treatment
        method=None,            # causal.DoubleML or PSM instance
        confounders: list[str] | None = None,
        forward_periods: int = 5,
        top_n: int = 20,
        long_only: bool = False,
        refit_freq: str = "W-MON",
        min_train_days: int = 60,
    ):
        self._treatment = treatment
        self._method = method
        self._confounders = confounders or []
        self.forward_periods = forward_periods
        self.top_n = top_n
        self.long_only = long_only
        self.refit_freq = refit_freq
        self.min_train_days = min_train_days

        # State
        self._cate: pd.Series | None = None
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
        # Ensure data is symbol-indexed (single date)
        if isinstance(data.index, pd.MultiIndex):
            if "trade_date" in data.index.names:
                data = data.xs(date, level="trade_date")

        # Refit if needed
        if self._should_refit(date):
            self._refit(date, data)

        if self._cate is None or len(self._cate) == 0:
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

    def _should_refit(self, date: pd.Timestamp) -> bool:
        if self._last_refit_date is None:
            return True
        if self.refit_freq == "daily":
            return True
        if self.refit_freq == "W-MON" and date.dayofweek == 0:
            return True
        if self.refit_freq == "monthly" and date.day <= 7 and date.dayofweek == 0:
            return True
        return False

    def _refit(self, date: pd.Timestamp, current_data: pd.DataFrame) -> None:
        """Refit causal model on current data snapshot."""
        try:
            from causal.dml import DoubleML

            T = self._treatment.compute_treatment(current_data)
            if T.sum() < 5:
                return

            # Forward returns as outcome
            if "close" not in current_data.columns:
                return

            Y = current_data["close"].pct_change().shift(-self.forward_periods)

            X_cols = self._confounders or [
                c for c in current_data.columns
                if c not in ("close", "symbol", "trade_date")
            ]
            X = current_data[X_cols].select_dtypes(include=[np.number])

            Y, T, X = Y.align(T, join="inner"), T.align(X, join="inner"), X
            common = Y.dropna().index.intersection(T.dropna().index).intersection(X.dropna().index)
            Y, T, X = Y.loc[common], T.loc[common], X.loc[common]

            if len(Y) < 30:
                return

            dml = self._method or DoubleML(n_folds=3)
            result = dml.estimate_cate(Y, T, X)

            if result.cate is not None:
                self._cate = result.cate
                self._last_refit_date = date
                logger.info("Causal model refit: ATE=%.6f, p=%.4f, n=%d",
                            result.ate, result.ate_p_value, result.n_obs)
        except Exception as e:
            logger.warning("Causal model refit failed: %s", e)
