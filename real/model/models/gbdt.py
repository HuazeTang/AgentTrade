"""Gradient boosting decision tree alpha model (LightGBM backend)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.base import AlphaModel


class GBDTModel(AlphaModel):
    """LightGBM wrapper for alpha prediction.

    Falls back to sklearn GradientBoostingRegressor if lightgbm is not installed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 20,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs
        self._model = None
        self._feature_names: list[str] = []
        self._importance: pd.Series | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> GBDTModel:
        self._feature_names = list(X.columns)

        try:
            import lightgbm as lgb

            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                verbose=-1,
                **self.kwargs,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            self._model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                **self.kwargs,
            )

        self._model.fit(X.values, y.values, sample_weight=sample_weight)
        self._importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(X.values)

    def save(self, path: str) -> None:
        import joblib

        state = {
            "model": self._model,
            "feature_names": self._feature_names,
            "importance": self._importance,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> GBDTModel:
        import joblib

        state = joblib.load(path)
        obj = cls()
        obj._model = state["model"]
        obj._feature_names = state["feature_names"]
        obj._importance = state.get("importance")
        return obj

    @property
    def feature_importance(self) -> pd.Series:
        if self._importance is None:
            return pd.Series(dtype=float)
        return self._importance
