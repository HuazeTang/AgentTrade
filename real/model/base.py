"""Abstract base class for alpha models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AlphaModel(ABC):
    """ML model that predicts forward returns (alpha signal)."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> AlphaModel:
        """Train the model.

        Args:
            X: Features (n_samples, n_features).
            y: Target forward returns.
            sample_weight: Optional per-sample weights.

        Returns:
            self (for chaining).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted alpha for each row."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> AlphaModel:
        """Deserialize model from disk."""
        ...

    @property
    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Feature importance scores indexed by feature name."""
        ...
