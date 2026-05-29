"""Stage 1 yaogu screener: lightweight logistic regression for high-recall screening.

The screener is trained on single-day derived features (no sequence) to maximize
recall at the cost of precision. Its job is to reduce the candidate pool from
the full market (~800 stocks) to ~20-40%, while capturing >=95% of true yaogu events.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor
from dl.derived_features import (
    DERIVED_FEATURE_COLUMNS,
    build_normalized_feature_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_SCREENER_PATH = "data/models/yaogu_screener.joblib"


class YaoguScreener:
    """Logistic regression screener tuned for high recall."""

    def __init__(
        self,
        recall_target: float = 0.95,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self._recall_target = recall_target
        self._threshold: float = 0.5
        self._feature_cols: list[str] = []
        self._is_fitted = False

        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )

    # ── Properties ──

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    @property
    def coef_(self) -> np.ndarray | None:
        return self._model.coef_ if self._is_fitted else None

    # ── Training ──

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: list[str] | None = None,
    ) -> "YaoguScreener":
        """Train LR model and tune decision threshold for recall >= recall_target.

        Args:
            X: Feature matrix (n_samples, n_features) — already normalized.
            y: Binary labels (0/1).
            feature_cols: Feature column names for logging.

        Returns:
            self (fitted).
        """
        if feature_cols:
            self._feature_cols = list(feature_cols)
        else:
            self._feature_cols = [f"feat_{i}" for i in range(X.shape[1])]

        pos_rate = y.mean()
        logger.info("Screener training: %d samples, %.2f%% positive", len(y), pos_rate * 100)

        if pos_rate < 0.001 or pos_rate > 0.3:
            logger.warning("Screener: extreme class balance (%.4f%%), results may be unstable",
                           pos_rate * 100)

        self._model.fit(X, y)
        probs = self._model.predict_proba(X)[:, 1]
        self._threshold, metrics = self._tune_threshold(probs, y)
        self._is_fitted = True

        logger.info("Screener fitted: threshold=%.4f, recall=%.3f, precision=%.3f, "
                     "candidate_rate=%.2f%%",
                     self._threshold, metrics["recall"], metrics["precision"],
                     metrics["candidate_rate"] * 100)
        return self

    def _tune_threshold(
        self, probs: np.ndarray, y: np.ndarray,
    ) -> tuple[float, dict]:
        """Find the highest threshold that still achieves recall >= recall_target.

        Iterates thresholds from HIGH to LOW. Picks the first (highest) threshold
        meeting the recall target — this minimizes candidate rate while maintaining
        the recall guarantee. If no threshold meets the target, picks the one with
        best recall.
        """
        thresholds = np.linspace(0.02, 0.98, 97)
        best_threshold = 0.5
        best_metrics = {}
        best_recall_fallback = 0.0
        best_fallback = {}

        # Iterate HIGH → LOW: pick the tightest filter that still meets recall target
        for t in sorted(thresholds, reverse=True):
            preds = (probs >= t).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            candidate_rate = (tp + fp) / len(y)

            # Track best recall as fallback
            if recall > best_recall_fallback:
                best_recall_fallback = recall
                best_fallback = {
                    "threshold": float(t), "recall": recall,
                    "precision": precision, "candidate_rate": candidate_rate,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn),
                }

            if recall >= self._recall_target:
                best_threshold = t
                best_metrics = {
                    "threshold": float(t), "recall": recall,
                    "precision": precision, "candidate_rate": candidate_rate,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn),
                }
                break
        else:
            # No threshold meets recall target — use best recall available
            best_metrics = best_fallback
            best_threshold = best_fallback.get("threshold", 0.5)
            logger.warning(
                "Screener: no threshold meets recall >= %.2f. "
                "Best available: recall=%.3f at threshold=%.3f",
                self._recall_target, best_recall_fallback, best_threshold,
            )

        return best_threshold, best_metrics

    # ── Inference ──

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return yaogu probability scores (shape: n_samples,)."""
        if not self._is_fitted:
            raise RuntimeError("Screener not fitted")
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary mask (True = candidate passes screener)."""
        return self.predict_proba(X) >= self._threshold

    def score_day(
        self,
        feature_cache: pd.DataFrame,
        td: pd.Timestamp,
        symbols: list[str],
    ) -> pd.Series:
        """Score all stocks on a single day.

        Returns a Series indexed by symbol with probability scores.
        """
        try:
            day_data = feature_cache.xs(td, level="trade_date")
        except KeyError:
            return pd.Series(dtype=float)

        cols = [c for c in self._feature_cols if c in day_data.columns]
        if not cols:
            return pd.Series(dtype=float)

        common_syms = [s for s in symbols if s in day_data.index]
        if not common_syms:
            return pd.Series(dtype=float)

        X = day_data.loc[common_syms][cols].fillna(0).values.astype(np.float64)
        probs = self.predict_proba(X)
        return pd.Series(probs, index=common_syms)

    # ── Persistence ──

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "threshold": self._threshold,
            "recall_target": self._recall_target,
            "feature_cols": self._feature_cols,
        }, path)
        logger.info("Screener saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, recall_target: float = 0.95) -> "YaoguScreener":
        data = joblib.load(path)
        screener = cls(recall_target=data.get("recall_target", recall_target))
        screener._model = data["model"]
        screener._threshold = data["threshold"]
        screener._feature_cols = data.get("feature_cols", [])
        screener._is_fitted = True
        logger.info("Screener loaded from %s (threshold=%.4f)", path, screener._threshold)
        return screener


# ── Registered Factor ────────────────────────────────────────────────────

@register_factor
class YaoguScreenerFactor(Factor):
    """Stage 1 screener as a registered Factor for integration into the scoring pipeline."""

    meta = FactorMeta(
        name="yaogu_screener",
        category="pattern",
        description="Logistic regression screener: high-recall yaogu candidate filter",
        lookback_days=60,
    )

    def __init__(
        self,
        screener_path: str | None = None,
        feature_cache: pd.DataFrame | None = None,
    ):
        super().__init__()
        self._screener_path = screener_path or DEFAULT_SCREENER_PATH
        self._feature_cache = feature_cache
        self._screener: YaoguScreener | None = None
        self._loaded = False

    @property
    def required_fields(self) -> list[str]:
        return ["close", "open", "high", "low", "volume", "amount", "pre_close"]

    def _load_screener(self) -> None:
        if self._loaded:
            return
        path = Path(self._screener_path)
        if not path.exists():
            logger.warning("Screener not found: %s — factor returns NaN", path)
            self._loaded = True
            return
        self._screener = YaoguScreener.load(path)
        self._loaded = True

    def compute(self, data: pd.DataFrame) -> pd.Series:
        self._load_screener()
        result = pd.Series(np.nan, index=data.index, name=self.meta.name)

        if self._screener is None:
            return result

        if self._feature_cache is None:
            self._feature_cache = build_normalized_feature_cache(data)

        symbols = sorted(data.index.get_level_values("symbol").unique())
        trade_dates = sorted(data.index.get_level_values("trade_date").unique())

        for td in trade_dates:
            try:
                scores = self._screener.score_day(
                    self._feature_cache, pd.Timestamp(td), symbols
                )
                for sym, prob in scores.items():
                    result.loc[(pd.Timestamp(td), sym)] = prob
            except Exception as e:
                logger.debug("Screener compute failed for %s: %s", td, e)

        logger.info("YaoguScreenerFactor: computed %d values, mean=%.4f",
                     result.notna().sum(), result.mean())
        return result.sort_index()
