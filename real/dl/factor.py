"""Factor wrapper for trained DualTowerModel.

Registers a Factor that loads the trained model and runs batch inference on
each day's data. The output is a probability score (0-1) usable in the
existing composite scoring system.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from factor.base import Factor, FactorMeta
from factor.registry import register_factor
from dl import DualTowerModel
from dl.derived_features import (
    DERIVED_FEATURE_COLUMNS,
    build_normalized_feature_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "data/models/yaogu_best.pt"
SEQUENCE_LENGTH = 60


@register_factor
class DlYaoguFactor(Factor):
    """Deep learning 妖股 detection factor.

    Uses a dual-tower CNN+Transformer model trained on 60-day OHLCV sequences
    to predict the probability of a stock becoming a 妖股 in the next 10 days.
    """

    meta = FactorMeta(
        name="dl_yaogu",
        category="pattern",
        description="Dual-tower CNN+Transformer probability of yaogu launch (10d >30% with limit-ups)",
        lookback_days=SEQUENCE_LENGTH,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "open", "high", "low", "volume", "amount", "pre_close"]

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 feature_cache = None):
        super().__init__()
        self._model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._model: DualTowerModel | None = None
        self._threshold: float = 0.5
        self._loaded = False
        self._feature_cache = feature_cache

    def _load_model(self) -> None:
        """Lazy-load the model from checkpoint."""
        if self._loaded:
            return

        if not self._model_path.exists():
            logger.warning("Model checkpoint not found: %s — factor returns NaN", self._model_path)
            self._loaded = True
            return

        checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)
        model_kwargs = checkpoint.get("model_kwargs", {})
        self._model = DualTowerModel(**model_kwargs)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()
        self._threshold = checkpoint.get("threshold", 0.5)
        self._loaded = True
        logger.info("DlYaoguFactor: loaded model (threshold=%.2f, params=%d) on %s",
                     self._threshold,
                     sum(p.numel() for p in self._model.parameters()),
                     self._device)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute yaogu probability for all stocks on all dates.

        Uses per-day CS z-scored derived features (no per-sequence normalization).
        """
        self._load_model()

        result = pd.Series(np.nan, index=data.index, name=self.meta.name)

        if self._model is None:
            return result

        if self._feature_cache is None:
            self._feature_cache = build_normalized_feature_cache(data)

        close = data["close"].unstack()
        dates = sorted(close.index)
        symbols = sorted(data.index.get_level_values("symbol").unique())

        feature_cols = [c for c in DERIVED_FEATURE_COLUMNS
                        if c in self._feature_cache.columns]

        if len(dates) < SEQUENCE_LENGTH + 1:
            return result

        for i in range(SEQUENCE_LENGTH, len(dates)):
            td = dates[i]
            seq_start = i - SEQUENCE_LENGTH
            seq_dates = dates[seq_start:i]

            batch_feats = []
            batch_symbols = []

            for sym in symbols:
                try:
                    mask = (
                        self._feature_cache.index.get_level_values("trade_date").isin(seq_dates) &
                        (self._feature_cache.index.get_level_values("symbol") == sym)
                    )
                    rows = self._feature_cache.loc[mask][feature_cols]
                except (KeyError, IndexError):
                    continue

                if len(rows) < SEQUENCE_LENGTH:
                    continue

                rows = rows.sort_index(level="trade_date")
                X = rows.values[:SEQUENCE_LENGTH].astype(np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                batch_feats.append(X)
                batch_symbols.append(sym)

            if not batch_feats:
                continue

            X_tensor = torch.from_numpy(np.stack(batch_feats)).to(self._device)
            with torch.no_grad():
                probs = self._model.predict_proba(X_tensor).cpu().numpy()

            for sym, prob in zip(batch_symbols, probs):
                idx = (pd.Timestamp(td), sym)
                if idx in result.index:
                    result.loc[idx] = float(prob)

        logger.info("DlYaoguFactor: computed %d values, mean=%.4f",
                     result.notna().sum(), result.mean())
        return result.sort_index()
