"""Two-stage yaogu factor: screener (high recall) → DL model (high precision).

Loads Stage 1 LR screener and Stage 2 DualTowerModel, runs cascade inference:
1. Screener filters candidates from full market
2. DL model scores only candidates
3. Non-candidates get score 0.0
"""

from __future__ import annotations

import logging
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
    RAW_OHLCV_COLUMNS,
    build_normalized_feature_cache,
)
from dl.screener import YaoguScreener, DEFAULT_SCREENER_PATH

logger = logging.getLogger(__name__)

DEFAULT_DL_PATH = "data/models/yaogu_best.pt"
SEQUENCE_LENGTH = 20


@register_factor
class TwoStageYaoguFactor(Factor):
    """Two-stage yaogu detection factor.

    Stage 1: LR screener filters market to ~20-40% candidates (high recall).
    Stage 2: DualTowerModel scores candidates with DL model (high precision).

    Non-candidates receive a score of 0.0 (ranked last in composite scoring).
    """

    meta = FactorMeta(
        name="yaogu_two_stage",
        category="pattern",
        description="Two-stage yaogu detection: LR screener → CNN+Transformer DL model",
        lookback_days=SEQUENCE_LENGTH,
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "open", "high", "low", "volume", "amount", "pre_close"]

    def __init__(
        self,
        screener_path: str | None = None,
        dl_path: str | None = None,
        device: str | None = None,
        feature_cache: pd.DataFrame | None = None,
    ):
        super().__init__()
        self._screener_path = Path(screener_path or DEFAULT_SCREENER_PATH)
        self._dl_path = Path(dl_path or DEFAULT_DL_PATH)
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._feature_cache = feature_cache

        self._screener: YaoguScreener | None = None
        self._dl_model: DualTowerModel | None = None
        self._dl_threshold: float = 0.5
        self._feature_cols: list[str] = DERIVED_FEATURE_COLUMNS
        self._loaded = False

    def _load_models(self) -> None:
        if self._loaded:
            return

        # Load screener
        if self._screener_path.exists():
            self._screener = YaoguScreener.load(self._screener_path)
            logger.info("Loaded screener (threshold=%.4f)", self._screener.threshold)
        else:
            logger.warning("Screener not found: %s", self._screener_path)

        # Load DL model
        if self._dl_path.exists():
            checkpoint = torch.load(self._dl_path, map_location=self._device, weights_only=False)
            model_kwargs = checkpoint.get("model_kwargs", {})
            # Auto-detect feature set from checkpoint
            self._feature_cols = checkpoint.get("feature_cols", DERIVED_FEATURE_COLUMNS)
            self._dl_model = DualTowerModel(**model_kwargs)
            self._dl_model.load_state_dict(checkpoint["model_state_dict"])
            self._dl_model.to(self._device)
            self._dl_model.eval()
            self._dl_threshold = checkpoint.get("threshold", 0.5)
            logger.info("Loaded DL model (threshold=%.2f, params=%d, features=%d cols)",
                         self._dl_threshold,
                         sum(p.numel() for p in self._dl_model.parameters()),
                         len(self._feature_cols))
        else:
            logger.warning("DL model not found: %s", self._dl_path)

        self._loaded = True

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Run two-stage inference on all stocks/dates."""
        self._load_models()
        result = pd.Series(np.nan, index=data.index, name=self.meta.name)

        if self._screener is None or self._dl_model is None:
            return result

        if self._feature_cache is None:
            self._feature_cache = build_normalized_feature_cache(data)

        close = data["close"].unstack()
        all_dates = sorted(close.index)
        symbols = sorted(data.index.get_level_values("symbol").unique())

        feature_cols = [c for c in self._feature_cols
                        if c in self._feature_cache.columns]
        in_features = len(feature_cols)

        for i in range(SEQUENCE_LENGTH, len(all_dates)):
            td = all_dates[i]
            td_pd = pd.Timestamp(td)

            # ── Stage 1: Screener ──
            try:
                screener_scores = self._screener.score_day(
                    self._feature_cache, td_pd, symbols
                )
            except Exception as e:
                logger.debug("Screener failed for %s: %s", td, e)
                continue

            if screener_scores.empty:
                continue

            candidates = screener_scores[screener_scores >= self._screener.threshold]
            if candidates.empty:
                continue

            # ── Stage 2: DL model on candidates ──
            seq_start = i - SEQUENCE_LENGTH
            seq_dates = all_dates[seq_start:i]

            batch_feats = []
            batch_syms = []

            for sym in candidates.index:
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
                batch_syms.append(sym)

            if not batch_feats:
                continue

            X_tensor = torch.from_numpy(np.stack(batch_feats)).to(self._device)
            with torch.no_grad():
                probs = self._dl_model.predict_proba(X_tensor).cpu().numpy()

            for sym, prob in zip(batch_syms, probs):
                idx = (td_pd, sym)
                if idx in result.index:
                    result.loc[idx] = float(prob) if prob >= self._dl_threshold else 0.0

            if i % 100 == 0:
                logger.debug("TwoStage: %s: %d candidates → %d DL scores",
                             td.date() if hasattr(td, 'date') else td,
                             len(candidates), len(batch_syms))

        logger.info("TwoStageYaoguFactor: %d values (%.1f%% non-zero)",
                     result.notna().sum(),
                     (result > 0).sum() / max(result.notna().sum(), 1) * 100)
        return result.sort_index()
