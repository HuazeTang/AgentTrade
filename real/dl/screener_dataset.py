"""Dataset builder for Stage 1 screener training.

Builds (X, y) numpy arrays from single-day derived features + yaogu labels.
Unlike YaoguDataset, this produces point-in-time feature vectors (not sequences).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from dl.derived_features import (
    DERIVED_FEATURE_COLUMNS,
    build_normalized_feature_cache,
)
from dl.dataset import YaoguDataset

logger = logging.getLogger(__name__)


class ScreenerDataset:
    """Build (X, y) for sklearn from single-day normalized features.

    Each row is one (trade_date, symbol) with:
      X = normalized derived features at that date
      y = 1 if forward 10d max ret >= 30% + 2 limit-ups
    """

    def __init__(
        self,
        daily_cache: pd.DataFrame,
        symbols: list[str],
        start_date: date,
        end_date: date,
        *,
        feature_cols: list[str] | None = None,
        max_neg_per_pos: int | None = None,  # None = keep all negatives
        feature_cache: pd.DataFrame | None = None,
        min_limit_up: int = 2,
        min_cum_ret: float = 0.30,
    ):
        self._feature_cols = feature_cols or DERIVED_FEATURE_COLUMNS

        # Use YaoguDataset to get labeled samples (with sequence_length=1 for single-day)
        # Build feature cache if not provided
        if feature_cache is not None:
            self._feature_cache = feature_cache
        else:
            self._feature_cache = build_normalized_feature_cache(daily_cache)

        # Build labels by scanning dates
        close = daily_cache["close"].unstack()
        dates = sorted(close.index)
        syms = [s for s in symbols if s in close.columns]

        from llm.yaogu.case_extractor import YaoguCaseExtractor
        extractor = YaoguCaseExtractor(
            forward_window=10, min_cum_ret=min_cum_ret, min_limit_up=min_limit_up,
        )
        fwd_max_ret = extractor._compute_forward_max_return(close, dates, 10, syms)
        limit_up_mask = extractor._detect_limit_ups(
            close, daily_cache["pre_close"].unstack(), dates, syms
        )

        cache_dates = set(self._feature_cache.index.get_level_values("trade_date").unique())

        self.X_list: list[np.ndarray] = []
        self.y_list: list[int] = []
        self._meta: list[dict] = []

        start_idx = 60  # need history for derived features
        end_idx = len(dates) - 10 - 1

        pos_count = 0
        neg_samples: list[tuple[np.ndarray, dict]] = []

        for i in range(start_idx, end_idx):
            td = dates[i]
            td_date = _to_date(td)
            if td_date < start_date or td_date > end_date:
                continue
            if pd.Timestamp(td) not in cache_dates:
                continue

            try:
                day_data = self._feature_cache.xs(pd.Timestamp(td), level="trade_date")
            except KeyError:
                continue

            cols_avail = [c for c in self._feature_cols if c in day_data.columns]

            for sym in syms:
                if sym not in day_data.index:
                    continue

                fwd_ret = fwd_max_ret.iloc[i][sym] if sym in fwd_max_ret.columns else None
                if pd.isna(fwd_ret):
                    continue

                fwd_end = min(i + 10, len(dates) - 1)
                lu_streak = extractor._max_consecutive_limit_up(
                    limit_up_mask, dates, i + 1, fwd_end, sym
                )

                feat_vec = day_data.loc[sym][cols_avail].fillna(0).values.astype(np.float64)
                label = 1 if (fwd_ret >= min_cum_ret and lu_streak >= min_limit_up) else 0
                meta = {"date": td, "symbol": sym, "fwd_ret": fwd_ret}

                if label == 1:
                    self.X_list.append(feat_vec)
                    self.y_list.append(1)
                    self._meta.append(meta)
                    pos_count += 1
                else:
                    neg_samples.append((feat_vec, {**meta, "label": 0}))

        # Subsample negatives
        if max_neg_per_pos is not None and pos_count > 0:
            max_neg = pos_count * max_neg_per_pos
            if len(neg_samples) > max_neg:
                rng = np.random.default_rng(42)
                idxs = rng.choice(len(neg_samples), size=max_neg, replace=False)
                neg_samples = [neg_samples[i] for i in idxs]

        for vec, meta in neg_samples:
            self.X_list.append(vec)
            self.y_list.append(0)
            self._meta.append(meta)

        logger.info("ScreenerDataset: %d samples (%d positive, %.2f%% pos)",
                     len(self.X_list), pos_count, pos_count / max(len(self.X_list), 1) * 100)

    @property
    def X(self) -> np.ndarray:
        return np.array(self.X_list) if self.X_list else np.empty((0, len(self._feature_cols)))

    @property
    def y(self) -> np.ndarray:
        return np.array(self.y_list, dtype=int)

    @property
    def feature_cols(self) -> list[str]:
        return [c for c in self._feature_cols if c in self._feature_cache.columns]

    @property
    def feature_cache(self) -> pd.DataFrame:
        return self._feature_cache

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y


def _to_date(val) -> date:
    if hasattr(val, "date"):
        val = val.date()
    if hasattr(val, "date"):
        return val.date()
    if isinstance(val, pd.Timestamp):
        return val.date()
    return val
