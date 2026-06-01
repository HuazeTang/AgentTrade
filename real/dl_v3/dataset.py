"""V3 Dataset: pre-launch detection with quiet-now + surge-soon labels.

Key difference from V2:
  - Positive: past 20d max return < 10% AND forward [T+1, T+15] >= 30%
  - "Stock is quiet now, will surge sometime in next 15 days"
  - Uses V3 features (OHLCV + consolidation + divergence + per-stock z-score)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from llm.yaogu.case_extractor import YaoguCaseExtractor
from dl_v3.derived_features import (
    ALL_V3_COLUMNS,
    PER_STOCK_ZSCORE_COLS,
    build_v3_feature_cache,
)

logger = logging.getLogger(__name__)

# ── V3 Config ──
SEQUENCE_LENGTH = 20
FORWARD_START = 1             # start from T+1 (no skip)
FORWARD_END = 15              # look up to T+15
MIN_CUM_RET = 0.30            # 30% minimum forward return
MAX_PRE_RET = 0.10            # past 20d max return must be < 10% (not already launched)
PRE_WINDOW = 20               # lookback for pre-check


class YaoguDatasetV3(Dataset):
    """V3 Dataset for pre-launch yaogu detection.

    Each item is (X, y) where:
      X: (sequence_length, n_features) normalized features
      y: 1 if stock was quiet (past 20d < 10%) then surged (fwd 5-15d >= 30%)
         0 otherwise
    """

    def __init__(
        self,
        daily_cache: pd.DataFrame,
        symbols: list[str],
        start_date: date,
        end_date: date,
        *,
        sequence_length: int = SEQUENCE_LENGTH,
        forward_start: int = FORWARD_START,
        forward_end: int = FORWARD_END,
        min_cum_ret: float = MIN_CUM_RET,
        max_pre_ret: float = MAX_PRE_RET,
        pre_window: int = PRE_WINDOW,
        feature_cols: list[str] | None = None,
        max_neg_per_pos: int = 20,
        feature_cache: pd.DataFrame | None = None,
        candidate_symbols: set[str] | None = None,
    ):
        self.sequence_length = sequence_length
        self.forward_start = forward_start
        self.forward_end = forward_end
        self._feature_cols = feature_cols or ALL_V3_COLUMNS

        close = daily_cache["close"].unstack()
        self._dates = sorted(close.index)
        self._symbols = [s for s in symbols if s in close.columns]

        # ── Build or use feature cache ──
        if feature_cache is not None:
            raw_cache = feature_cache
            logger.info("YaoguDatasetV3: using pre-computed feature cache (%d rows)",
                         len(raw_cache))
        else:
            raw_cache = build_v3_feature_cache(daily_cache)
            logger.info("YaoguDatasetV3: built V3 feature cache internally")

        # Verify feature columns
        available = [c for c in self._feature_cols if c in raw_cache.columns]
        if len(available) < len(self._feature_cols):
            missing = set(self._feature_cols) - set(available)
            logger.warning("Missing V3 feature columns: %s", missing)
        self._feature_cols = available

        # Unstack each feature column for O(1) lookups
        self._feature_dfs: dict[str, pd.DataFrame] = {}
        for col in self._feature_cols:
            self._feature_dfs[col] = raw_cache[col].unstack().sort_index()

        # ── Build V3 labels ──
        self._samples = []
        first_col = self._feature_cols[0] if self._feature_cols else None
        cache_dates = set(self._feature_dfs[first_col].index) if first_col else set()

        pos_count = 0
        neg_buffer: list[dict] = []

        # Valid date range: need pre_window before + forward_end after
        start_idx = max(sequence_length, pre_window)
        end_idx = len(self._dates) - forward_end - 1

        for i in range(start_idx, end_idx):
            td = self._dates[i]
            td_date = _to_date(td)
            if td_date < start_date:
                continue
            if td_date > end_date:
                continue
            if pd.Timestamp(td) not in cache_dates:
                continue

            for sym in self._symbols:
                if candidate_symbols and sym not in candidate_symbols:
                    continue

                if sym not in close.columns:
                    continue

                # ── Pre-check: past PRE_WINDOW max return must be < max_pre_ret ──
                pre_start = i - pre_window
                pre_prices = close.iloc[pre_start:i + 1][sym].values
                pre_prices = pre_prices[~np.isnan(pre_prices) & (pre_prices > 0)]
                if len(pre_prices) < 5:
                    continue
                pre_max_ret = float((np.max(pre_prices) - pre_prices[0]) / pre_prices[0])
                if pre_max_ret >= max_pre_ret:
                    continue  # already launched, skip

                # ── Forward check: max return in [T+forward_start, T+forward_end] ──
                fwd_s = min(i + forward_start, len(self._dates) - 1)
                fwd_e = min(i + forward_end, len(self._dates) - 1)
                if fwd_s >= fwd_e:
                    continue

                fwd_prices = close.iloc[fwd_s:fwd_e + 1][sym].values
                fwd_prices = fwd_prices[~np.isnan(fwd_prices) & (fwd_prices > 0)]
                if len(fwd_prices) < 2:
                    continue

                start_price = close.iloc[i][sym]
                if pd.isna(start_price) or start_price <= 0:
                    continue

                fwd_max_ret = float((np.max(fwd_prices) - start_price) / start_price)

                sample = {"date_idx": i, "symbol": sym, "date": td}

                if fwd_max_ret >= min_cum_ret:
                    self._samples.append({**sample, "label": 1})
                    pos_count += 1
                else:
                    neg_buffer.append({**sample, "label": 0})

        # Subsample negatives
        max_neg = pos_count * max_neg_per_pos
        neg_count = len(neg_buffer)
        if neg_count > max_neg and pos_count > 0:
            rng = np.random.default_rng(42)
            keep_idx = rng.choice(neg_count, size=max_neg, replace=False)
            neg_buffer = [neg_buffer[i] for i in keep_idx]

        self._samples.extend(neg_buffer)

        logger.info("YaoguDatasetV3: %d samples (%d pos, %d neg, ratio 1:%.1f)",
                     len(self._samples), pos_count, len(neg_buffer),
                     len(neg_buffer) / max(pos_count, 1))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        i = sample["date_idx"]
        sym = sample["symbol"]

        seq_start = i - self.sequence_length + 1
        seq_dates = self._dates[seq_start:i + 1]

        feats = []
        for col in self._feature_cols:
            df = self._feature_dfs.get(col)
            if df is not None and sym in df.columns:
                vals = df.loc[df.index.isin(seq_dates), sym].values
                if len(vals) < self.sequence_length:
                    vals = np.pad(vals, (0, self.sequence_length - len(vals)),
                                  constant_values=np.nan)
            else:
                vals = np.full(self.sequence_length, np.nan)
            feats.append(vals[:self.sequence_length])

        X = np.column_stack(feats).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(X),
            torch.tensor(sample["label"], dtype=torch.float32),
        )

    @property
    def positive_rate(self) -> float:
        pos = sum(1 for s in self._samples if s["label"] == 1)
        return pos / len(self._samples) if self._samples else 0

    @property
    def n_features(self) -> int:
        return len(self._feature_cols)

    @property
    def feature_dfs(self) -> dict[str, pd.DataFrame]:
        return self._feature_dfs

    def get_all_samples(self) -> list[dict]:
        return list(self._samples)


def compute_sample_weights(dataset: YaoguDatasetV3) -> torch.Tensor:
    """Inverse-frequency weights for WeightedRandomSampler."""
    labels = np.array([s["label"] for s in dataset._samples])
    pos_count = int(labels.sum())
    neg_count = len(labels) - pos_count
    if pos_count == 0 or neg_count == 0:
        return torch.ones(len(dataset))
    weights = np.where(labels == 1, 1.0 / pos_count, 1.0 / neg_count)
    return torch.from_numpy(weights).float()


def _to_date(val) -> date:
    if hasattr(val, "date"):
        val = val.date()
    if hasattr(val, "date"):
        return val.date()
    if isinstance(val, pd.Timestamp):
        return val.date()
    return val


def build_dataloaders_v3(
    daily_cache: pd.DataFrame,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    batch_size: int = 2048,
    num_workers: int = 0,
    max_neg_per_pos: int = 20,
    feature_cache: pd.DataFrame | None = None,
    candidate_symbols: set[str] | None = None,
    use_weighted_sampler: bool = True,
    **dataset_kwargs,
) -> tuple[YaoguDatasetV3, YaoguDatasetV3, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build V3 train and validation datasets + dataloaders."""

    train_ds = YaoguDatasetV3(
        daily_cache, symbols,
        start_date=train_start, end_date=train_end,
        max_neg_per_pos=max_neg_per_pos,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        **dataset_kwargs,
    )
    val_ds = YaoguDatasetV3(
        daily_cache, symbols,
        start_date=val_start, end_date=val_end,
        max_neg_per_pos=max_neg_per_pos,  # don't subsample val
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        **dataset_kwargs,
    )

    if use_weighted_sampler:
        train_weights = compute_sample_weights(train_ds)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            train_weights, num_samples=len(train_weights), replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=False, drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    logger.info("DataLoaders V3: train=%d batches, val=%d batches",
                 len(train_loader), len(val_loader))
    return train_ds, val_ds, train_loader, val_loader
