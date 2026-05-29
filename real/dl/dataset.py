"""PyTorch Dataset for yaogu detection: (sequence, label) pairs from OHLCV data.

Each sample is a 60-day sequence of CS z-scored derived features for a single
stock on a given date. Labels = 1 if forward 10-day max return >= 30% with
>= 2 consecutive limit-ups.

All features are per-day cross-sectionally z-scored (no per-sequence normalization).
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
from dl.derived_features import (
    DERIVED_FEATURE_COLUMNS,
    RAW_OHLCV_COLUMNS,
    build_normalized_feature_cache,
)

logger = logging.getLogger(__name__)

# ── Config ──
SEQUENCE_LENGTH = 20          # trading days of history (~1 month)
FORWARD_WINDOW = 10           # forward days for label
MIN_CUM_RET = 0.30            # 30% minimum forward return
MIN_LIMIT_UP = 2              # consecutive limit-up days


class YaoguDataset(Dataset):
    """PyTorch Dataset for 妖股 binary classification.

    Each item is (X, y) where:
      X: (sequence_length, n_features) CS-normalized derived features
      y: 0/1 label (1 = becomes 妖股 within forward window)

    All features are per-day cross-sectionally winsorized + z-scored at
    construction time. No per-sequence normalization is applied.
    """

    def __init__(
        self,
        daily_cache: pd.DataFrame,
        symbols: list[str],
        start_date: date,
        end_date: date,
        *,
        sequence_length: int = SEQUENCE_LENGTH,
        forward_window: int = FORWARD_WINDOW,
        min_cum_ret: float = MIN_CUM_RET,
        min_limit_up: int = MIN_LIMIT_UP,
        feature_cols: list[str] | None = None,
        max_neg_per_pos: int = 20,
        feature_cache: pd.DataFrame | None = None,
        candidate_symbols: set[str] | None = None,
        return_year: bool = False,
        raw_ohlcv: bool = False,
    ):
        self.sequence_length = sequence_length
        self.forward_window = forward_window
        self._feature_cols = feature_cols or (RAW_OHLCV_COLUMNS if raw_ohlcv else DERIVED_FEATURE_COLUMNS)
        self._return_year = return_year

        # Unstack close for label computation
        close = daily_cache["close"].unstack()
        self._dates = sorted(close.index)
        self._symbols = [s for s in symbols if s in close.columns]

        # ── Build or use feature cache ──
        if feature_cache is not None:
            raw_cache = feature_cache
            logger.info("YaoguDataset: using pre-computed feature cache (%d rows)",
                         len(raw_cache))
        else:
            raw_cache = build_normalized_feature_cache(daily_cache, raw_ohlcv=raw_ohlcv)
            logger.info("YaoguDataset: built feature cache internally")

        # Verify feature columns
        available = [c for c in self._feature_cols if c in raw_cache.columns]
        if len(available) < len(self._feature_cols):
            missing = set(self._feature_cols) - set(available)
            logger.warning("Missing feature columns in cache: %s", missing)
        self._feature_cols = available

        # Unstack each feature column for O(1) date×symbol lookups in __getitem__
        self._feature_dfs: dict[str, pd.DataFrame] = {}
        for col in self._feature_cols:
            self._feature_dfs[col] = raw_cache[col].unstack().sort_index()

        # ── Build labels ──
        extractor = YaoguCaseExtractor(
            forward_window=forward_window,
            min_cum_ret=min_cum_ret,
            min_limit_up=min_limit_up,
        )

        fwd_max_ret = extractor._compute_forward_max_return(
            close, self._dates, forward_window, self._symbols
        )
        limit_up_mask = extractor._detect_limit_ups(
            close, daily_cache["pre_close"].unstack(),
            self._dates, self._symbols
        )

        # ── Build sample list ──
        start_idx = sequence_length
        end_idx = len(self._dates) - forward_window - 1

        self._samples: list[dict] = []
        # Use first feature DataFrame's index for available dates
        first_col = self._feature_cols[0] if self._feature_cols else None
        cache_dates = set(self._feature_dfs[first_col].index) if first_col else set()

        pos_count = 0
        neg_buffer: list[dict] = []

        for i in range(start_idx, end_idx):
            td = self._dates[i]
            td_date = _to_date(td)
            if td_date < start_date:
                continue
            if td_date > end_date:
                continue

            # Skip dates not in feature cache
            if pd.Timestamp(td) not in cache_dates:
                continue

            for sym in self._symbols:
                if candidate_symbols and sym not in candidate_symbols:
                    continue

                fwd_ret = fwd_max_ret.iloc[i][sym] if sym in fwd_max_ret.columns else None
                if pd.isna(fwd_ret):
                    continue

                fwd_end = min(i + forward_window, len(self._dates) - 1)
                lu_streak = extractor._max_consecutive_limit_up(
                    limit_up_mask, self._dates, i + 1, fwd_end, sym
                )

                sample = {"date_idx": i, "symbol": sym, "date": td}

                if fwd_ret >= min_cum_ret and lu_streak >= min_limit_up:
                    self._samples.append({**sample, "label": 1})
                    pos_count += 1
                else:
                    neg_buffer.append({**sample, "label": 0})

        # Subsample negatives to control ratio
        max_neg = pos_count * max_neg_per_pos
        neg_count = len(neg_buffer)
        if neg_count > max_neg and pos_count > 0:
            rng = np.random.default_rng(42)
            keep_idx = rng.choice(neg_count, size=max_neg, replace=False)
            neg_buffer = [neg_buffer[i] for i in keep_idx]
            self._sample_neg_kept = max_neg

        self._samples.extend(neg_buffer)

        # Year labels for domain adversarial training
        self._years = sorted(set(_to_date(s["date"]).year for s in self._samples))
        self._year_to_idx = {y: i for i, y in enumerate(self._years)}
        for s in self._samples:
            s["year_label"] = self._year_to_idx[_to_date(s["date"]).year]

        logger.info("YaoguDataset: %d samples (%d pos, %d neg, ratio 1:%.1f)",
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

        # Build sequence from unstacked feature DataFrames (O(1) per column)
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

        if self._return_year:
            return (
                torch.from_numpy(X),
                torch.tensor(sample["label"], dtype=torch.float32),
                torch.tensor(sample["year_label"], dtype=torch.long),
            )
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
    def n_years(self) -> int:
        return len(self._years)

    @property
    def feature_dfs(self) -> dict[str, pd.DataFrame]:
        return self._feature_dfs

    def get_all_samples(self) -> list[dict]:
        """Return all samples with metadata (for screener training data extraction)."""
        return list(self._samples)


def compute_sample_weights(dataset: YaoguDataset) -> torch.Tensor:
    """Inverse-frequency weights for WeightedRandomSampler.

    Each positive sample gets weight 1/n_pos, negative gets 1/n_neg,
    so each class is sampled with equal probability per batch.
    """
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


def build_dataloaders(
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
    use_weighted_sampler: bool = False,
    return_year: bool = False,
    raw_ohlcv: bool = False,
    **dataset_kwargs,
) -> tuple[YaoguDataset, YaoguDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and validation datasets + dataloaders.

    Args:
        feature_cache: Optional pre-computed normalized feature cache (shared across stages).
        candidate_symbols: If provided, only include these symbols as samples.
        use_weighted_sampler: If True, use WeightedRandomSampler for balanced batches.
        return_year: If True, __getitem__ returns (X, y, year_label).
    """

    train_ds = YaoguDataset(
        daily_cache, symbols,
        start_date=train_start, end_date=train_end,
        max_neg_per_pos=max_neg_per_pos,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        return_year=return_year,
        raw_ohlcv=raw_ohlcv,
        **dataset_kwargs,
    )
    val_ds = YaoguDataset(
        daily_cache, symbols,
        start_date=val_start, end_date=val_end,
        max_neg_per_pos=max_neg_per_pos,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        return_year=return_year,
        raw_ohlcv=raw_ohlcv,
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

    logger.info("DataLoaders: train=%d batches, val=%d batches",
                 len(train_loader), len(val_loader))
    return train_ds, val_ds, train_loader, val_loader
