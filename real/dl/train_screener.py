"""Training entry point for Stage 1 screener (logistic regression).

Usage:
    python -m dl.train_screener --train-start 2020-01-01 --train-end 2023-12-31 \\
                                --val-start 2024-01-01 --val-end 2024-12-31

Trains a YaoguScreener on single-day derived features, tunes threshold for
95% recall, and validates candidate-pool reduction rate.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from dl.derived_features import build_normalized_feature_cache
from dl.screener import YaoguScreener
from dl.screener_dataset import ScreenerDataset

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/models")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_screener(
    daily_cache,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    *,
    recall_target: float = 0.95,
    C: float = 1.0,
    max_neg_per_pos: int = 10,
    save_path: str | None = None,
) -> tuple[YaoguScreener, dict]:
    """Train Stage 1 screener.

    Returns (fitted_screener, metrics_dict).
    """
    logger.info("=" * 50)
    logger.info("Training Stage 1 Screener (Logistic Regression)")
    logger.info("  Train: %s ~ %s", train_start, train_end)
    logger.info("  Val:   %s ~ %s", val_start, val_end)
    logger.info("  Recall target: %.0f%%", recall_target * 100)
    logger.info("=" * 50)

    # Build feature cache once (shared by train and val)
    feature_cache = build_normalized_feature_cache(daily_cache)
    logger.info("Feature cache built: %d rows", len(feature_cache))

    # ── Build training data ──
    train_ds = ScreenerDataset(
        daily_cache, symbols,
        start_date=train_start, end_date=train_end,
        feature_cache=feature_cache,
        max_neg_per_pos=max_neg_per_pos,
    )
    X_train, y_train = train_ds.get_data()
    feature_cols = train_ds.feature_cols
    logger.info("Train: %d samples, %.2f%% positive, %d features",
                 len(y_train), y_train.mean() * 100, len(feature_cols))

    if len(y_train) < 200:
        logger.warning("Too few training samples (%d). Expand date range.", len(y_train))

    # ── Train screener ──
    screener = YaoguScreener(recall_target=recall_target, C=C)
    screener.fit(X_train, y_train, feature_cols=feature_cols)

    # ── Validate ──
    val_ds = ScreenerDataset(
        daily_cache, symbols,
        start_date=val_start, end_date=val_end,
        feature_cache=feature_cache,
        max_neg_per_pos=None,  # use all negatives for validation
    )
    X_val, y_val = val_ds.get_data()

    if len(y_val) > 0:
        val_probs = screener.predict_proba(X_val)
        val_preds = val_probs >= screener.threshold

        tp = ((val_preds == 1) & (y_val == 1)).sum()
        fn = ((val_preds == 0) & (y_val == 1)).sum()
        fp = ((val_preds == 1) & (y_val == 0)).sum()

        val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        candidate_rate = (tp + fp) / len(y_val)
        total_positives = tp + fn

        logger.info("Validation (n=%d):", len(y_val))
        logger.info("  Recall: %.3f (target >= %.3f) %s",
                     val_recall, recall_target,
                     "✓" if val_recall >= recall_target * 0.95 else "✗")
        logger.info("  Precision: %.3f", val_precision)
        logger.info("  Candidate rate: %.1f%% (pool reduction: %.1fx)",
                     candidate_rate * 100,
                     1 / candidate_rate if candidate_rate > 0 else float("inf"))
        logger.info("  TP=%d, FN=%d, FP=%d (total pos=%d)", tp, fn, fp, total_positives)
    else:
        val_recall = 0
        val_precision = 0
        candidate_rate = 1.0
        logger.warning("Empty validation set")

    # ── Feature importance ──
    logger.info("Top screener features (by |coef|):")
    coefs = screener.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[-10:][::-1]
    for i in top_idx:
        logger.info("  %-25s %+.4f", feature_cols[i] if i < len(feature_cols) else f"f{i}", coefs[i])

    # ── Save ──
    path = save_path or str(CHECKPOINT_DIR / "yaogu_screener.joblib")
    screener.save(path)

    metrics = {
        "val_recall": val_recall,
        "val_precision": val_precision,
        "candidate_rate": candidate_rate,
        "threshold": screener.threshold,
        "recall_target": recall_target,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }

    return screener, metrics
