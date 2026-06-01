"""V3 Training: SmoothAP + FocalLoss with forward-skip labels.

Same architecture and loss structure as V2, but uses V3 dataset (pre-launch
labels, consolidation/divergence features, per-stock z-score).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dl import (
    DualTowerModel,
    FocalLoss,
    SmoothAPLoss,
    compute_average_precision,
    count_parameters,
)
from dl_v3.dataset import YaoguDatasetV3, build_dataloaders_v3
from dl_v3.derived_features import ALL_V3_COLUMNS, build_v3_feature_cache

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/models")


def train_v3(
    daily_cache,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    *,
    # Core hyperparams
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 1024,
    patience: int = 15,
    sequence_length: int = 20,
    # Label config
    forward_start: int = 1,
    forward_end: int = 15,
    min_cum_ret: float = 0.30,
    max_pre_ret: float = 0.10,
    # Loss configuration
    ap_tau: float = 0.01,
    focal_alpha: float = 0.85,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.1,
    # Sampling
    use_weighted_sampler: bool = True,
    # Model
    model_kwargs: dict | None = None,
    weight_decay: float = 1e-3,
    # I/O
    save_path: str | None = None,
    device: str = "cpu",
    feature_cache=None,
    candidate_symbols: set | None = None,
) -> tuple[DualTowerModel, dict]:
    """Train DualTowerModel with V3 pre-launch labels and features.

    Primary loss: SmoothAPLoss (optimizes ranking)
    Auxiliary:   FocalLoss (stabilizes training)
    Checkpoint selection: best val_AP.

    Returns (trained_model, training_metrics).
    """
    feature_cols = [c for c in ALL_V3_COLUMNS]
    if feature_cache is not None:
        feature_cols = [c for c in feature_cols if c in feature_cache.columns]

    logger.info("=" * 50)
    logger.info("Training V3: Pre-Launch Detection")
    logger.info("  Label: fwd [T+%d,T+%d] >= %.0f%% AND pre 20d < %.0f%%",
                 forward_start, forward_end, min_cum_ret * 100, max_pre_ret * 100)
    logger.info("  Train: %s ~ %s", train_start, train_end)
    logger.info("  Val:   %s ~ %s", val_start, val_end)
    logger.info("  Seq len: %d | Features: V3 (%d cols)",
                 sequence_length, len(feature_cols))
    logger.info("=" * 50)

    # Build shared feature cache if not provided
    if feature_cache is None:
        logger.info("Building shared V3 feature cache...")
        feature_cache = build_v3_feature_cache(daily_cache)
        feature_cols = [c for c in ALL_V3_COLUMNS if c in feature_cache.columns]

    # Build datasets
    train_ds, val_ds, train_loader, val_loader = build_dataloaders_v3(
        daily_cache, symbols, train_start, train_end, val_start, val_end,
        batch_size=batch_size, max_neg_per_pos=20,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        use_weighted_sampler=use_weighted_sampler,
        sequence_length=sequence_length,
        forward_start=forward_start,
        forward_end=forward_end,
        min_cum_ret=min_cum_ret,
        max_pre_ret=max_pre_ret,
        feature_cols=feature_cols,
    )

    logger.info("Train: %d samples (%.2f%% positive)",
                 len(train_ds), train_ds.positive_rate * 100)
    logger.info("Val:   %d samples (%.2f%% positive)",
                 len(val_ds), val_ds.positive_rate * 100)

    if train_ds.positive_rate < 0.005:
        logger.warning("Positive rate very low (< 0.5%%). Consider lowering min_cum_ret.")

    # Model (V3 defaults: higher dropout + weight decay to fight overfitting)
    model_kwargs = dict(model_kwargs) if model_kwargs else {}
    model_kwargs.setdefault("in_features", train_ds.n_features)
    model_kwargs.setdefault("cnn_dropout", 0.3)
    model_kwargs.setdefault("trans_dropout", 0.3)
    model_kwargs.setdefault("head_dropout", 0.5)
    model = DualTowerModel(**model_kwargs)
    model.to(device)

    n_params = count_parameters(model)
    logger.info("Model: %.1fK parameters", n_params / 1000)

    # Loss functions
    ap_criterion = SmoothAPLoss(tau=ap_tau)
    focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # MPS warmup
    logger.info("Warming up MPS...")
    warmup_batch = next(iter(train_loader))
    model.train()
    _ = model(warmup_batch[0].to(device))
    logger.info("Warmup complete, starting training.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = Path(save_path).parent if save_path else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s", run_id)

    best_val_ap = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_ap": [], "train_loss": [], "val_ap": [], "val_loss": []}

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_ap_losses, epoch_focal_losses = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            ap_loss = ap_criterion(logits, y)
            focal_loss = focal_criterion(logits, y)
            loss = ap_loss + focal_weight * focal_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_ap_losses.append(ap_loss.item())
            epoch_focal_losses.append(focal_loss.item())

        avg_ap = np.mean(epoch_ap_losses)
        avg_focal = np.mean(epoch_focal_losses)
        avg_train_loss = avg_ap + focal_weight * avg_focal

        # ── Validate ──
        model.eval()
        all_logits, all_labels = [], []
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss = ap_criterion(logits, y) + focal_weight * focal_criterion(logits, y)
                val_losses.append(val_loss.item())
                all_logits.append(logits.cpu())
                all_labels.append(y)

        val_ap = compute_average_precision(torch.cat(all_logits), torch.cat(all_labels))
        avg_val_loss = np.mean(val_losses)

        history["train_ap"].append(1.0 - avg_ap)
        history["train_loss"].append(avg_train_loss)
        history["val_ap"].append(val_ap)
        history["val_loss"].append(avg_val_loss)

        scheduler.step(val_ap)

        # ── Logging ──
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d | AP: %.4f/%.4f | loss: %.4f/%.4f | "
            "focal: %.4f | lr: %.1e | %.0fs",
            epoch, 1.0 - avg_ap, val_ap, avg_train_loss, avg_val_loss,
            avg_focal, lr_now, elapsed,
        )

        # ── Save checkpoint ──
        ep_path = ckpt_dir / f"yaogu_v3_{run_id}_ep{epoch}.pt"
        checkpoint = {
            "epoch": epoch,
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ap": val_ap,
            "history": history,
            "model_kwargs": model_kwargs,
            "feature_cols": feature_cols,
            "sequence_length": sequence_length,
            "forward_start": forward_start,
            "forward_end": forward_end,
            "min_cum_ret": min_cum_ret,
            "max_pre_ret": max_pre_ret,
        }
        torch.save(checkpoint, ep_path)

        # ── Best model tracking ──
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            no_improve = 0
            best_path = ckpt_dir / f"yaogu_v3_{run_id}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info("  → Best (val_ap=%.4f) → %s", val_ap, f"yaogu_v3_{run_id}_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no AP improvement for %d epochs)",
                            epoch, patience)
                break

    # ── Final ──
    elapsed = time.perf_counter() - t0
    logger.info("Training complete: best epoch=%d, best AP=%.4f, %.0fs total",
                best_epoch, best_val_ap, elapsed)

    best_path = ckpt_dir / f"yaogu_v3_{run_id}_best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    training_metrics = {
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "history": history,
        "n_params": n_params,
        "train_positive_rate": train_ds.positive_rate,
        "val_positive_rate": val_ds.positive_rate,
    }

    return model, training_metrics
