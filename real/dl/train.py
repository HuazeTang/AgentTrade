"""Training pipeline for DualTowerModel with Focal Loss.

Optimized for precision on extreme class-imbalanced yaogu detection.
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

from dl import DualTowerModel, FocalLoss, find_precision_threshold, count_parameters
from dl.dataset import YaoguDataset, build_dataloaders
from dl.derived_features import (
    build_normalized_feature_cache,
    DERIVED_FEATURE_COLUMNS,
    RAW_OHLCV_COLUMNS,
)
from dl.screener_dataset import ScreenerDataset
from dl.screener import YaoguScreener

logger = logging.getLogger(__name__)

# ── Defaults ──
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 2048
DEFAULT_PATIENCE = 20
DEFAULT_TARGET_PRECISION = 0.3
CHECKPOINT_DIR = Path("data/models")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train(
    daily_cache,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_tp: int = 5,
    model_kwargs: dict | None = None,
    save_path: str | None = None,
    device: str = "cpu",
    feature_cache = None,
    candidate_symbols: set | None = None,
    raw_ohlcv: bool = False,
) -> tuple[DualTowerModel, dict]:
    """Train a DualTowerModel for yaogu detection.

    Returns (trained_model, training_metrics).
    """
    feature_cols = RAW_OHLCV_COLUMNS if raw_ohlcv else DERIVED_FEATURE_COLUMNS

    logger.info("=" * 50)
    logger.info("Training DualTowerModel for Yaogu Detection")
    logger.info("  Train: %s ~ %s", train_start, train_end)
    logger.info("  Val:   %s ~ %s", val_start, val_end)
    logger.info("  Features: %s (%d cols)", "raw_ohlcv" if raw_ohlcv else "derived", len(feature_cols))
    logger.info("=" * 50)

    # Build datasets
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        daily_cache, symbols, train_start, train_end, val_start, val_end,
        batch_size=batch_size, max_neg_per_pos=20,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        feature_cols=feature_cols,
    )

    logger.info("Train: %d samples (%.2f%% positive)", len(train_ds), train_ds.positive_rate * 100)
    logger.info("Val:   %d samples (%.2f%% positive)", len(val_ds), val_ds.positive_rate * 100)

    if len(train_ds) < 1000:
        logger.warning("Too few training samples. Consider expanding date range.")

    # Model — infer in_features from dataset if not specified
    model_kwargs = model_kwargs or {}
    model_kwargs.setdefault("in_features", train_ds.n_features)
    model = DualTowerModel(**model_kwargs)
    model.to(device)

    n_params = count_parameters(model)
    logger.info("Model: %.1fK parameters", n_params / 1000)

    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # Training loop
    best_val_precision = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_precision": []}
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = Path(save_path).parent if save_path else CHECKPOINT_DIR
    logger.info("Run ID: %s", run_id)

    # ── MPS warmup ──
    logger.info("Warming up MPS (first forward pass compiles compute graph)...")
    warmup_batch = next(iter(train_loader))
    model.train()
    _ = model(warmup_batch[0].to(device))
    logger.info("Warmup complete, starting training.")

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # ── Validate ──
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        # ── Precision-optimized threshold on val ──
        threshold, metrics = find_precision_threshold(
            model, val_loader, target_precision=target_precision, device=device,
            min_tp=min_tp,
        )

        val_f1 = metrics["f1"]
        val_precision = metrics["precision"]
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)

        scheduler.step(val_precision)

        # ── Logging (every epoch) ──
        if True:
            elapsed = time.perf_counter() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d | loss: %.4f/%.4f | prec: %.3f recall: %.3f f1: %.3f | "
                "thresh: %.2f tp:%d fp:%d | lr: %.1e | %.0fs",
                epoch, avg_train_loss, avg_val_loss,
                val_precision, metrics["recall"], val_f1,
                threshold, metrics["tp"], metrics["fp"],
                lr_now, elapsed,
            )

        # ── Save every epoch checkpoint ──
        ep_path = ckpt_dir / f"yaogu_{run_id}_ep{epoch}.pt"
        checkpoint = {
            "epoch": epoch,
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "threshold": threshold,
            "metrics": metrics,
            "history": history,
            "model_kwargs": model_kwargs,
            "feature_cols": feature_cols,
            "raw_ohlcv": raw_ohlcv,
        }
        torch.save(checkpoint, ep_path)

        # ── Best model tracking ──
        if val_precision > best_val_precision:
            best_val_precision = val_precision
            best_epoch = epoch
            no_improve = 0
            torch.save(checkpoint, ckpt_dir / f"yaogu_{run_id}_best.pt")
            logger.info("  → Best (prec=%.4f, tp=%d, fp=%d, thresh=%.2f) → %s",
                        val_precision, metrics["tp"], metrics["fp"],
                        threshold, f"yaogu_{run_id}_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, patience)
                break

    # ── Final evaluation ──
    elapsed = time.perf_counter() - t0
    logger.info("Training complete: best epoch=%d, best precision=%.4f, %.0fs total",
                best_epoch, best_val_precision, elapsed)

    # Load best checkpoint
    best_path = ckpt_dir / f"yaogu_{run_id}_best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Final threshold
    final_threshold, final_metrics = find_precision_threshold(
        model, val_loader, target_precision=target_precision, device=device
    )
    logger.info("Final threshold=%.2f, precision=%.3f, recall=%.3f",
                final_threshold, final_metrics["precision"], final_metrics["recall"])

    training_metrics = {
        "best_epoch": best_epoch,
        "best_val_precision": best_val_precision,
        "threshold": final_threshold,
        "metrics": final_metrics,
        "history": history,
        "n_params": n_params,
        "train_positive_rate": train_ds.positive_rate,
        "val_positive_rate": val_ds.positive_rate,
    }

    return model, training_metrics


def train_two_stage(
    daily_cache,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    *,
    screener_kwargs: dict | None = None,
    dl_kwargs: dict | None = None,
    recall_target: float = 0.95,
    save_screener_path: str | None = None,
    save_dl_path: str | None = None,
    device: str = "cpu",
    resume: bool = True,
    raw_ohlcv: bool = False,
) -> tuple[YaoguScreener, DualTowerModel, dict]:
    """Two-stage training: screener (Stage 1) → DL model on candidates (Stage 2).

    Stage 1: Train logistic regression on single-day features for high recall.
    Stage 2: Train DualTowerModel only on screener-filtered candidates for high precision.

    Returns (screener, dl_model, combined_metrics).
    """
    feature_cols = RAW_OHLCV_COLUMNS if raw_ohlcv else DERIVED_FEATURE_COLUMNS

    logger.info("=" * 60)
    logger.info("Two-Stage Training: Stage 1 (LR screener) → Stage 2 (DL model)")
    logger.info("  Train: %s ~ %s", train_start, train_end)
    logger.info("  Val:   %s ~ %s", val_start, val_end)
    logger.info("  Features: %s (%d cols)", "raw_ohlcv" if raw_ohlcv else "derived", len(feature_cols))
    logger.info("=" * 60)

    # Shared feature cache
    feature_cache = build_normalized_feature_cache(daily_cache, raw_ohlcv=raw_ohlcv)
    logger.info("Shared feature cache: %d rows, %d columns",
                 len(feature_cache), len(feature_cache.columns))

    # ═══════ Stage 1: Train / load screener ═══════
    screener_path = Path(save_screener_path or str(CHECKPOINT_DIR / "yaogu_screener.joblib"))

    if resume and screener_path.exists():
        logger.info("--- Stage 1: Loading existing screener from %s ---", screener_path)
        screener = YaoguScreener.load(screener_path)
    else:
        logger.info("--- Stage 1: Training Screener ---")
        train_ds_s = ScreenerDataset(
            daily_cache, symbols,
            start_date=train_start, end_date=train_end,
            feature_cache=feature_cache,
            feature_cols=feature_cols,
            max_neg_per_pos=10,
        )
        X_train, y_train = train_ds_s.get_data()
        feature_cols = train_ds_s.feature_cols

        screener = YaoguScreener(
            recall_target=recall_target,
            **(screener_kwargs or {}),
        )
        screener.fit(X_train, y_train, feature_cols=feature_cols)
        # Save immediately after fit (crash protection)
        screener.save(screener_path)
        logger.info("  Screener saved to %s", screener_path)

    # Validate screener
    val_ds_s = ScreenerDataset(
        daily_cache, symbols,
        start_date=val_start, end_date=val_end,
        feature_cache=feature_cache,
        feature_cols=feature_cols,
        max_neg_per_pos=None,
    )
    X_val, y_val = val_ds_s.get_data()
    val_probs = screener.predict_proba(X_val)
    val_preds = val_probs >= screener.threshold

    tp = ((val_preds == 1) & (y_val == 1)).sum()
    fn = ((val_preds == 0) & (y_val == 1)).sum()
    fp = ((val_preds == 1) & (y_val == 0)).sum()
    stage1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    stage1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    candidate_rate = (tp + fp) / len(y_val)

    logger.info("Stage 1 validation:")
    logger.info("  Recall: %.3f | Precision: %.3f | Candidate rate: %.1f%%",
                 stage1_recall, stage1_precision, candidate_rate * 100)

    # Build candidate symbol set from screener predictions
    candidate_symbols = set()
    day_data_indices = val_ds_s._meta
    for i, meta in enumerate(day_data_indices):
        if val_preds[i]:
            candidate_symbols.add(meta["symbol"])
    logger.info("  Candidate symbol count: %d", len(candidate_symbols))

    # ═══════ Stage 2: Train DL model on candidates ═══════
    logger.info("--- Stage 2: Training DL Model on Candidates ---")
    dl_kwargs = dl_kwargs or {}

    model, dl_metrics = train(
        daily_cache, symbols,
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        raw_ohlcv=raw_ohlcv,
        epochs=dl_kwargs.get("epochs", 100),
        lr=dl_kwargs.get("lr", 1e-3),
        batch_size=dl_kwargs.get("batch_size", 2048),
        patience=dl_kwargs.get("patience", 20),
        target_precision=dl_kwargs.get("target_precision", 0.3),
        min_tp=dl_kwargs.get("min_tp", 5),
        model_kwargs=dl_kwargs.get("model_kwargs"),
        save_path=save_dl_path,
        device=device,
    )

    # Screener already saved after fit; ensure it exists at the target path
    if not screener_path.exists():
        screener.save(screener_path)

    # Combined metrics
    combined = {
        "stage1": {
            "recall": stage1_recall,
            "precision": stage1_precision,
            "candidate_rate": candidate_rate,
            "threshold": screener.threshold,
            "n_candidates": len(candidate_symbols),
        },
        "stage2": dl_metrics,
    }

    logger.info("=" * 60)
    logger.info("Two-Stage Training Complete")
    logger.info("  Stage 1: recall=%.3f, candidate_rate=%.1f%%",
                 stage1_recall, candidate_rate * 100)
    logger.info("  Stage 2: precision=%.3f, f1=%.4f, threshold=%.2f",
                 dl_metrics["metrics"]["precision"], dl_metrics["metrics"]["f1"],
                 dl_metrics["threshold"])
    logger.info("=" * 60)

    return screener, model, combined


# ═══════════════════════════════════════════════════════════════════════════
# V2 Training: SmoothAP + Adversarial + Balanced Sampling
# ═══════════════════════════════════════════════════════════════════════════

def train_v2(
    daily_cache,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    *,
    # Core hyperparams
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 2048,
    patience: int = 20,
    sequence_length: int = 20,
    # Loss configuration
    ap_tau: float = 0.01,
    focal_alpha: float = 0.85,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.1,
    # Adversarial training
    lambda_grl: float = 0.0,       # 0 = no adversarial training
    grl_gamma: float = 10.0,
    # Sampling
    use_weighted_sampler: bool = True,
    # Model
    model_kwargs: dict | None = None,
    # I/O
    save_path: str | None = None,
    device: str = "cpu",
    feature_cache = None,
    candidate_symbols: set | None = None,
    raw_ohlcv: bool = True,
) -> tuple[DualTowerModel, dict]:
    """Train DualTowerModel with SmoothAP + optional adversarial + weighted sampling.

    Primary loss: SmoothAPLoss (optimizes ranking directly)
    Auxiliary:   FocalLoss (low weight, stabilizes training)
    Adversarial: CrossEntropy on year classifier (forces year-invariant features)

    Checkpoint selection: best val AP (not precision).

    Returns (trained_model, training_metrics).
    """
    from dl import (
        SmoothAPLoss, compute_average_precision, AdversarialDualTowerModel,
        compute_grl_lambda, count_parameters,
    )
    from dl.dataset import compute_sample_weights

    feature_cols = RAW_OHLCV_COLUMNS if raw_ohlcv else DERIVED_FEATURE_COLUMNS
    use_adversarial = lambda_grl > 0

    logger.info("=" * 50)
    logger.info("Training V2: SmoothAP + %sAdversarial + Weighted Sampler",
                 "" if use_adversarial else "No ")
    logger.info("  Train: %s ~ %s", train_start, train_end)
    logger.info("  Val:   %s ~ %s", val_start, val_end)
    logger.info("  Seq len: %d | Features: %s (%d cols)",
                 sequence_length, "raw_ohlcv" if raw_ohlcv else "derived", len(feature_cols))
    logger.info("=" * 50)

    # Build datasets
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        daily_cache, symbols, train_start, train_end, val_start, val_end,
        batch_size=batch_size, max_neg_per_pos=20,
        feature_cache=feature_cache,
        candidate_symbols=candidate_symbols,
        use_weighted_sampler=use_weighted_sampler,
        return_year=use_adversarial,  # only need year labels for adversarial
        raw_ohlcv=raw_ohlcv,
        sequence_length=sequence_length,
        feature_cols=feature_cols,
    )

    logger.info("Train: %d samples (%.2f%% positive, %d years)",
                 len(train_ds), train_ds.positive_rate * 100, train_ds.n_years)
    logger.info("Val:   %d samples (%.2f%% positive, %d years)",
                 len(val_ds), val_ds.positive_rate * 100, val_ds.n_years)

    # Model
    model_kwargs = model_kwargs or {}
    model_kwargs.setdefault("in_features", train_ds.n_features)
    base_model = DualTowerModel(**model_kwargs)

    if use_adversarial:
        model = AdversarialDualTowerModel(
            base_model=base_model,
            n_years=train_ds.n_years,
            grl_lambda=0.0,
        )
    else:
        model = base_model

    model.to(device)
    n_params = count_parameters(model)
    logger.info("Model: %.1fK parameters (adversarial=%s)", n_params / 1000, use_adversarial)

    # Loss functions
    ap_criterion = SmoothAPLoss(tau=ap_tau)
    focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    ce_criterion = nn.CrossEntropyLoss() if use_adversarial else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # MPS warmup
    logger.info("Warming up MPS...")
    warmup_batch = next(iter(train_loader))
    model.train()
    if use_adversarial:
        _ = model(warmup_batch[0].to(device))
    else:
        _ = model(warmup_batch[0].to(device))
    logger.info("Warmup complete, starting training.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = Path(save_path).parent if save_path else CHECKPOINT_DIR
    logger.info("Run ID: %s", run_id)

    best_val_ap = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_ap": [], "train_loss": [], "val_ap": [], "val_loss": []}

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # Update GRL lambda on schedule
        if use_adversarial:
            grl_val = compute_grl_lambda(epoch, epochs, lambda_grl, grl_gamma)
            model.set_grl_lambda(grl_val)
        else:
            grl_val = 0.0

        # ── Train ──
        model.train()
        epoch_ap_losses, epoch_focal_losses, epoch_yr_losses = [], [], []

        for batch in train_loader:
            if use_adversarial:
                x, y, yr = batch
                x, y, yr = x.to(device), y.to(device), yr.to(device)
                yaogu_logits, year_logits = model(x)

                ap_loss = ap_criterion(yaogu_logits, y)
                focal_loss = focal_criterion(yaogu_logits, y)
                yr_loss = ce_criterion(year_logits, yr)
                loss = ap_loss + focal_weight * focal_loss + grl_val * yr_loss
            else:
                x, y, *_ = batch
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
            if use_adversarial:
                epoch_yr_losses.append(yr_loss.item())

        avg_ap = np.mean(epoch_ap_losses)
        avg_focal = np.mean(epoch_focal_losses)
        avg_yr = np.mean(epoch_yr_losses) if epoch_yr_losses else 0.0
        avg_train_loss = avg_ap + focal_weight * avg_focal + grl_val * avg_yr

        # ── Validate ──
        model.eval()
        all_logits, all_labels = [], []
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                if use_adversarial:
                    x, y, _ = batch
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                else:
                    x, y, *_ = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)

                val_loss = ap_criterion(logits, y) + focal_weight * focal_criterion(logits, y)
                val_losses.append(val_loss.item())
                all_logits.append(logits.cpu())
                all_labels.append(y)

        val_ap = compute_average_precision(torch.cat(all_logits), torch.cat(all_labels))
        avg_val_loss = np.mean(val_losses)

        history["train_ap"].append(1.0 - avg_ap)  # store AP not loss
        history["train_loss"].append(avg_train_loss)
        history["val_ap"].append(val_ap)
        history["val_loss"].append(avg_val_loss)

        scheduler.step(val_ap)

        # ── Logging ──
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d | AP: %.4f/%.4f | loss: %.4f/%.4f | "
            "focal: %.4f yr: %.4f grl: %.2f | lr: %.1e | %.0fs",
            epoch, 1.0 - avg_ap, val_ap, avg_train_loss, avg_val_loss,
            avg_focal, avg_yr, grl_val, lr_now, elapsed,
        )

        # ── Save checkpoint ──
        ep_path = ckpt_dir / f"yaogu_{run_id}_ep{epoch}.pt"
        base_state = model.base.state_dict() if use_adversarial else model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "run_id": run_id,
            "model_state_dict": base_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ap": val_ap,
            "history": history,
            "model_kwargs": model_kwargs,
            "feature_cols": feature_cols,
            "raw_ohlcv": raw_ohlcv,
            "sequence_length": sequence_length,
        }
        torch.save(checkpoint, ep_path)

        # ── Best model tracking (on val AP) ──
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            no_improve = 0
            torch.save(checkpoint, ckpt_dir / f"yaogu_{run_id}_best.pt")
            logger.info("  → Best (val_ap=%.4f) → %s", val_ap, f"yaogu_{run_id}_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no AP improvement for %d epochs)",
                            epoch, patience)
                break

    # ── Final evaluation ──
    elapsed = time.perf_counter() - t0
    logger.info("Training complete: best epoch=%d, best AP=%.4f, %.0fs total",
                best_epoch, best_val_ap, elapsed)

    # Load best
    best_path = ckpt_dir / f"yaogu_{run_id}_best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        if use_adversarial:
            model.base.load_state_dict(ck["model_state_dict"])
        else:
            model.load_state_dict(ck["model_state_dict"])

    training_metrics = {
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "history": history,
        "n_params": n_params,
        "train_positive_rate": train_ds.positive_rate,
        "val_positive_rate": val_ds.positive_rate,
        "train_n_years": train_ds.n_years,
    }

    return model.base if use_adversarial else model, training_metrics
