"""Entry point for training the DL yaogu detection model.

Usage:
    python -m dl.run --train-start 2020-01-01 --train-end 2023-12-31 \
                     --val-start 2024-01-01 --val-end 2024-12-31 \
                     --epochs 100 --batch-size 2048
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_agent_simulation import AgentSimulation, get_trading_days
from data.cache import read_daily
from dl.train import train, train_two_stage, train_v2
from dl.train_screener import train_screener

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Train DL Yaogu Detection Model")
    p.add_argument("--train-start", type=str, default="2020-01-01")
    p.add_argument("--train-end", type=str, default="2023-12-31")
    p.add_argument("--val-start", type=str, default="2024-01-01")
    p.add_argument("--val-end", type=str, default="2024-12-31")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--target-precision", type=float, default=0.3)
    p.add_argument("--save", type=str, default="data/models/yaogu_best.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mode", type=str, default="single",
                   choices=["single", "two-stage", "screener-only", "v2"],
                   help="Training mode: single (DL only), two-stage (screener+DL), "
                        "screener-only (LR only), v2 (SmoothAP + adversarial)")
    p.add_argument("--recall-target", type=float, default=0.95,
                   help="Stage 1 screener recall target")
    p.add_argument("--screener-save", type=str, default="data/models/yaogu_screener.joblib")
    p.add_argument("--no-resume", action="store_true",
                   help="Force retrain even if checkpoints exist")
    p.add_argument("--raw-ohlcv", action="store_true",
                   help="Use raw OHLCV features (returns + log volume) instead of derived factors")
    # V2 args
    p.add_argument("--sequence-length", type=int, default=20,
                   help="Input sequence length in trading days (v2 mode)")
    p.add_argument("--ap-tau", type=float, default=0.01,
                   help="SmoothAP temperature (v2 mode)")
    p.add_argument("--focal-weight", type=float, default=0.1,
                   help="Auxiliary focal loss weight (v2 mode)")
    p.add_argument("--lambda-grl", type=float, default=0.0,
                   help="GRL lambda max, 0 = no adversarial training (v2 mode)")
    p.add_argument("--grl-gamma", type=float, default=10.0,
                   help="GRL schedule steepness (v2 mode)")
    p.add_argument("--no-weighted-sampler", action="store_true",
                   help="Disable weighted sampler (v2 mode)")
    p.add_argument("--test-start", type=str, default=None,
                   help="Test period start for final evaluation")
    p.add_argument("--test-end", type=str, default=None,
                   help="Test period end for final evaluation")
    args = p.parse_args()

    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end)
    val_start = date.fromisoformat(args.val_start)
    val_end = date.fromisoformat(args.val_end)

    # Device
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # ── Load data ──
    logger.info("Loading data...")
    seq_len = getattr(args, 'sequence_length', 60)
    load_start = train_start - timedelta(days=seq_len * 5 + 100)  # history for sequences
    load_end = val_end

    sim = AgentSimulation(start=train_start, end=val_end, mode="factor")
    sim._trading_days = get_trading_days(sim.start, sim.end)
    sim._daily_cache = read_daily(load_start, load_end)

    all_syms = sorted(sim._daily_cache.index.get_level_values("symbol").unique().tolist())
    symbols = sim._generate_stock_pool(all_syms)
    pool_mask = sim._daily_cache.index.get_level_values("symbol").isin(symbols)
    sim._daily_cache = sim._daily_cache[pool_mask]

    logger.info("Symbols: %d, rows: %d", len(symbols), len(sim._daily_cache))

    # ── Train ──
    if args.mode == "screener-only":
        logger.info("Training Stage 1 screener only...")
        screener, metrics = train_screener(
            sim._daily_cache, symbols,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            recall_target=args.recall_target,
            save_path=args.screener_save,
        )
        # ── Summary ──
        print("\n" + "=" * 50)
        print("Stage 1 Screener Summary")
        print("=" * 50)
        print(f"Val Recall: {metrics['val_recall']:.3f} (target >= {args.recall_target})")
        print(f"Val Precision: {metrics['val_precision']:.3f}")
        print(f"Candidate rate: {metrics['candidate_rate']*100:.1f}%")
        print(f"Threshold: {metrics['threshold']:.4f}")
        print(f"Saved to: {args.screener_save}")
        return

    elif args.mode == "two-stage":
        logger.info("Running two-stage training...")
        screener, model, metrics = train_two_stage(
            sim._daily_cache, symbols,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            recall_target=args.recall_target,
            save_screener_path=args.screener_save,
            save_dl_path=args.save,
            device=device,
            resume=not args.no_resume,
            raw_ohlcv=args.raw_ohlcv,
            dl_kwargs=dict(
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                patience=args.patience, target_precision=args.target_precision,
            ),
        )
        # ── Summary ──
        print("\n" + "=" * 50)
        print("Two-Stage Training Summary")
        print("=" * 50)
        s1 = metrics["stage1"]
        s2 = metrics["stage2"]
        print(f"Stage 1 — Recall: {s1['recall']:.3f}, Candidate rate: {s1['candidate_rate']*100:.1f}%")
        print(f"Stage 2 — Best epoch: {s2['best_epoch']}, "
              f"Precision: {s2['best_val_precision']:.4f}")
        print(f"Stage 2 — Precision: {s2['metrics']['precision']:.3f}, "
              f"Recall: {s2['metrics']['recall']:.3f}")
        print(f"Stage 2 — TP:{s2['metrics']['tp']}, FP:{s2['metrics']['fp']}, "
              f"FN:{s2['metrics']['fn']}")
        print(f"Saved: {args.screener_save} + {args.save}")
        return

    # ── V2: SmoothAP + Adversarial ──
    if args.mode == "v2":
        logger.info("Running V2 training (SmoothAP + Adversarial)...")
        model, metrics = train_v2(
            sim._daily_cache, symbols,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
            sequence_length=args.sequence_length,
            ap_tau=args.ap_tau,
            focal_weight=args.focal_weight,
            lambda_grl=args.lambda_grl,
            grl_gamma=args.grl_gamma,
            use_weighted_sampler=not args.no_weighted_sampler,
            save_path=args.save, device=device,
            raw_ohlcv=True,
        )
        print("\n" + "=" * 50)
        print("V2 Training Summary")
        print("=" * 50)
        print(f"Best epoch: {metrics['best_epoch']}")
        print(f"Best val AP: {metrics['best_val_ap']:.4f}")
        print(f"Train pos rate: {metrics['train_positive_rate']*100:.2f}%")
        print(f"Val pos rate: {metrics['val_positive_rate']*100:.2f}%")
        print(f"Model saved to: {args.save}")
        return

    # ── Single-stage (original) ──
    model, metrics = train(
        sim._daily_cache, symbols,
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
        target_precision=args.target_precision,
        save_path=args.save, device=device,
    )

    # ── Summary ──
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best val F1: {metrics['best_val_f1']:.4f}")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print(f"Precision: {metrics['metrics']['precision']:.3f}")
    print(f"Recall: {metrics['metrics']['recall']:.3f}")
    print(f"TP: {metrics['metrics']['tp']}, FP: {metrics['metrics']['fp']}, FN: {metrics['metrics']['fn']}")
    print(f"Train pos rate: {metrics['train_positive_rate']*100:.2f}%")
    print(f"Val pos rate: {metrics['val_positive_rate']*100:.2f}%")
    print(f"Model saved to: {args.save}")


if __name__ == "__main__":
    main()
