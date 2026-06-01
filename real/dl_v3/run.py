"""Entry point for V3 training: pre-launch yaogu detection.

Usage:
    python -m dl_v3.run --train-start 2019-01-01 --train-end 2023-12-31 \
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
from dl_v3.train import train_v3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    p = argparse.ArgumentParser(description="V3 Pre-Launch Yaogu Detection Training")

    # Data
    p.add_argument("--train-start", type=str, default="2019-01-01")
    p.add_argument("--train-end", type=str, default="2023-12-31")
    p.add_argument("--val-start", type=str, default="2024-01-01")
    p.add_argument("--val-end", type=str, default="2024-12-31")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--sequence-length", type=int, default=20)

    # V3 Labels
    p.add_argument("--forward-start", type=int, default=1,
                   help="Forward window start (T+N) for label (default 1)")
    p.add_argument("--forward-end", type=int, default=15,
                   help="Forward window end for label (default 15)")
    p.add_argument("--min-cum-ret", type=float, default=0.30,
                   help="Minimum forward return for positive label")
    p.add_argument("--max-pre-ret", type=float, default=0.10,
                   help="Maximum pre-20d return to be 'quiet'")

    # Loss
    p.add_argument("--ap-tau", type=float, default=0.01)
    p.add_argument("--focal-weight", type=float, default=0.1)
    p.add_argument("--focal-alpha", type=float, default=0.85)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--no-weighted-sampler", action="store_true")

    # Regularization
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--head-dropout", type=float, default=0.5,
                   help="Dropout rate in MLP head (default 0.5)")
    p.add_argument("--tower-dropout", type=float, default=0.3,
                   help="Dropout rate in CNN/Transformer towers (default 0.3)")

    # I/O
    p.add_argument("--save", type=str, default="data/models/yaogu_v3_best.pt")
    p.add_argument("--device", type=str, default="auto")

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
    load_start = train_start - timedelta(days=args.sequence_length * 5 + 200)  # history
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
    logger.info("Running V3 training (Pre-Launch Detection)...")
    model_kwargs = {
        "head_dropout": args.head_dropout,
        "cnn_dropout": args.tower_dropout,
        "trans_dropout": args.tower_dropout,
    }
    model, metrics = train_v3(
        sim._daily_cache, symbols,
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
        sequence_length=args.sequence_length,
        forward_start=args.forward_start,
        forward_end=args.forward_end,
        min_cum_ret=args.min_cum_ret,
        max_pre_ret=args.max_pre_ret,
        ap_tau=args.ap_tau,
        focal_weight=args.focal_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_weighted_sampler=not args.no_weighted_sampler,
        weight_decay=args.weight_decay,
        model_kwargs=model_kwargs,
        save_path=args.save, device=device,
    )

    # ── Summary ──
    print("\n" + "=" * 50)
    print("V3 Training Summary")
    print("=" * 50)
    print(f"Label: fwd [T+{args.forward_start},T+{args.forward_end}] >= {args.min_cum_ret:.0%}")
    print(f"       AND pre 20d < {args.max_pre_ret:.0%}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best val AP: {metrics['best_val_ap']:.4f}")
    print(f"Train pos rate: {metrics['train_positive_rate']*100:.2f}%")
    print(f"Val pos rate: {metrics['val_positive_rate']*100:.2f}%")
    print(f"Model saved to: {args.save}")


if __name__ == "__main__":
    main()
