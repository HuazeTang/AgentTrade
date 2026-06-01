"""Check if V4's per-stock z-score helps re-recommend gap-down shakeout stocks."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dl import DualTowerModel
from dl_v4.derived_features import build_v4_feature_cache
from data.cache import read_daily

SEQUENCE_LENGTH = 20
TOP_K = 20  # wider top-k to see if stock appears at all
CHECKPOINT_V3 = "data/models/yaogu_v3_20260530_0532_best.pt"
CHECKPOINT_V4 = "data/models/yaogu_v4_20260530_1738_best.pt"

# Gap-down victims from V3 backtest analysis
VICTIMS = [
    # (symbol, name, gap_down_date, post_sell_peak_ret)
    ("600330", "天通股份", "2026-03-10", 0.655),
    ("603803", "瑞斯康达", "2026-02-24", 0.172),
]

# Load data
load_start = date(2019, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading data...")
daily = read_daily(load_start, load_end, prefix="daily_badj")

logging.info("Building V4 feature cache...")
cache_v4 = build_v4_feature_cache(daily)
feature_cols_v4 = [c for c in cache_v4.columns if not c.startswith("_")]

dates_all = sorted(cache_v4.index.get_level_values("trade_date").unique())
symbols_all = sorted(cache_v4.index.get_level_values("symbol").unique())

# Build feature tensor V4
feat_mats_v4 = []
for col in feature_cols_v4:
    mat = cache_v4[col].unstack()
    mat = mat.reindex(index=dates_all, columns=symbols_all)
    feat_mats_v4.append(mat.values)
feat_tensor_v4 = np.stack(feat_mats_v4, axis=-1).astype(np.float32)
feat_tensor_v4 = np.nan_to_num(feat_tensor_v4, nan=0.0, posinf=0.0, neginf=0.0)

symbol_to_idx = {s: i for i, s in enumerate(symbols_all)}
n_features_v4 = len(feature_cols_v4)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load V4 model
ck_v4 = torch.load(CHECKPOINT_V4, map_location=device, weights_only=False)
model_v4 = DualTowerModel(in_features=n_features_v4)
model_v4.load_state_dict(ck_v4["model_state_dict"])
model_v4.to(device)
model_v4.eval()
logging.info("V4 model loaded (in_features=%d)", n_features_v4)

# For each victim, check V4 rank for 30 trading days after gap-down
for sym, name, gap_date, peak_ret in VICTIMS:
    sym_idx = symbol_to_idx.get(sym)
    if sym_idx is None:
        logging.warning("%s not in symbols", sym)
        continue

    gap_dt = pd.Timestamp(gap_date)
    try:
        gap_pos = dates_all.index(gap_dt)
    except ValueError:
        nearest = min(dates_all, key=lambda d: abs(d - gap_dt))
        gap_pos = dates_all.index(nearest)
        logging.info("Gap date %s → nearest %s", gap_date, nearest.date())

    close = daily["close"].unstack()
    sym_close = close[sym].values

    logging.info("\n=== %s %s | gap-down: %s | post-sell peak: +%.1f%% ===", sym, name, gap_date, peak_ret * 100)

    for offset in range(1, 61, 5):  # check every 5 days for 60 days
        idx = gap_pos + offset
        if idx >= len(dates_all) - 1:
            break
        if idx < SEQUENCE_LENGTH:
            continue

        td = dates_all[idx]
        if td < pd.Timestamp(gap_date):
            continue

        # V4 prediction
        seq = feat_tensor_v4[idx - SEQUENCE_LENGTH:idx, sym_idx, :]
        X = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            score_v4 = model_v4.predict_proba(X).item()

        # Rank V4
        seq_all = feat_tensor_v4[idx - SEQUENCE_LENGTH:idx, :, :]
        batch = np.transpose(seq_all, (1, 0, 2))
        X_all = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            all_scores_v4 = model_v4.predict_proba(X_all).cpu().numpy().flatten()

        rank_v4 = int((all_scores_v4 > score_v4).sum()) + 1
        total = len(all_scores_v4)

        px = sym_close[idx] if idx < len(sym_close) else np.nan
        gap_px = sym_close[gap_pos] if gap_pos < len(sym_close) else np.nan
        ret_from_gap = (px - gap_px) / gap_px if (not np.isnan(px) and not np.isnan(gap_px) and gap_px > 0) else 0

        in_top = "★ TOP3" if rank_v4 <= 3 else ("TOP10" if rank_v4 <= 10 else "")
        logging.info("  T+%2d  %s  score_v4=%.4f  rank=%d/%d  px=%.2f  ret_from_gap=%+.1f%%  %s",
                     offset, str(td)[:10], score_v4, rank_v4, total, px, ret_from_gap * 100, in_top)

logging.info("\nDone.")
