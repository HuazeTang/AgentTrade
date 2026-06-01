"""Compare V2 / V3 / V4 top recommendations on 2026-05-29."""
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
from dl.derived_features import build_normalized_feature_cache, RAW_OHLCV_COLUMNS, DERIVED_FEATURE_COLUMNS
from dl_v3.derived_features import build_v3_feature_cache
from dl_v4.derived_features import build_v4_feature_cache
from data.cache import read_daily

SEQUENCE_LENGTH = 20
FACTOR_SEQ_LENGTH = 60  # original factor model uses 60-day window
TOP_K = 15

CHECKPOINT_F0 = "data/models/yaogu_best.pt"
CHECKPOINT_V2 = "data/models/yaogu_20260529_0315_best.pt"
CHECKPOINT_V3 = "data/models/yaogu_v3_20260530_0532_best.pt"
CHECKPOINT_V4 = "data/models/yaogu_v4_20260530_1738_best.pt"

# Load data
load_start = date(2019, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading daily badj data...")
daily = read_daily(load_start, load_end, prefix="daily_badj")

# Fix: pre_close is NaN for May 7-28 (baostock data lost it). Fill from close.shift(1).
# The 05-29 tushare download is fine, but 05-01 through 05-28 were overwritten by baostock.
if "pre_close" in daily.columns:
    pre_nan = daily["pre_close"].isna().sum()
    if pre_nan > 0:
        logging.info("Fixing %d NaN pre_close values...", pre_nan)
        close = daily["close"].unstack()
        pc = close.shift(1)
        # Backfill NaN pre_close with close.shift(1)
        fixed = daily["pre_close"].copy()
        fixed_unstack = fixed.unstack()
        nan_mask = fixed_unstack.isna()
        fixed_unstack[nan_mask] = pc[nan_mask]
        daily = daily.drop(columns=["pre_close"])
        daily["pre_close"] = fixed_unstack.stack()
        logging.info("Fixed pre_close: %d NaN remaining", daily["pre_close"].isna().sum())

# Load stock info for ST filtering
info = pd.read_parquet("data/cache/stock_list.parquet")
st_mask = info['name'].str.contains(r'\*?ST', na=False)
st_symbols = set(info.loc[st_mask, 'symbol'].unique())
non_st = set(info.loc[~st_mask, 'symbol'].unique())

# ── V2 features ──
logging.info("Building V2 feature cache (8 raw OHLCV + CS z-score)...")
cache_v2 = build_normalized_feature_cache(daily, raw_ohlcv=True)
feature_cols_v2 = RAW_OHLCV_COLUMNS

# ── V3 features ──
logging.info("Building V3 feature cache (20 features)...")
cache_v3 = build_v3_feature_cache(daily, per_stock_zscore=False)
feature_cols_v3 = [c for c in cache_v3.columns if not c.startswith("_")]

# ── V4 features ──
logging.info("Building V4 feature cache (8 raw + 8 per-stock z-score)...")
cache_v4 = build_v4_feature_cache(daily)
feature_cols_v4 = [c for c in cache_v4.columns if not c.startswith("_")]

# ── F0 (original factor) features ──
logging.info("Building F0 feature cache (14 derived + CS z-score)...")
cache_f0 = build_normalized_feature_cache(daily, raw_ohlcv=False)
feature_cols_f0 = DERIVED_FEATURE_COLUMNS

dates_all = sorted(cache_v2.index.get_level_values("trade_date").unique())
symbols_v2 = sorted(cache_v2.index.get_level_values("symbol").unique())
symbols_v3 = sorted(cache_v3.index.get_level_values("symbol").unique())
symbols_v4 = sorted(cache_v4.index.get_level_values("symbol").unique())
symbols_f0 = sorted(cache_f0.index.get_level_values("symbol").unique())

logging.info("Dates: %d, Symbols V2: %d, V3: %d, V4: %d, F0: %d",
             len(dates_all), len(symbols_v2), len(symbols_v3), len(symbols_v4), len(symbols_f0))

# Build feature tensors
def build_tensor(cache, feature_cols, dates, symbols):
    feats = []
    for col in feature_cols:
        mat = cache[col].unstack()
        mat = mat.reindex(index=dates, columns=symbols)
        feats.append(mat.values)
    tensor = np.stack(feats, axis=-1).astype(np.float32)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor

feat_v2 = build_tensor(cache_v2, feature_cols_v2, dates_all, symbols_v2)
feat_v3 = build_tensor(cache_v3, feature_cols_v3, dates_all, symbols_v3)
feat_v4 = build_tensor(cache_v4, feature_cols_v4, dates_all, symbols_v4)
feat_f0 = build_tensor(cache_f0, feature_cols_f0, dates_all, symbols_f0)

sym_to_idx_v2 = {s: i for i, s in enumerate(symbols_v2)}
sym_to_idx_v3 = {s: i for i, s in enumerate(symbols_v3)}
sym_to_idx_v4 = {s: i for i, s in enumerate(symbols_v4)}
sym_to_idx_f0 = {s: i for i, s in enumerate(symbols_f0)}

device = "mps" if torch.backends.mps.is_available() else "cpu"
logging.info("Device: %s", device)

# Load models
ck_v2 = torch.load(CHECKPOINT_V2, map_location=device, weights_only=False)
model_v2 = DualTowerModel(in_features=len(feature_cols_v2))
model_v2.load_state_dict(ck_v2["model_state_dict"])
model_v2.to(device)
model_v2.eval()

ck_v3 = torch.load(CHECKPOINT_V3, map_location=device, weights_only=False)
model_v3 = DualTowerModel(in_features=len(feature_cols_v3))
model_v3.load_state_dict(ck_v3["model_state_dict"])
model_v3.to(device)
model_v3.eval()

ck_v4 = torch.load(CHECKPOINT_V4, map_location=device, weights_only=False)
model_v4 = DualTowerModel(in_features=len(feature_cols_v4))
model_v4.load_state_dict(ck_v4["model_state_dict"])
model_v4.to(device)
model_v4.eval()

# F0 model uses model_kwargs from checkpoint (14 in_features)
ck_f0 = torch.load(CHECKPOINT_F0, map_location=device, weights_only=False)
model_f0 = DualTowerModel(in_features=len(feature_cols_f0))
model_f0.load_state_dict(ck_f0["model_state_dict"])
model_f0.to(device)
model_f0.eval()

logging.info("Models loaded. F0: %d, V2: %d, V3: %d, V4: %d features",
             len(feature_cols_f0), len(feature_cols_v2), len(feature_cols_v3), len(feature_cols_v4))

# ── Run inference on last date ──
target_date = dates_all[-1]
target_idx = len(dates_all) - 1
logging.info("Inference date: %s (idx=%d)", target_date.date(), target_idx)

def run_inference(model, feat_tensor, sym_to_idx, symbols, feature_cols, label, seq_len=SEQUENCE_LENGTH):
    ranks = []
    seq_start = target_idx - seq_len
    valid_symbols = []

    for sym in symbols:
        idx = sym_to_idx.get(sym)
        if idx is None:
            continue
        seq = feat_tensor[seq_start:target_idx, idx, :]
        if np.isnan(seq).any() or np.abs(seq).max() > 100:
            continue
        valid_symbols.append((sym, idx, seq))

    logging.info("[%s] %d valid symbols for inference", label, len(valid_symbols))

    batch_seqs = np.stack([s for _, _, s in valid_symbols], axis=0)
    X = torch.from_numpy(batch_seqs).to(device)
    with torch.no_grad():
        scores = model.predict_proba(X).cpu().numpy().flatten()

    for (sym, _, _), score in zip(valid_symbols, scores):
        ranks.append((sym, float(score)))

    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks

ranks_f0 = run_inference(model_f0, feat_f0, sym_to_idx_f0, symbols_f0, feature_cols_f0, "F0", FACTOR_SEQ_LENGTH)
ranks_v2 = run_inference(model_v2, feat_v2, sym_to_idx_v2, symbols_v2, feature_cols_v2, "V2")
ranks_v3 = run_inference(model_v3, feat_v3, sym_to_idx_v3, symbols_v3, feature_cols_v3, "V3")
ranks_v4 = run_inference(model_v4, feat_v4, sym_to_idx_v4, symbols_v4, feature_cols_v4, "V4")

# ── Print results ──
def filter_st(ranks):
    """Return ranks with ST removed, preserving original rank info."""
    return [(sym, score) for sym, score in ranks if sym not in st_symbols]

def print_topk(ranks, label, topk=TOP_K):
    name_map = dict(zip(info['symbol'], info['name']))
    non_st = filter_st(ranks)
    print(f"\n{'='*80}")
    print(f"  {label} Top-{topk} Recommendations for {target_date.date()} (ST excluded)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Symbol':<10} {'Name':<12} {'Score':<10}")
    print("-" * 50)
    for i, (sym, score) in enumerate(non_st[:topk]):
        name = name_map.get(sym, "?")
        print(f"{i+1:<6} {sym:<10} {name:<12} {score:<10.4f}")

# Also show ST count for context
f0_st = sum(1 for s, _ in ranks_f0[:15] if s in st_symbols)
v2_st = sum(1 for s, _ in ranks_v2[:15] if s in st_symbols)
v3_st = sum(1 for s, _ in ranks_v3[:15] if s in st_symbols)
v4_st = sum(1 for s, _ in ranks_v4[:15] if s in st_symbols)
print(f"\nST in top-15 — F0: {f0_st}/15, V2: {v2_st}/15, V3: {v3_st}/15, V4: {v4_st}/15")

print_topk(ranks_f0, "F0 (14 derived factors, CS z-score, 60d window)")
print_topk(ranks_v2, "V2 (8 raw OHLCV, CS z-score)")
print_topk(ranks_v3, "V3 (20 features, CS z-score)")
print_topk(ranks_v4, "V4 (8 raw + 8 per-stock z-score)")

# ── Overlap analysis (ST excluded) ──
f0_top20 = {s for s, _ in filter_st(ranks_f0)[:20]}
v2_top20 = {s for s, _ in filter_st(ranks_v2)[:20]}
v3_top20 = {s for s, _ in filter_st(ranks_v3)[:20]}
v4_top20 = {s for s, _ in filter_st(ranks_v4)[:20]}
print(f"\nOverlap in top-20 (ST excluded):")
print(f"  F0∩V2: {len(f0_top20 & v2_top20)}  F0∩V3: {len(f0_top20 & v3_top20)}  F0∩V4: {len(f0_top20 & v4_top20)}")
print(f"  V2∩V3: {len(v2_top20 & v3_top20)}  V2∩V4: {len(v2_top20 & v4_top20)}  V3∩V4: {len(v3_top20 & v4_top20)}")

logging.info("Done.")
