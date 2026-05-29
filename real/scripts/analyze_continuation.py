"""Test if model distinguishes continuation vs A-kill among already-running stocks."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dl import DualTowerModel
from dl.derived_features import RAW_OHLCV_COLUMNS, build_normalized_feature_cache
from data.cache import read_daily

SEQUENCE_LENGTH = 20
FORWARD_WINDOW = 10
CHECKPOINT = "data/models/yaogu_20260529_1646_best.pt"

load_start = date(2025, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading data...")
daily = read_daily(load_start, load_end, prefix="daily_badj")

stock_list = pd.read_parquet("data/cache/stock_list.parquet")
st_symbols = set(stock_list[stock_list["name"].str.contains(r"\*?ST", na=False)]["symbol"])

cache = build_normalized_feature_cache(daily, raw_ohlcv=True)
feature_cols = [c for c in RAW_OHLCV_COLUMNS if c in cache.columns]

dates_all = sorted(cache.index.get_level_values("trade_date").unique())
symbols_all = sorted(cache.index.get_level_values("symbol").unique())

feat_mats = []
for col in feature_cols:
    mat = cache[col].unstack()
    mat = mat.reindex(index=dates_all, columns=symbols_all)
    feat_mats.append(mat.values)
feat_tensor = np.stack(feat_mats, axis=-1).astype(np.float32)
feat_tensor = np.nan_to_num(feat_tensor, nan=0.0, posinf=0.0, neginf=0.0)

close = daily["close"].unstack()
close_mat = close.reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)

date_to_idx = {d: i for i, d in enumerate(dates_all)}
symbol_to_idx = {s: i for i, s in enumerate(symbols_all)}
non_st_indices = np.array([symbol_to_idx[s] for s in symbols_all if s not in st_symbols])
non_st_symbols = [s for s in symbols_all if s not in st_symbols]

stock_info = stock_list.set_index("symbol")[["name"]]

# ── Load model ──
device = "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = DualTowerModel(in_features=checkpoint["model_kwargs"]["in_features"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ── For each trading day in 2026, classify all already-running stocks ──
LOOKBACK = 10

def max_return(prices: np.ndarray) -> float:
    valid = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid) < 2:
        return np.nan
    return float((np.nanmax(valid) - valid[0]) / valid[0])

start_2026 = max(SEQUENCE_LENGTH, date_to_idx.get(pd.Timestamp("2026-01-02"), SEQUENCE_LENGTH))
# Only analyze completed windows (need forward data)
end_analysis = min(len(dates_all) - FORWARD_WINDOW - 1, len(dates_all) - 1)

# Sample every 5th trading day to keep it fast
sample_indices = [i for i in range(start_2026, end_analysis) if (i - start_2026) % 5 == 0]

all_records = []
for idx in sample_indices:
    td = dates_all[idx]
    if td.date() < date(2026, 1, 1):
        continue

    # Pre-10d returns for all non-ST stocks
    pre_start = idx - LOOKBACK
    pre_prices = close_mat[pre_start:idx, :][:, non_st_indices]
    pre_rets = np.array([max_return(pre_prices[:, j]) for j in range(len(non_st_indices))])

    # Post-10d returns
    post_end = min(idx + FORWARD_WINDOW + 1, len(dates_all))
    post_prices = close_mat[idx:post_end, :][:, non_st_indices]
    post_rets = np.array([max_return(post_prices[:, j]) for j in range(len(non_st_indices))])

    # Model scores
    seq_slice = feat_tensor[idx - SEQUENCE_LENGTH:idx, :, :]
    seq_slice = seq_slice[:, non_st_indices, :]
    batch = np.transpose(seq_slice, (1, 0, 2))
    X_tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        scores = model.predict_proba(X_tensor).cpu().numpy().flatten()

    for j in range(len(non_st_indices)):
        all_records.append({
            "date_idx": idx,
            "sym_idx": non_st_indices[j],
            "symbol": non_st_symbols[j],
            "score": float(scores[j]),
            "pre_ret": pre_rets[j] if not np.isnan(pre_rets[j]) else None,
            "post_ret": post_rets[j] if not np.isnan(post_rets[j]) else None,
        })

df_all = pd.DataFrame(all_records)
df_all = df_all.dropna(subset=["pre_ret", "post_ret"])

# ── Focus: already-running stocks (pre 10d >= 20%) ──
running = df_all[df_all["pre_ret"] >= 0.20].copy()
running["continuation"] = running["post_ret"] >= 0.10  # True continuation
running["a_kill"] = running["post_ret"] <= -0.10       # A-kill (>10% loss)
running["stall"] = (running["post_ret"] > -0.10) & (running["post_ret"] < 0.10)  # stalled

# ── Binned analysis: by score quantile among already-running stocks ──
running["score_bin"] = pd.qcut(running["score"], q=5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
running["score_bin"] = running["score_bin"].astype(str)

print(f"\n{'='*90}")
print(f"Continuation vs A-Kill Discrimination (stocks with pre-10d ≥ 20%)")
print(f"{'='*90}")
print(f"Total already-running observations: {len(running)} across {len(sample_indices)} sample dates")
print(f"\n  Overall: continuation={running['continuation'].mean():.1%}  "
      f"A-kill={running['a_kill'].mean():.1%}  stall={running['stall'].mean():.1%}")

print(f"\n{'Score Bin':<12} {'Count':>7} {'Continuation':>14} {'Stall':>8} {'A-Kill':>9} {'Mean Post':>10} {'Mean Pre':>9}")
print(f"{'-'*75}")
for bin_name in ["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"]:
    b = running[running["score_bin"] == bin_name]
    print(f"{bin_name:<12} {len(b):>7} {b['continuation'].mean():>13.1%} "
          f"{b['stall'].mean():>7.1%} {b['a_kill'].mean():>8.1%} "
          f"{b['post_ret'].mean():>9.2%} {b['pre_ret'].mean():>8.2%}")

# ── Compare: Q5 vs Q1 ──
q1 = running[running["score_bin"] == "Q1(low)"]
q5 = running[running["score_bin"] == "Q5(high)"]
print(f"\n{'='*50}")
print(f"Q5(high) vs Q1(low) among already-running:")
print(f"  Continuation: {q5['continuation'].mean():.1%} vs {q1['continuation'].mean():.1%}  "
      f"({q5['continuation'].mean()/max(q1['continuation'].mean(),0.001):.1f}x)")
print(f"  A-Kill:       {q5['a_kill'].mean():.1%} vs {q1['a_kill'].mean():.1%}  "
      f"({'lower' if q5['a_kill'].mean() < q1['a_kill'].mean() else 'higher'})")

# ── Correlation ──
from scipy.stats import spearmanr
corr, pval = spearmanr(running["score"], running["post_ret"])
print(f"\n  Spearman rank corr (score vs post_ret): {corr:.4f} (p={pval:.2e})")

# ── Top-1 picks among already-running ──
print(f"\n{'='*90}")
print(f"Among already-running stocks, what does top-1 get?")
print(f"{'='*90}")
running_sorted = running.sort_values(["date_idx", "score"], ascending=[True, False])
top1_each_day = running_sorted.groupby("date_idx").first().reset_index()
print(f"  Days with running stocks: {len(top1_each_day)}")
print(f"  Continuation: {top1_each_day['continuation'].mean():.1%}")
print(f"  A-Kill:       {top1_each_day['a_kill'].mean():.1%}")
print(f"  Mean post:    {top1_each_day['post_ret'].mean():.2%}")
