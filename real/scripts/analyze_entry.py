"""Analyze whether top-1 picks catch stocks before or after launch."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import read_daily

SEQUENCE_LENGTH = 20
FORWARD_WINDOW = 10
LOOKBACK_WINDOW = 10

# ── Re-read the 2026 results ──
# We'll regenerate the picks (same logic as top1_2026.py) but focus on analysis
import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dl import DualTowerModel
from dl.derived_features import RAW_OHLCV_COLUMNS, build_normalized_feature_cache

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
open_ = daily["open"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
high_ = daily["high"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
low_ = daily["low"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
pre_close_ = daily["pre_close"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)

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

# ── Run all picks ──
start_2026 = max(SEQUENCE_LENGTH, date_to_idx.get(pd.Timestamp("2026-01-02"), SEQUENCE_LENGTH))
end_idx = len(dates_all) - 1

records = []
for idx in range(start_2026, end_idx + 1):
    td = dates_all[idx]
    if td.date() < date(2026, 1, 1):
        continue
    seq_slice = feat_tensor[idx - SEQUENCE_LENGTH:idx, :, :]
    seq_slice = seq_slice[:, non_st_indices, :]
    batch = np.transpose(seq_slice, (1, 0, 2))

    X_tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        scores = model.predict_proba(X_tensor).cpu().numpy().flatten()

    top_idx = int(np.argmax(scores))
    top_sym_idx = non_st_indices[top_idx]
    records.append({
        "date": td,
        "date_idx": idx,
        "symbol": non_st_symbols[top_idx],
        "sym_idx": top_sym_idx,
        "score": float(scores[top_idx]),
    })

df = pd.DataFrame(records)

# ── For each unique symbol, find FIRST appearance ──
first_picks = df.groupby("symbol").first().reset_index()
first_picks = first_picks.sort_values("date")

# ── Classify each first pick ──
def max_return_window(prices: np.ndarray) -> float:
    """Max return relative to first valid price in window."""
    valid = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid) < 2:
        return np.nan
    start = valid[0]
    if start <= 0:
        return np.nan
    return float((np.nanmax(valid) - start) / start)

def has_limit_up(opens, highs, lows, pre_closes, threshold=0.095):
    """Check if any day in the window had a limit-up."""
    for o, h, l, pc in zip(opens, highs, lows, pre_closes):
        if pc <= 0 or np.isnan(pc) or o <= 0 or np.isnan(o):
            continue
        if (o / pc) - 1.0 >= threshold:
            return True
        if (h / pc) - 1.0 >= threshold:
            return True
    return False

results = []
for _, r in first_picks.iterrows():
    sym = r["symbol"]
    idx = r["date_idx"]
    sym_i = r["sym_idx"]
    td = r["date"]

    # Pre-window: LOOKBACK_WINDOW days before pick
    pre_start = max(0, idx - LOOKBACK_WINDOW)
    pre_close = close_mat[pre_start:idx, sym_i]
    pre_ret = max_return_window(pre_close)

    # Pre limit-ups?
    pre_opens = open_[pre_start:idx, sym_i]
    pre_highs = high_[pre_start:idx, sym_i]
    pre_lows = low_[pre_start:idx, sym_i]
    pre_pcs = pre_close_[pre_start:idx, sym_i]
    pre_limit_up = has_limit_up(pre_opens, pre_highs, pre_lows, pre_pcs)

    # Post-window: FORWARD_WINDOW days after pick
    post_end = min(len(dates_all), idx + FORWARD_WINDOW + 1)
    post_close = close_mat[idx:post_end, sym_i]
    post_ret = max_return_window(post_close)

    # Classification
    if pd.isna(post_ret):
        status = "pending"
    elif post_ret >= 0.30 and (pd.isna(pre_ret) or pre_ret < 0.10):
        status = "启动前 ✓"
    elif not pd.isna(pre_ret) and pre_ret >= 0.20:
        status = "已启动"
    elif post_ret >= 0.30:
        status = "启动前(有前兆)"
    elif post_ret >= 0.10:
        status = "小涨"
    elif post_ret < 0:
        status = "亏损"
    else:
        status = "平淡"

    results.append({
        "date": td,
        "symbol": sym,
        "pre_ret": pre_ret if not pd.isna(pre_ret) else None,
        "pre_limit": pre_limit_up,
        "post_ret": post_ret if not pd.isna(post_ret) else None,
        "status": status,
    })

res_df = pd.DataFrame(results)
res_df["name"] = res_df["symbol"].apply(lambda s: stock_info.loc[s, "name"] if s in stock_info.index else "?")
res_df["date_str"] = pd.to_datetime(res_df["date"]).dt.strftime("%Y-%m-%d")

# ── Output ──
print(f"\n{'='*110}")
print(f"Entry Analysis: First-time Top-1 Picks — Pre vs Post Launch")
print(f"{'='*110}")
print(f"{'Date':<12} {'Symbol':<8} {'Name':<10} {'Pre 10d':>9} {'Pre 涨停':>8} {'Post 10d':>9} {'判定':>16}")
print(f"{'-'*110}")

counts = {}
for _, r in res_df.iterrows():
    pre_str = f"{r['pre_ret']:.1%}" if r['pre_ret'] is not None else "?"
    post_str = f"{r['post_ret']:.1%}" if r['post_ret'] is not None else "?"
    pre_lu = "是" if r['pre_limit'] else "否"
    st = r['status']
    counts[st] = counts.get(st, 0) + 1
    print(f"{r['date_str']:<12} {r['symbol']:<8} {str(r['name']):<10} {pre_str:>9} {pre_lu:>8} {post_str:>9} {st:>16}")

print(f"\n{'='*50}")
print(f"Summary by Category")
print(f"{'='*50}")
total = len(res_df)
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<24} {v:>3} ({v/total:.0%})")

# ── Detailed: 启动前 picks ──
pre_launch = res_df[res_df["status"].str.contains("启动前")]
if len(pre_launch) > 0:
    print(f"\n{'='*80}")
    print(f"Pre-Launch Picks (first-time, pre < 10%, post >= 30%)")
    print(f"{'='*80}")
    for _, r in pre_launch.iterrows():
        print(f"  {r['date_str']} {r['symbol']} {r['name']} → post 10d: {r['post_ret']:.1%}")

    valid_post = pre_launch["post_ret"].dropna()
    if len(valid_post) > 0:
        print(f"\n  Count: {len(pre_launch)}, Mean post return: {valid_post.mean():.1%}")
