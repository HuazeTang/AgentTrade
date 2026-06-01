"""Analyze V3 top-1 picks: pre-launch vs already-launched classification."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dl import DualTowerModel
from dl_v3.derived_features import ALL_V3_COLUMNS, build_v3_feature_cache
from data.cache import read_daily

SEQUENCE_LENGTH = 20
FORWARD_WINDOW = 15
LOOKBACK_WINDOW = 10
CHECKPOINT = "data/models/yaogu_v3_20260530_0532_best.pt"

load_start = date(2025, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading data: %s ~ %s", load_start, load_end)
daily = read_daily(load_start, load_end, prefix="daily_badj")
logging.info("Loaded %d rows", len(daily))

stock_list = pd.read_parquet("data/cache/stock_list.parquet")
st_symbols = set(stock_list[stock_list["name"].str.contains(r"\*?ST", na=False)]["symbol"])
logging.info("ST symbols to filter: %d", len(st_symbols))

# Build V3 feature cache
logging.info("Building V3 feature cache...")
cache = build_v3_feature_cache(daily)
feature_cols = [c for c in ALL_V3_COLUMNS if c in cache.columns]
logging.info("Feature cols: %d", len(feature_cols))

# Build tensors
dates_all = sorted(cache.index.get_level_values("trade_date").unique())
symbols_all = sorted(cache.index.get_level_values("symbol").unique())
logging.info("Dates: %d, Symbols: %d", len(dates_all), len(symbols_all))

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

# Load model
device = "mps" if torch.backends.mps.is_available() else "cpu"
logging.info("Loading V3 model from %s (device=%s)", CHECKPOINT, device)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = DualTowerModel(in_features=checkpoint["model_kwargs"]["in_features"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
logging.info("Model loaded: epoch=%d, val_ap=%.4f", checkpoint["epoch"], checkpoint["val_ap"])

# Run all picks
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
first_picks = df.groupby("symbol").first().reset_index()
first_picks = first_picks.sort_values("date")

# Classification helpers
def max_return_window(prices: np.ndarray) -> float:
    valid = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid) < 2:
        return np.nan
    start = valid[0]
    if start <= 0:
        return np.nan
    return float((np.nanmax(valid) - start) / start)

def has_limit_up(opens, highs, lows, pre_closes, threshold=0.095):
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

    # Pre-window
    pre_start = max(0, idx - LOOKBACK_WINDOW)
    pre_close_slice = close_mat[pre_start:idx, sym_i]
    pre_ret = max_return_window(pre_close_slice)

    pre_opens = open_[pre_start:idx, sym_i]
    pre_highs = high_[pre_start:idx, sym_i]
    pre_lows = low_[pre_start:idx, sym_i]
    pre_pcs = pre_close_[pre_start:idx, sym_i]
    pre_limit_up = has_limit_up(pre_opens, pre_highs, pre_lows, pre_pcs)

    # Post-window
    post_end = min(len(dates_all), idx + FORWARD_WINDOW + 1)
    post_close_slice = close_mat[idx:post_end, sym_i]
    post_ret = max_return_window(post_close_slice)

    # Classification (V3: pre < 10% + post >= 30% = pre-launch)
    pre_ret_20d = max_return_window(close_mat[max(0, idx - 20):idx, sym_i])

    if pd.isna(post_ret):
        status = "pending (数据不足)"
    elif pre_ret is not None and not pd.isna(pre_ret) and pre_ret >= 0.20:
        status = "已启动 (pre 10d>=20%)"
    elif post_ret >= 0.30 and (pd.isna(pre_ret_20d) or pre_ret_20d < 0.10):
        status = "启动前 ✓ (静→爆)"
    elif post_ret >= 0.30:
        status = "启动前(有前兆)"
    elif post_ret >= 0.10:
        status = "小幅上涨"
    elif post_ret < 0:
        status = "亏损"
    else:
        status = "平淡"

    results.append({
        "date": r["date"],
        "symbol": sym,
        "pre_10d_ret": pre_ret if not pd.isna(pre_ret) else None,
        "pre_20d_ret": pre_ret_20d if not pd.isna(pre_ret_20d) else None,
        "pre_limit": pre_limit_up,
        "post_ret": post_ret if not pd.isna(post_ret) else None,
        "status": status,
    })

res_df = pd.DataFrame(results)
res_df["name"] = res_df["symbol"].apply(lambda s: stock_info.loc[s, "name"] if s in stock_info.index else "?")
res_df["date_str"] = pd.to_datetime(res_df["date"]).dt.strftime("%Y-%m-%d")

# Output
print(f"\n{'='*120}")
print(f"V3 Entry Analysis: First-Time Top-1 Picks — Pre-Launch Detection")
print(f"Model: {CHECKPOINT} (val_ap={checkpoint['val_ap']:.4f})")
print(f"{'='*120}")
print(f"{'Date':<12} {'Symbol':<8} {'Name':<10} {'Pre10d':>8} {'Pre20d':>8} {'Pre涨停':>7} {'Post15d':>8} {'判定':>24}")
print(f"{'-'*120}")

counts = {}
for _, r in res_df.iterrows():
    pre10 = f"{r['pre_10d_ret']:.1%}" if r['pre_10d_ret'] is not None else "?"
    pre20 = f"{r['pre_20d_ret']:.1%}" if r['pre_20d_ret'] is not None else "?"
    post = f"{r['post_ret']:.1%}" if r['post_ret'] is not None else "?"
    pre_lu = "是" if r['pre_limit'] else "否"
    st = r['status']
    counts[st] = counts.get(st, 0) + 1
    print(f"{r['date_str']:<12} {r['symbol']:<8} {str(r['name']):<10} {pre10:>8} {pre20:>8} {pre_lu:>7} {post:>8} {st:>24}")

print(f"\n{'='*60}")
print(f"V3 Summary by Category")
print(f"{'='*60}")
total = len(res_df)
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<32} {v:>3} ({v/total:.0%})")

# Pre-launch picks detail
pre_launch = res_df[res_df["status"].str.contains("启动前")]
if len(pre_launch) > 0:
    print(f"\n{'='*80}")
    print(f"V3 Pre-Launch Picks (first-time, pre 20d < 10%, post 15d >= 30%)")
    print(f"{'='*80}")
    for _, r in pre_launch.iterrows():
        print(f"  {r['date_str']} {r['symbol']} {r['name']} → pre10d:{r['pre_10d_ret']:.1%} pre20d:{r['pre_20d_ret']:.1%} post15d:{r['post_ret']:.1%}")

    valid_post = pre_launch["post_ret"].dropna()
    if len(valid_post) > 0:
        print(f"\n  Count: {len(pre_launch)}, Mean post return: {valid_post.mean():.1%}")
else:
    print(f"\n  ⚠ No pre-launch picks found.")

# Compare: what % of first picks had pre_ret >= 20%?
completed = res_df[res_df["pre_10d_ret"].notna()]
if len(completed) > 0:
    already_launched = (completed["pre_10d_ret"] >= 0.20).sum()
    pre_launched_count = len(pre_launch)
    print(f"\n{'='*60}")
    print(f"Key Metric: Pre-Launch Detection Rate")
    print(f"{'='*60}")
    print(f"  Total first-time picks: {total}")
    print(f"  Already launched (pre10d>=20%): {already_launched} ({already_launched/len(completed):.0%})")
    print(f"  True pre-launch (quiet→surge):   {pre_launched_count} ({pre_launched_count/len(completed):.0%})")
    print(f"  (V2 was: 81% already launched, 0% true pre-launch)")
