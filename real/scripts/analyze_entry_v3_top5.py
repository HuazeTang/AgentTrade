"""Analyze V3 top-5 picks per day: pre-launch detection coverage."""
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
TOP_K = 5
CHECKPOINT = "data/models/yaogu_v3_20260530_0532_best.pt"

load_start = date(2025, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading data: %s ~ %s", load_start, load_end)
daily = read_daily(load_start, load_end, prefix="daily_badj")
logging.info("Loaded %d rows", len(daily))

stock_list = pd.read_parquet("data/cache/stock_list.parquet")
st_symbols = set(stock_list[stock_list["name"].str.contains(r"\*?ST", na=False)]["symbol"])
logging.info("ST symbols to filter: %d", len(st_symbols))

logging.info("Building V3 feature cache...")
cache = build_v3_feature_cache(daily)
feature_cols = [c for c in ALL_V3_COLUMNS if c in cache.columns]
logging.info("Feature cols: %d", len(feature_cols))

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

device = "mps" if torch.backends.mps.is_available() else "cpu"
logging.info("Loading V3 model from %s", CHECKPOINT)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = DualTowerModel(in_features=checkpoint["model_kwargs"]["in_features"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

def max_return_window(prices: np.ndarray) -> float:
    valid = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid) < 2:
        return np.nan
    start = valid[0]
    if start <= 0:
        return np.nan
    return float((np.nanmax(valid) - start) / start)

# ── Run top-K per day ──
start_2026 = max(SEQUENCE_LENGTH, date_to_idx.get(pd.Timestamp("2026-01-02"), SEQUENCE_LENGTH))
end_idx = len(dates_all) - 1

all_topk = []  # list of lists per day
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

    top_k_idx = np.argsort(scores)[-TOP_K:][::-1]
    day_picks = []
    for rank, ti in enumerate(top_k_idx, 1):
        sym_idx = non_st_indices[ti]
        sym = non_st_symbols[ti]

        # Pre-window
        pre_start = max(0, idx - LOOKBACK_WINDOW)
        pre_10d = max_return_window(close_mat[pre_start:idx, sym_idx])
        pre_20d = max_return_window(close_mat[max(0, idx - 20):idx, sym_idx])

        # Post-window
        post_end = min(len(dates_all), idx + FORWARD_WINDOW + 1)
        post_ret = max_return_window(close_mat[idx:post_end, sym_idx])

        # Classification
        if pd.isna(post_ret):
            cat = "pending"
        elif pre_10d is not None and not pd.isna(pre_10d) and pre_10d >= 0.20:
            cat = "已启动"
        elif post_ret >= 0.30 and (pd.isna(pre_20d) or pre_20d < 0.10):
            cat = "启动前 ✓"
        elif post_ret >= 0.30:
            cat = "启动前(有前兆)"
        elif post_ret >= 0.10:
            cat = "小幅上涨"
        elif post_ret < 0:
            cat = "亏损"
        else:
            cat = "平淡"

        day_picks.append({
            "date": td,
            "rank": rank,
            "symbol": sym,
            "name": stock_info.loc[sym, "name"] if sym in stock_info.index else "?",
            "score": float(scores[ti]),
            "pre_10d": pre_10d if not pd.isna(pre_10d) else None,
            "pre_20d": pre_20d if not pd.isna(pre_20d) else None,
            "post_ret": post_ret if not pd.isna(post_ret) else None,
            "cat": cat,
        })
    all_topk.append(day_picks)

# ── Coverage analysis: per day, best category in top-K ──
print(f"\n{'='*130}")
print(f"V3 Top-{TOP_K} Analysis: Pre-Launch Detection Coverage")
print(f"Model: {CHECKPOINT} (val_ap={checkpoint['val_ap']:.4f})")
print(f"{'='*130}")

# Show each day with top-5 picks and best result
days_with_prelaunch = 0
days_with_prelaunch_any = 0  # including "有前兆"
days_launched_only = 0
all_cats = {}

for day_picks in all_topk:
    td = day_picks[0]["date"]
    td_str = pd.Timestamp(td).strftime("%Y-%m-%d")
    cats_today = set(p["cat"] for p in day_picks)
    best_cat = "pending"
    for c in ["启动前 ✓", "启动前(有前兆)", "小幅上涨", "已启动", "平淡", "亏损", "pending"]:
        if c in cats_today:
            best_cat = c
            break

    # Find the best post_ret in top-5
    completed = [p for p in day_picks if p["post_ret"] is not None]
    best_post = max(p["post_ret"] for p in completed) if completed else None

    # Count pre-launch in top-5
    prelaunch_today = [p for p in day_picks if "启动前" in p["cat"]]
    launched_today = [p for p in day_picks if p["cat"] == "已启动"]

    if any("启动前 ✓" in p["cat"] for p in day_picks):
        days_with_prelaunch += 1
    if any("启动前" in p["cat"] for p in day_picks):
        days_with_prelaunch_any += 1
    if any(p["cat"] == "已启动" for p in day_picks) and not any("启动前" in p["cat"] for p in day_picks):
        days_launched_only += 1

    for p in day_picks:
        all_cats[p["cat"]] = all_cats.get(p["cat"], 0) + 1

    # Print daily summary
    pre_str = f"best post={best_post:.1%}" if best_post is not None else "pending"
    prelaunch_str = ""
    if prelaunch_today:
        prelaunch_str = f" ★ 启动前: {prelaunch_today[0]['symbol']} {prelaunch_today[0]['name']} (rank#{prelaunch_today[0]['rank']})"
    print(f"  {td_str} | Best: {best_cat:<16} | {pre_str:<20}{prelaunch_str}")

print(f"\n{'='*80}")
print(f"Coverage Summary (out of {len(all_topk)} trading days)")
print(f"{'='*80}")
print(f"  Days with 启动前 ✓ (true pre-launch) in top-{TOP_K}:  {days_with_prelaunch} ({days_with_prelaunch/len(all_topk):.0%})")
print(f"  Days with 启动前 (any) in top-{TOP_K}:              {days_with_prelaunch_any} ({days_with_prelaunch_any/len(all_topk):.0%})")
print(f"  Days with only '已启动' in top-{TOP_K}:              {days_launched_only} ({days_launched_only/len(all_topk):.0%})")

print(f"\n  All top-{TOP_K} picks by category:")
total_picks = sum(all_cats.values())
for k, v in sorted(all_cats.items(), key=lambda x: -x[1]):
    print(f"    {k:<24} {v:>4} ({v/total_picks:.0%})")

# ── Detailed: full listing of pre-launch picks ──
prelaunch_all = []
for day_picks in all_topk:
    for p in day_picks:
        if "启动前" in p["cat"]:
            prelaunch_all.append(p)

if prelaunch_all:
    print(f"\n{'='*110}")
    print(f"All Pre-Launch Picks in Top-{TOP_K} ({len(prelaunch_all)} total)")
    print(f"{'='*110}")
    print(f"{'Date':<12} {'Rank':<5} {'Symbol':<8} {'Name':<10} {'Score':>7} {'Pre10d':>8} {'Pre20d':>8} {'Post15d':>8} {'Cat':>16}")
    print(f"{'-'*110}")
    for p in prelaunch_all:
        pre10 = f"{p['pre_10d']:.1%}" if p['pre_10d'] is not None else "?"
        pre20 = f"{p['pre_20d']:.1%}" if p['pre_20d'] is not None else "?"
        post = f"{p['post_ret']:.1%}" if p['post_ret'] is not None else "?"
        print(f"{pd.Timestamp(p['date']).strftime('%Y-%m-%d'):<12} {p['rank']:<5} {p['symbol']:<8} {str(p['name']):<10} {p['score']:>7.4f} {pre10:>8} {pre20:>8} {post:>8} {p['cat']:>16}")

    valid = [p for p in prelaunch_all if p["post_ret"] is not None]
    if valid:
        posts = [p["post_ret"] for p in valid]
        print(f"\n  Mean post return: {np.mean(posts):.1%}, Median: {np.median(posts):.1%}")
        print(f"  >=50%: {sum(1 for r in posts if r >= 0.50)}/{len(posts)}")

# ── Latest day top-5 detail ──
print(f"\n{'='*90}")
latest = all_topk[-1]
print(f"Latest: {pd.Timestamp(latest[0]['date']).strftime('%Y-%m-%d')} Top-{TOP_K}")
print(f"{'='*90}")
for p in latest:
    pre10 = f"{p['pre_10d']:.1%}" if p['pre_10d'] is not None else "?"
    post = f"{p['post_ret']:.1%}" if p['post_ret'] is not None else "?"
    print(f"  #{p['rank']} {p['symbol']:<8} {str(p['name']):<10} score={p['score']:.4f} pre10d={pre10} post={post}")
