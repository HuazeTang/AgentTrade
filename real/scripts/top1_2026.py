"""Top-1 picks for 2026, showing forward returns where available."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dl import DualTowerModel
from dl.derived_features import RAW_OHLCV_COLUMNS, build_normalized_feature_cache
from data.cache import read_daily

SEQUENCE_LENGTH = 20
FORWARD_WINDOW = 10
CHECKPOINT = "data/models/yaogu_20260529_1646_best.pt"

# ── Data ──
load_start = date(2025, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading data: %s ~ %s", load_start, load_end)
daily = read_daily(load_start, load_end, prefix="daily_badj")
logging.info("Loaded %d rows, %d symbols", len(daily), daily.index.get_level_values("symbol").nunique())

# ── ST filter ──
stock_list = pd.read_parquet("data/cache/stock_list.parquet")
st_symbols = set(stock_list[stock_list["name"].str.contains(r"\*?ST", na=False)]["symbol"])
logging.info("ST symbols to filter: %d", len(st_symbols))

# ── Feature cache ──
logging.info("Building feature cache...")
cache = build_normalized_feature_cache(daily, raw_ohlcv=True)
feature_cols = [c for c in RAW_OHLCV_COLUMNS if c in cache.columns]

# ── Build tensors ──
dates_all = sorted(cache.index.get_level_values("trade_date").unique())
symbols_all = sorted(cache.index.get_level_values("symbol").unique())

feat_mats = []
for col in feature_cols:
    mat = cache[col].unstack()
    mat = mat.reindex(index=dates_all, columns=symbols_all)
    feat_mats.append(mat.values)

feat_tensor = np.stack(feat_mats, axis=-1).astype(np.float32)
feat_tensor = np.nan_to_num(feat_tensor, nan=0.0, posinf=0.0, neginf=0.0)

date_to_idx = {d: i for i, d in enumerate(dates_all)}
symbol_to_idx = {s: i for i, s in enumerate(symbols_all)}
non_st_indices = np.array([symbol_to_idx[s] for s in symbols_all if s not in st_symbols])
non_st_symbols = [s for s in symbols_all if s not in st_symbols]

close = daily["close"].unstack()
close_mat = close.reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)

# OHLC for next-day gap-lock check
open_ = daily["open"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
high_ = daily["high"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
low_ = daily["low"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
pre_close_ = daily["pre_close"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)

def is_gap_lock(idx: int, sym_idx: int) -> bool:
    """Check if next day is 一字涨停 (unbuyable)."""
    next_i = idx + 1
    if next_i >= len(dates_all):
        return False
    o = open_[next_i, sym_idx]
    h = high_[next_i, sym_idx]
    l = low_[next_i, sym_idx]
    pc = pre_close_[next_i, sym_idx]
    if pc <= 0 or np.isnan(pc):
        # fallback: use current close as reference
        pc = close_mat[idx, sym_idx]
    if pc <= 0 or np.isnan(pc) or o <= 0 or np.isnan(o):
        return False
    limit_pct = (o / pc) - 1.0
    # Limit-up and never traded below open
    return bool(limit_pct >= 0.095 and l >= o * 0.995)

stock_info = stock_list.set_index("symbol")[["name"]]

# ── Load model ──
device = "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = DualTowerModel(in_features=checkpoint["model_kwargs"]["in_features"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ── Run top-1 for 2026 ──
start_2026 = max(SEQUENCE_LENGTH, date_to_idx.get(pd.Timestamp("2026-01-02"), SEQUENCE_LENGTH))
end_idx = min(len(dates_all) - 1, len(dates_all) - 1)  # last available date

records = []
for idx in range(start_2026, end_idx + 1):
    td = dates_all[idx]
    if td.date() < date(2026, 1, 1):
        continue

    seq_slice = feat_tensor[idx - SEQUENCE_LENGTH:idx, :, :]
    seq_slice = seq_slice[:, non_st_indices, :]
    batch = np.transpose(seq_slice, (1, 0, 2))

    # Forward return (may be NaN for recent dates)
    fwd_end = min(idx + FORWARD_WINDOW, len(dates_all) - 1)
    fwd_prices = close_mat[idx + 1: fwd_end + 1, :][:, non_st_indices]
    start_prices = close_mat[idx, non_st_indices]
    if fwd_prices.shape[0] == 0:
        fwd_ret = np.full(len(non_st_indices), np.nan)
    else:
        valid = (start_prices > 0) & (~np.isnan(start_prices))
        fwd_max = np.nanmax(np.where(~np.isnan(fwd_prices), fwd_prices, -np.inf), axis=0)
        fwd_ret = np.where(valid & (fwd_max > 0), (fwd_max - start_prices) / start_prices, np.nan)

    X_tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        scores = model.predict_proba(X_tensor).cpu().numpy().flatten()

    top_idx = int(np.argmax(scores))
    top_sym_idx = non_st_indices[top_idx]
    gap_lock = is_gap_lock(idx, top_sym_idx)
    records.append({
        "date": td,
        "symbol": non_st_symbols[top_idx],
        "score": float(scores[top_idx]),
        "fwd_ret": float(fwd_ret[top_idx]) if not np.isnan(fwd_ret[top_idx]) else None,
        "gap_lock": gap_lock,
    })

df = pd.DataFrame(records)
df["name"] = df["symbol"].apply(lambda s: stock_info.loc[s, "name"] if s in stock_info.index else "?")
df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

# ── Output ──
print(f"\n{'='*100}")
print(f"2026 Top-1 Daily Picks — ex-ST Main Board (V2 SmoothAP Model)")
print(f"{'='*100}")
print(f"{'Date':<12} {'Symbol':<8} {'Name':<10} {'Score':>8} {'Fwd 10d Ret':>12} {'1字':>4} {'Note':>18}")
print(f"{'-'*100}")

completed = 0
completed_win = 0
completed_30 = 0
gap_lock_count = 0
completed_buyable = 0
completed_buyable_win = 0
completed_buyable_30 = 0
recent_scores = []

for _, r in df.iterrows():
    fwd = r["fwd_ret"]
    gl = "🔒" if r["gap_lock"] else ""
    if r["gap_lock"]:
        gap_lock_count += 1
    if fwd is not None:
        completed += 1
        if fwd > 0:
            completed_win += 1
        if fwd >= 0.30:
            completed_30 += 1
        if not r["gap_lock"]:
            completed_buyable += 1
            if fwd > 0:
                completed_buyable_win += 1
            if fwd >= 0.30:
                completed_buyable_30 += 1
        note = "← 30%+!" if fwd >= 0.30 else ("+" if fwd > 0 else "-")
        print(f"{r['date_str']:<12} {r['symbol']:<8} {str(r['name']):<10} {r['score']:>8.4f} {fwd:>11.2%} {gl:>4} {note:>18}")
    else:
        days_avail = (dates_all[-1] - r["date"]).days
        note = f"(need {FORWARD_WINDOW - days_avail:.0f} more days)"
        print(f"{r['date_str']:<12} {r['symbol']:<8} {str(r['name']):<10} {r['score']:>8.4f} {'pending':>12} {gl:>4} {note:>18}")
        recent_scores.append(r["score"])

# ── Summary ──
print(f"\n{'='*50}")
print(f"Summary")
print(f"{'='*50}")
if completed > 0:
    print(f"\n  All completed picks ({completed}):")
    print(f"    Mean return: {df['fwd_ret'].dropna().mean():.2%}")
    print(f"    Win rate:    {completed_win/completed:.1%}")
    print(f"    Hit 30%+:    {completed_30/completed:.1%} ({completed_30}/{completed})")
    print(f"    一字涨停:     {gap_lock_count} ({gap_lock_count/completed:.0%})")
if completed_buyable > 0:
    buyable = df[df["gap_lock"] == False]
    print(f"\n  Excluding 一字涨停 ({completed_buyable} buyable):")
    print(f"    Mean return: {buyable['fwd_ret'].dropna().mean():.2%}")
    print(f"    Win rate:    {completed_buyable_win/completed_buyable:.1%}")
    print(f"    Hit 30%+:    {completed_buyable_30/completed_buyable:.1%} ({completed_buyable_30}/{completed_buyable})")
if recent_scores:
    print(f"\nPending picks (not enough forward data): {len(recent_scores)}")
    print(f"  Score range: {min(recent_scores):.4f} ~ {max(recent_scores):.4f}")
    print(f"  Score median: {np.median(recent_scores):.4f}")

# ── Latest day top-5 ──
print(f"\n{'='*90}")
print(f"Latest trading day: {df['date_str'].iloc[-1]}")
print(f"{'='*90}")
latest_idx = len(dates_all) - 1
seq_slice = feat_tensor[latest_idx - SEQUENCE_LENGTH:latest_idx, :, :]
seq_slice = seq_slice[:, non_st_indices, :]
batch = np.transpose(seq_slice, (1, 0, 2))
X_tensor = torch.from_numpy(batch).to(device)
with torch.no_grad():
    scores = model.predict_proba(X_tensor).cpu().numpy().flatten()

top5 = np.argsort(scores)[-5:][::-1]
print(f"{'Rank':<6} {'Symbol':<8} {'Name':<10} {'Score':>8}")
print(f"{'-'*35}")
for rank, ti in enumerate(top5, 1):
    sym = non_st_symbols[ti]
    name = stock_info.loc[sym, "name"] if sym in stock_info.index else "?"
    if isinstance(name, pd.Series):
        name = name.iloc[0]
    print(f"{rank:<6} {sym:<8} {str(name):<10} {scores[ti]:>8.4f}")
