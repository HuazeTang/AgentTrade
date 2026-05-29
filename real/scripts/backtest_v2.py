"""Backtest Top-1 per day across 2020-2025, show forward 10-day return by year."""
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

# ── Data (wide range for multi-year backtest) ──
load_start = date(2019, 9, 1)
load_end = date(2025, 12, 31)

logging.info("Loading data: %s ~ %s", load_start, load_end)
daily = read_daily(load_start, load_end, prefix="daily_badj")
logging.info("Loaded %d rows, %d symbols", len(daily), daily.index.get_level_values("symbol").nunique())

# ── Feature cache ──
logging.info("Building feature cache...")
cache = build_normalized_feature_cache(daily, raw_ohlcv=True)
feature_cols = [c for c in RAW_OHLCV_COLUMNS if c in cache.columns]

# ── ST filter ──
stock_list = pd.read_parquet("data/cache/stock_list.parquet")
st_symbols = set(stock_list[stock_list["name"].str.contains(r"\*?ST", na=False)]["symbol"])

# ── Build 3D tensor ──
dates_all = sorted(cache.index.get_level_values("trade_date").unique())
symbols_all = sorted(cache.index.get_level_values("symbol").unique())

feat_mats = []
for col in feature_cols:
    mat = cache[col].unstack()
    mat = mat.reindex(index=dates_all, columns=symbols_all)
    feat_mats.append(mat.values)

feat_tensor = np.stack(feat_mats, axis=-1).astype(np.float32)
feat_tensor = np.nan_to_num(feat_tensor, nan=0.0, posinf=0.0, neginf=0.0)
n_dates, n_symbols, n_features = feat_tensor.shape
logging.info("Feature tensor: %d dates × %d symbols × %d features", n_dates, n_symbols, n_features)

date_to_idx = {d: i for i, d in enumerate(dates_all)}
symbol_to_idx = {s: i for i, s in enumerate(symbols_all)}
non_st_indices = np.array([symbol_to_idx[s] for s in symbols_all if s not in st_symbols])
non_st_symbols = [s for s in symbols_all if s not in st_symbols]
logging.info("Non-ST symbols: %d", len(non_st_indices))

# ── Close prices ──
close = daily["close"].unstack()
close_mat = close.reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)

# ── Stock info ──
stock_info = stock_list.set_index("symbol")[["name", "board"]]

# ── Load model ──
device = "mps" if torch.backends.mps.is_available() else "cpu"
logging.info("Device: %s", device)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = DualTowerModel(in_features=checkpoint["model_kwargs"]["in_features"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ── Backtest: every trading day, pick top-1 ──
# Only use dates from 2020 onwards (need 2019 for history)
test_start_idx = max(SEQUENCE_LENGTH, date_to_idx.get(pd.Timestamp("2020-01-02"), SEQUENCE_LENGTH))
test_end_idx = min(n_dates - FORWARD_WINDOW - 1, n_dates - 1)

# Sample weekly (Wednesdays) to avoid overlapping holding periods
sample_indices = [i for i in range(test_start_idx, test_end_idx)]
# Use every trading day but we can aggregate by year

logging.info("Running backtest from %s to %s...", dates_all[test_start_idx].date(), dates_all[test_end_idx].date())

top1_records = []

for idx in range(test_start_idx, test_end_idx):
    td = dates_all[idx]
    # Build sequences: (n_syms, seq_len, n_features)
    seq_slice = feat_tensor[idx - SEQUENCE_LENGTH:idx, :, :]
    seq_slice = seq_slice[:, non_st_indices, :]
    batch = np.transpose(seq_slice, (1, 0, 2))

    # Forward returns
    fwd_prices = close_mat[idx + 1: idx + FORWARD_WINDOW + 1, :][:, non_st_indices]
    start_prices = close_mat[idx, non_st_indices]
    valid = (start_prices > 0) & (~np.isnan(start_prices))
    fwd_max = np.nanmax(np.where(~np.isnan(fwd_prices), fwd_prices, -np.inf), axis=0)
    fwd_ret = np.where(valid & (fwd_max > 0), (fwd_max - start_prices) / start_prices, np.nan)

    # Model inference
    X_tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        scores = model.predict_proba(X_tensor).cpu().numpy().flatten()

    # Top-1
    top_idx = int(np.argmax(scores))
    top1_records.append({
        "date": td,
        "symbol": non_st_symbols[top_idx],
        "score": float(scores[top_idx]),
        "fwd_ret": float(fwd_ret[top_idx]) if not np.isnan(fwd_ret[top_idx]) else np.nan,
    })

    if (idx - test_start_idx + 1) % 200 == 0:
        logging.info("  %d/%d: %s", idx - test_start_idx + 1, test_end_idx - test_start_idx, td.date())

df = pd.DataFrame(top1_records)
df = df.dropna(subset=["fwd_ret"])
df["year"] = pd.to_datetime(df["date"]).dt.year
logging.info("Total top-1 picks: %d", len(df))

# ── By year stats ──
print(f"\n{'='*90}")
print(f"Top-1 Daily Pick — Forward {FORWARD_WINDOW}-day Max Return by Year (ex-ST)")
print(f"{'='*90}")
print(f"{'Year':<6} {'Days':>6} {'Mean Ret':>10} {'Median':>8} {'Std':>10} {'Win%':>7} {'Hit10%':>8} {'Hit30%':>8} {'Max':>10} {'Min':>10}")
print(f"{'-'*90}")

yearly = []
for yr in sorted(df["year"].unique()):
    yd = df[df["year"] == yr]
    mean_r = yd["fwd_ret"].mean()
    med_r = yd["fwd_ret"].median()
    win = (yd["fwd_ret"] > 0).mean()
    hit10 = (yd["fwd_ret"] >= 0.10).mean()
    hit30 = (yd["fwd_ret"] >= 0.30).mean()
    yearly.append({"year": yr, "mean": mean_r, "hit30": hit30, "n": len(yd)})
    print(f"{yr:<6} {len(yd):>6} {mean_r:>9.4f} {med_r:>8.4f} {yd['fwd_ret'].std():>9.4f} "
          f"{win:>6.2%} {hit10:>7.2%} {hit30:>7.2%} {yd['fwd_ret'].max():>9.4f} {yd['fwd_ret'].min():>9.4f}")

# Total
print(f"{'-'*90}")
print(f"{'Total':<6} {len(df):>6} {df['fwd_ret'].mean():>9.4f} {df['fwd_ret'].median():>8.4f} "
      f"{df['fwd_ret'].std():>9.4f} {(df['fwd_ret']>0).mean():>6.2%} "
      f"{(df['fwd_ret']>=0.10).mean():>7.2%} {(df['fwd_ret']>=0.30).mean():>7.2%} "
      f"{df['fwd_ret'].max():>9.4f} {df['fwd_ret'].min():>9.4f}")

# ── Top-1 picks that hit 30%+ ──
print(f"\n{'='*90}")
print(f"Top-1 picks that achieved 30%+ forward return")
print(f"{'='*90}")
big_wins = df[df["fwd_ret"] >= 0.30].sort_values("date")
print(f"{'Date':<12} {'Symbol':<8} {'Name':<8} {'Score':>8} {'Fwd Ret':>10}")
print(f"{'-'*55}")
for _, r in big_wins.iterrows():
    sym = r["symbol"]
    name = stock_info.loc[sym, "name"] if sym in stock_info.index else "?"
    if isinstance(name, pd.Series):
        name = name.iloc[0]
    print(f"{str(r['date'].date()):<12} {sym:<8} {str(name):<8} {r['score']:>8.4f} {r['fwd_ret']:>9.2%}")

print(f"\nTotal big wins (30%+): {len(big_wins)} out of {len(df)} days ({len(big_wins)/len(df):.1%})")
