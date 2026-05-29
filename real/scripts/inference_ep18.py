"""Quick inference with ep18 checkpoint on latest data."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from dl.two_stage_factor import TwoStageYaoguFactor
from dl.derived_features import (
    DERIVED_FEATURE_COLUMNS,
    build_normalized_feature_cache,
)
from data.cache import read_daily, data_summary

# Check data range
summary = data_summary("daily_badj")
print(f"Data range: {summary['dates'][0]} ~ {summary['dates'][1]}, {summary['symbols']} symbols")

# Load stock list for names
stock_list = pd.read_parquet("data/cache/stock_list.parquet")
stock_info = stock_list.set_index("symbol")[["name", "board"]]

# Load recent data
end_date = summary["dates"][1]
start_date = end_date - timedelta(days=120)  # need 60+ days for features
print(f"Loading data: {start_date} ~ {end_date}")

daily = read_daily(start_date, end_date, prefix="daily_badj")
print(f"Loaded {len(daily)} rows, {daily.index.get_level_values('symbol').nunique()} symbols")

# Feature cache
print("Building feature cache...")
cache = build_normalized_feature_cache(daily)
print(f"Feature cache: {len(cache)} rows, {len(cache.columns)} columns")

# Load factor with ep18
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

factor = TwoStageYaoguFactor(
    screener_path="data/models/yaogu_screener_v2.joblib",
    dl_path="data/models/yaogu_20260528_2051_ep18.pt",
    device=device,
    feature_cache=cache,
)
factor._load_models()
print(f"DL threshold: {factor._dl_threshold:.4f}")
print(f"Screener threshold: {factor._screener.threshold:.4f}")

# Get latest trading day
close = daily["close"].unstack()
all_dates = sorted(close.index)
latest_date = all_dates[-1]
print(f"\nLatest trading date: {latest_date.date()}")

# Run screener
symbols = sorted(daily.index.get_level_values("symbol").unique())
screener_scores = factor._screener.score_day(cache, latest_date, symbols)
candidates = screener_scores[screener_scores >= factor._screener.threshold]
print(f"Candidates: {len(candidates)} (screener score >= {factor._screener.threshold:.4f})")

if candidates.empty:
    print("No candidates passed screener.")
    sys.exit(0)

# Run DL model on candidates
SEQUENCE_LENGTH = 60
seq_start = len(all_dates) - 1 - SEQUENCE_LENGTH
seq_dates = all_dates[seq_start:-1]
print(f"Sequence: {seq_dates[0].date()} ~ {seq_dates[-1].date()}")

feature_cols = [c for c in DERIVED_FEATURE_COLUMNS if c in cache.columns]

batch_feats = []
batch_syms = []
for sym in candidates.index:
    try:
        mask = (
            cache.index.get_level_values("trade_date").isin(seq_dates) &
            (cache.index.get_level_values("symbol") == sym)
        )
        rows = cache.loc[mask][feature_cols]
    except (KeyError, IndexError):
        continue
    if len(rows) < SEQUENCE_LENGTH:
        continue
    rows = rows.sort_index(level="trade_date")
    X = rows.values[:SEQUENCE_LENGTH].astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    batch_feats.append(X)
    batch_syms.append(sym)

print(f"Valid sequences for DL: {len(batch_feats)}")

if not batch_feats:
    print("No valid sequences.")
    sys.exit(0)

X_tensor = torch.from_numpy(np.stack(batch_feats)).to(device)
factor._dl_model.eval()
with torch.no_grad():
    probs = factor._dl_model.predict_proba(X_tensor).cpu().numpy().flatten()

# Build results with stock info, price, and change
# Get latest day's price data
latest_mask = daily.index.get_level_values("trade_date") == latest_date
latest_data = daily[latest_mask].reset_index(level=0, drop=True)
latest_data = latest_data[["close", "pre_close", "board"]]

results = []
for sym, prob in zip(batch_syms, probs):
    name = stock_info.loc[sym, "name"] if sym in stock_info.index else "?"
    board = "?"
    price = np.nan
    pct_chg = np.nan
    if sym in latest_data.index:
        row = latest_data.loc[sym]
        board = row.get("board", "?")
        price = row.get("close", np.nan)
        pct_chg = (row["close"] - row["pre_close"]) / row["pre_close"] * 100 if pd.notna(row.get("pre_close")) else np.nan

    results.append({
        "symbol": sym, "name": name, "board": board,
        "price": price, "pct_chg": pct_chg,
        "dl_score": float(prob), "screener_score": float(candidates[sym]),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("dl_score", ascending=False)

# Map board codes to readable names
BOARD_NAMES = {"main_board": "主板", "chinext": "创业板", "star_market": "科创板", "beijing": "北交所"}
results_df["board_cn"] = results_df["board"].map(BOARD_NAMES).fillna(results_df["board"])

# Top 30
top_n = 30
print(f"\n{'='*110}")
print(f"Top {top_n} Yaogu Candidates — {latest_date.date()} (ep18, thr={factor._dl_threshold:.2f})")
print(f"{'='*110}")
print(f"{'代码':<8} {'名称':<8} {'板块':<6} {'现价':>8} {'涨跌幅':>8} {'DL Score':>10} {'Screener':>10}")
print(f"{'-'*110}")
for _, r in results_df.head(top_n).iterrows():
    marker = " ←" if r["dl_score"] >= factor._dl_threshold else ""
    pct_str = f"{r['pct_chg']:+.2f}%" if pd.notna(r['pct_chg']) else "-"
    print(f"{r['symbol']:<8} {r['name']:<8} {r['board_cn']:<6} {r['price']:>8.2f} {pct_str:>8} {r['dl_score']:>10.4f} {r['screener_score']:>10.4f}{marker}")

print(f"\nDL threshold: {factor._dl_threshold:.4f} | Stocks above threshold: {len(results_df[results_df['dl_score'] >= factor._dl_threshold])}")
