"""Inference with V2 epoch 2 model (SmoothAP, raw OHLCV, 20-day) on latest data."""
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

from dl import DualTowerModel
from dl.derived_features import (
    RAW_OHLCV_COLUMNS,
    build_normalized_feature_cache,
)
from data.cache import read_daily, data_summary

SEQUENCE_LENGTH = 20
CHECKPOINT = "data/models/yaogu_20260529_1646_best.pt"

# ── Data ──
summary = data_summary("daily_badj")
print(f"Data range: {summary['dates'][0]} ~ {summary['dates'][1]}, {summary['symbols']} symbols")

end_date = summary["dates"][1]
start_date = end_date - timedelta(days=120)
print(f"Loading data: {start_date} ~ {end_date}")

daily = read_daily(start_date, end_date, prefix="daily_badj")
nsyms = daily.index.get_level_values("symbol").nunique()
print(f"Loaded {len(daily)} rows, {nsyms} symbols")

# ── Stock info ──
stock_list = pd.read_parquet("data/cache/stock_list.parquet")
stock_info = stock_list.set_index("symbol")[["name", "board"]]

# ── Feature cache (raw OHLCV) ──
print("Building raw OHLCV feature cache...")
cache = build_normalized_feature_cache(daily, raw_ohlcv=True)
feature_cols = [c for c in RAW_OHLCV_COLUMNS if c in cache.columns]
print(f"Feature cache: {len(cache)} rows, {len(feature_cols)} columns")

# ── Load model ──
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
in_features = checkpoint["model_kwargs"]["in_features"]
model = DualTowerModel(in_features=in_features)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print(f"Model loaded: epoch={checkpoint['epoch']}, val_ap={checkpoint['val_ap']:.4f}, features={in_features}")

# ── Latest date ──
close = daily["close"].unstack()
all_dates = sorted(close.index)
latest_date = all_dates[-1]
print(f"\nLatest trading date: {latest_date.date()}")

# ── Run inference on ALL stocks ──
symbols = sorted(daily.index.get_level_values("symbol").unique())
seq_start = len(all_dates) - 1 - SEQUENCE_LENGTH
seq_dates = all_dates[seq_start:-1]
print(f"Sequence: {seq_dates[0].date()} ~ {seq_dates[-1].date()} ({SEQUENCE_LENGTH} days)")

batch_feats = []
batch_syms = []
for sym in symbols:
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

print(f"Valid sequences: {len(batch_feats)}")

X_tensor = torch.from_numpy(np.stack(batch_feats)).to(device)
with torch.no_grad():
    probs = model.predict_proba(X_tensor).cpu().numpy().flatten()

# ── Build results ──
latest_mask = daily.index.get_level_values("trade_date") == latest_date
latest_data = daily[latest_mask].reset_index(level=0, drop=True)

results = []
for sym, prob in zip(batch_syms, probs):
    name = stock_info.loc[sym, "name"] if sym in stock_info.index else "?"
    board = "?"
    price = np.nan
    pct_chg = np.nan
    if sym in latest_data.index:
        row = latest_data.loc[sym]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if isinstance(row, pd.Series):
            board = row.get("board", "?")
            if isinstance(board, pd.Series):
                board = str(board.iloc[0])
            price = row.get("close", np.nan)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0])
            pre_close = row.get("pre_close", np.nan)
            if isinstance(pre_close, pd.Series):
                pre_close = pre_close.iloc[0]
            if pd.notna(pre_close) and pd.notna(price) and float(pre_close) != 0:
                pct_chg = (float(price) - float(pre_close)) / float(pre_close) * 100

    results.append({
        "symbol": sym, "name": name, "board": str(board),
        "price": float(price) if pd.notna(price) else np.nan,
        "pct_chg": pct_chg,
        "score": float(prob),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("score", ascending=False)

BOARD_NAMES = {"main_board": "主板", "chinext": "创业板", "star_market": "科创板", "beijing": "北交所"}
results_df["board_cn"] = results_df["board"].map(BOARD_NAMES).fillna(results_df["board"])

# ── Output ──
top_n = 40
print(f"\n{'='*100}")
print(f"Top {top_n} V2 Yaogu Candidates — {latest_date.date()} (SmoothAP, raw OHLCV, 20-day)")
print(f"{'='*100}")
print(f"{'代码':<8} {'名称':<10} {'板块':<6} {'现价':>8} {'涨跌幅':>8} {'Score':>10}")
print(f"{'-'*100}")
for _, r in results_df.head(top_n).iterrows():
    pct_str = f"{r['pct_chg']:+.2f}%" if pd.notna(r['pct_chg']) else "-"
    print(f"{r['symbol']:<8} {r['name']:<10} {r['board_cn']:<6} {r['price']:>8.2f} {pct_str:>8} {r['score']:>10.4f}")

# ── Stats ──
print(f"\nScore distribution: min={results_df['score'].min():.4f} "
      f"Q25={results_df['score'].quantile(0.25):.4f} "
      f"median={results_df['score'].median():.4f} "
      f"Q75={results_df['score'].quantile(0.75):.4f} "
      f"max={results_df['score'].max():.4f}")

for thr in [0.3, 0.5, 0.7]:
    n = (results_df['score'] >= thr).sum()
    print(f"Score >= {thr:.1f}: {n} stocks")

# ── Board breakdown ──
print(f"\nBoard breakdown (top 20):")
print(results_df.head(20)["board_cn"].value_counts().to_string())
