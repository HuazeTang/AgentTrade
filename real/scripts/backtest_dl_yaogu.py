"""Forward-trading backtest for DL yaogu model.

For each trading day: compute probabilities → filter above threshold →
track forward N-day returns → aggregate P&L.

Usage: python scripts/backtest_dl_yaogu.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from datetime import date, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from run_agent_simulation import (
    AgentSimulation, get_trading_days, read_daily,
)
from dl import DualTowerModel
from dl.derived_features import (
    DERIVED_FEATURE_COLUMNS,
    build_normalized_feature_cache,
)
from dl.screener import YaoguScreener

# ── Config ──
BACKTEST_START = date(2024, 1, 1)
BACKTEST_END = date(2026, 5, 25)
MODEL_PATH = "data/models/yaogu_best.pt"
SCREENER_PATH = "data/models/yaogu_screener.joblib"
SEQUENCE_LENGTH = 60
FORWARD_DAYS = 10
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_date(val):
    if hasattr(val, "date"):
        val = val.date()
    if hasattr(val, "date"):
        return val.date()
    if isinstance(val, pd.Timestamp):
        return val.date()
    return val
TWO_STAGE = True  # use two-stage pipeline

print("=" * 60)
print("  DL Yaogu Forward-Trading Backtest")
print(f"  Period: {BACKTEST_START} ~ {BACKTEST_END}")
print(f"  Hold: {FORWARD_DAYS} days")
print("=" * 60)

# ── 1. Load model(s) ──
print("\n[1/5] Loading model(s)...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = DualTowerModel(**checkpoint.get("model_kwargs", {}))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
threshold = checkpoint.get("threshold", 0.5)
print(f"  DL threshold: {threshold:.3f}, params: {sum(p.numel() for p in model.parameters()):,}")

screener = None
if TWO_STAGE:
    import os
    if os.path.exists(SCREENER_PATH):
        screener = YaoguScreener.load(SCREENER_PATH)
        print(f"  Screener: threshold={screener.threshold:.4f}, {len(screener.feature_cols)} features")
    else:
        print("  WARNING: Two-stage mode but screener not found — running single-stage")

# ── 2. Load data ──
print("[2/5] Loading data...")
sim = AgentSimulation(start=BACKTEST_START, end=BACKTEST_END, mode="factor")
sim._trading_days = get_trading_days(sim.start, sim.end)
sim._daily_cache = read_daily(BACKTEST_START - timedelta(days=400), BACKTEST_END)

all_syms = sorted(sim._daily_cache.index.get_level_values("symbol").unique().tolist())
symbols = sim._generate_stock_pool(all_syms)
pool_mask = sim._daily_cache.index.get_level_values("symbol").isin(symbols)
sim._daily_cache = sim._daily_cache[pool_mask]
print(f"  Symbols: {len(symbols)}, rows: {len(sim._daily_cache)}")

# Unstack for sequence access
close = sim._daily_cache["close"].unstack()
dates = sorted(close.index)
symbols_in_data = [s for s in symbols if s in close.columns]

# Pre-compute forward max returns for evaluation
print("[3/5] Computing forward returns...")
from llm.yaogu.case_extractor import YaoguCaseExtractor
extractor = YaoguCaseExtractor(forward_window=FORWARD_DAYS, min_cum_ret=0.0, min_limit_up=0)
fwd_max_ret = extractor._compute_forward_max_return(close, dates, FORWARD_DAYS, symbols_in_data)

# ── 4. Daily inference + trading ──
print(f"[4/5] Running daily inference ({len(dates)} days)...")

# Build normalized feature cache (shared)
print("[3/5] Building normalized feature cache...")
feature_cache = build_normalized_feature_cache(sim._daily_cache)
feature_cols = [c for c in DERIVED_FEATURE_COLUMNS if c in feature_cache.columns]
print(f"  Cache: {len(feature_cache)} rows, {len(feature_cols)} features")

trades = []
daily_recs = []
dates_processed = 0
screened_total = 0
screened_kept = 0

start_i = SEQUENCE_LENGTH
end_i = len(dates) - FORWARD_DAYS - 1

for i in range(start_i, end_i):
    td = dates[i]
    td_date = _to_date(td)
    if td_date < BACKTEST_START:
        continue

    seq_start = i - SEQUENCE_LENGTH
    seq_dates = dates[seq_start:i]

    # Determine which symbols to evaluate
    eval_symbols = symbols_in_data

    if screener is not None:
        # Stage 1: screener filter
        screener_scores = screener.score_day(feature_cache, pd.Timestamp(td), symbols_in_data)
        candidates = screener_scores[screener_scores >= screener.threshold]
        screened_total += len(symbols_in_data)
        screened_kept += len(candidates)
        if candidates.empty:
            dates_processed += 1
            continue
        eval_symbols = list(candidates.index)

    batch_feats = []
    batch_symbols = []

    for sym in eval_symbols:
        try:
            mask = (
                feature_cache.index.get_level_values("trade_date").isin(seq_dates) &
                (feature_cache.index.get_level_values("symbol") == sym)
            )
            rows = feature_cache.loc[mask][feature_cols]
        except (KeyError, IndexError):
            continue

        if len(rows) < SEQUENCE_LENGTH:
            continue

        rows = rows.sort_index(level="trade_date")
        X = rows.values[:SEQUENCE_LENGTH].astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        batch_feats.append(X)
        batch_symbols.append(sym)

    if not batch_feats:
        dates_processed += 1
        continue

    # Batch inference
    X_tensor = torch.from_numpy(np.stack(batch_feats)).to(device)
    with torch.no_grad():
        probs = model.predict_proba(X_tensor).cpu().numpy()

    # Filter above threshold
    day_hits = []
    for sym, prob in zip(batch_symbols, probs):
        if prob >= threshold:
            fwd_ret = fwd_max_ret.iloc[i][sym] if sym in fwd_max_ret.columns else 0
            is_hit = fwd_ret >= 0.30
            trades.append({
                "date": td_date, "symbol": sym, "prob": float(prob),
                "fwd_max_ret": float(fwd_ret) if not pd.isna(fwd_ret) else 0,
                "hit": is_hit,
            })
            day_hits.append({"symbol": sym, "prob": float(prob),
                             "fwd_ret": float(fwd_ret) if not pd.isna(fwd_ret) else 0})

    if day_hits:
        daily_recs.append({"date": td_date, "count": len(day_hits), "recs": day_hits})

    dates_processed += 1
    if dates_processed % 100 == 0:
        if screener:
            print(f"  {dates_processed} days, {len(trades)} trades "
                  f"(screener: {screened_kept}/{screened_total} = "
                  f"{screened_kept/max(screened_total,1)*100:.0f}%)")
        else:
            print(f"  {dates_processed} days processed, {len(trades)} trades so far")

print(f"  Total: {dates_processed} days, {len(trades)} trades, {len(daily_recs)} days with recommendations")
if screener is not None:
    reduction = (1 - screened_kept / max(screened_total, 1)) * 100
    print(f"  Screener: {screened_kept}/{screened_total} kept ({reduction:.0f}% reduction)")

# ── 5. Results ──
print("\n[5/5] Results:")
print("=" * 60)

if not trades:
    print("  NO TRADES generated — threshold too high or model not firing.")
    exit()

trades_df = pd.DataFrame(trades)

# Overall stats
total_trades = len(trades_df)
hits = trades_df["hit"].sum()
precision = hits / total_trades if total_trades > 0 else 0
avg_ret = trades_df["fwd_max_ret"].mean()
avg_ret_hit = trades_df[trades_df["hit"]]["fwd_max_ret"].mean()
avg_ret_miss = trades_df[~trades_df["hit"]]["fwd_max_ret"].mean()
median_ret = trades_df["fwd_max_ret"].median()

print(f"\n  Total trades: {total_trades}")
print(f"  Hits (forward max ret >= 30%): {hits} ({precision*100:.1f}%)")
print(f"  Average forward max return: {avg_ret*100:+.1f}%")
print(f"    When hit:  {avg_ret_hit*100:+.1f}%")
print(f"    When miss: {avg_ret_miss*100:+.1f}%")
print(f"  Median forward max return: {median_ret*100:+.1f}%")

# Daily stats
days_with_recs = len(daily_recs)
total_days = dates_processed
print(f"\n  Days with >=1 recommendation: {days_with_recs}/{total_days} ({days_with_recs/total_days*100:.1f}%)")
avg_daily_count = np.mean([d["count"] for d in daily_recs]) if daily_recs else 0
print(f"  Average recommendations per active day: {avg_daily_count:.1f}")

# Monthly breakdown
trades_df["month"] = pd.to_datetime(trades_df["date"]).dt.to_period("M")
monthly = trades_df.groupby("month").agg(
    trades=("hit", "count"),
    hits=("hit", "sum"),
    avg_ret=("fwd_max_ret", "mean"),
).assign(precision=lambda x: x["hits"] / x["trades"])

print("\n  Monthly Breakdown:")
print(f"  {'Month':<10} {'Trades':>7} {'Hits':>5} {'Prec':>7} {'AvgRet':>8}")
for m, row in monthly.iterrows():
    print(f"  {str(m):<10} {int(row['trades']):>7} {int(row['hits']):>5} {row['precision']*100:>6.1f}% {row['avg_ret']*100:>7.1f}%")

# P&L simulation: buy each recommendation equally, hold for FORWARD_DAYS
# Simple: daily P&L = mean of all active positions' forward returns
daily_pnl = []
for rec_day in daily_recs:
    returns = [r["fwd_ret"] for r in rec_day["recs"]]
    daily_pnl.append(np.mean(returns))

if daily_pnl:
    cum_pnl = np.cumprod(1.0 + np.array(daily_pnl)) - 1.0
    sharpe = np.sqrt(252) * np.mean(daily_pnl) / np.std(daily_pnl) if np.std(daily_pnl) > 0 else 0
    print(f"\n  Cumulative P&L (equal weight, no compounding across days): {cum_pnl[-1]*100:+.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")

# Distribution of returns
print(f"\n  Return Distribution:")
for bucket in [(-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 10.0)]:
    count = ((trades_df["fwd_max_ret"] >= bucket[0]) & (trades_df["fwd_max_ret"] < bucket[1])).sum()
    pct = count / total_trades * 100
    print(f"    {bucket[0]:+.0%} ~ {bucket[1]:+.0%}: {count:>4} ({pct:5.1f}%)")

# Save detailed trade log
trades_df.to_csv(OUT_DIR / "dl_yaogu_trades.csv", index=False)
print(f"\n  Trade log saved: {OUT_DIR / 'dl_yaogu_trades.csv'}")
