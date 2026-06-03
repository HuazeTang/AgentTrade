"""Run factor-based recommendations on 2026-05-29 using the save-branch factor framework."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from data.cache import read_daily
from factor.registry import registry

# Import all factors to register them
import factor.factors  # noqa: F401 — triggers @register_factor decorators

# ── Load data ──
load_start = date(2019, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading daily badj data...")
daily = read_daily(load_start, load_end, prefix="daily_badj")

# Fix pre_close (same as before)
if "pre_close" in daily.columns:
    pre_nan = daily["pre_close"].isna().sum()
    if pre_nan > 0:
        logging.info("Fixing %d NaN pre_close values...", pre_nan)
        close = daily["close"].unstack()
        pc = close.shift(1)
        fixed = daily["pre_close"].copy()
        fixed_unstack = fixed.unstack()
        nan_mask = fixed_unstack.isna()
        fixed_unstack[nan_mask] = pc[nan_mask]
        daily = daily.drop(columns=["pre_close"])
        daily["pre_close"] = fixed_unstack.stack()

# ── ST filter ──
info = pd.read_parquet("data/cache/stock_list.parquet")
st_mask = info['name'].str.contains(r'\*?ST', na=False)
st_symbols = set(info.loc[st_mask, 'symbol'].unique())

# ── Compute all registered factors ──
factor_names = registry.list_all()
logging.info("Registered factors (%d): %s", len(factor_names), factor_names)

# Compute each factor individually
logging.info("Computing factors...")
factor_series = {}
for name in factor_names:
    try:
        cls = registry.get(name)
        f = cls()
        ser = f.compute(daily)
        factor_series[name] = ser
        logging.info("  %s: computed, mean=%.4f, coverage=%.1f%%",
                     name, ser.mean(), ser.notna().mean() * 100)
    except Exception as e:
        logging.warning("  %s: FAILED — %s", name, str(e)[:80])

# ── Build composite on latest date ──
target_date = pd.Timestamp("2026-05-29")
logging.info("Target date: %s", target_date.date())

# Extract values for target date, CS z-score, then average
scores = {}
for name, ser in factor_series.items():
    try:
        vals = ser.xs(target_date, level="trade_date")
    except KeyError:
        continue
    if vals.notna().sum() < 30:
        continue
    # CS z-score (higher = better for all factors — assumes positive IC direction)
    m = vals.mean()
    s = vals.std()
    if s < 1e-8:
        continue
    z = (vals - m) / s
    scores[name] = z

logging.info("Factors with valid values on target date: %d", len(scores))

if not scores:
    logging.error("No factors computed!")
    sys.exit(1)

# Composite: average z-score across all factors
composite = pd.DataFrame(scores).mean(axis=1)
composite.name = "composite"

# ── Print top-15 (ST excluded) ──
name_map = dict(zip(info['symbol'], info['name']))
ranked = composite.sort_values(ascending=False)

# ST breakdown
top15_all = ranked.head(15)
st_count = sum(1 for s in top15_all.index if s in st_symbols)
print(f"\nST in top-15: {st_count}/15")

# Print non-ST
non_st_ranked = ranked[~ranked.index.isin(st_symbols)]
print(f"\n{'='*80}")
print(f"  Factor Composite Top-15 Recommendations for {target_date.date()} (ST excluded)")
print(f"  {len(factor_names)} factors, equal-weighted CS z-score composite")
print(f"{'='*80}")
print(f"{'Rank':<6} {'Symbol':<10} {'Name':<12} {'Score':<10}")
print("-" * 50)
for i, (sym, score) in enumerate(non_st_ranked.head(15).items()):
    name = name_map.get(sym, "?")
    print(f"{i+1:<6} {sym:<10} {name:<12} {score:<10.4f}")

print(f"\nFactor contributions (sample top-3 non-ST stocks):")
for sym in non_st_ranked.head(3).index:
    name = name_map.get(sym, "?")
    print(f"\n  {sym} {name}: composite={composite[sym]:.4f}")
    contribs = []
    for fname in sorted(scores):
        contribs.append((fname, scores[fname].get(sym, np.nan)))
    contribs.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    for fname, val in contribs[:5]:
        print(f"    {fname}: {val:+.4f}")

logging.info("Done.")
