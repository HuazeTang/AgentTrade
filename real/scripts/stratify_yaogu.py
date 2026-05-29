"""Stratification backtest comparing baseline vs baseline+yaogu factors.

Usage: python scripts/stratify_yaogu.py
Output: data/results/stratify_yaogu.png
"""

import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from datetime import date, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

from run_agent_simulation import (
    AgentSimulation, get_trading_days, BASELINE_FACTORS,
    DISABLED_FACTORS, FactorEngine, read_daily,
)
from factor.validation import compute_rank_ic
from discovery.expr import Expr
from discovery.compiler import compile_expr

# ── Config ──
BACKTEST_START = date(2024, 1, 1)  # Focus on recent period where baseline decayed
BACKTEST_END = date(2026, 5, 25)
N_QUANTILES = 10
ROLLING_WINDOW_DAYS = 252
RECALIBRATE_MONTHS = 3
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YAOGU_JSON = Path("data/results/yaogu_factors.json")

print("=" * 60)
print("  Yaogu Factor Stratification Test")
print(f"  Period: {BACKTEST_START} ~ {BACKTEST_END}")
print("=" * 60)

# ── 1. Load data ──
print("\n[1/6] Loading data...")
sim = AgentSimulation(start=BACKTEST_START, end=BACKTEST_END, mode="factor")
sim._trading_days = get_trading_days(sim.start, sim.end)
sim._daily_cache = read_daily(BACKTEST_START - timedelta(days=400), sim.end)

all_syms = sorted(sim._daily_cache.index.get_level_values("symbol").unique().tolist())
symbols = sim._generate_stock_pool(all_syms)
pool_mask = sim._daily_cache.index.get_level_values("symbol").isin(symbols)
sim._daily_cache = sim._daily_cache[pool_mask]
sim._daily_cache = sim._add_derived_features(sim._daily_cache)
print(f"  Symbols: {len(symbols)}, rows: {len(sim._daily_cache)}")

# ── 2. Compute baseline factors ──
print("[2/6] Computing baseline factors...")
baseline_factors = [f for f in BASELINE_FACTORS if f not in DISABLED_FACTORS]

engine = FactorEngine()
sim._factor_df = engine.compute(baseline_factors, sim._daily_cache)
if sim._factor_df.index.names[1] is None:
    sim._factor_df.index = sim._factor_df.index.set_names("symbol", level=1)

shifted = sim._factor_df.unstack().shift(1).stack(future_stack=True)
sim._factor_df = shifted.reorder_levels(["trade_date", "symbol"]).sort_index()
print(f"  Baseline factors: {len(baseline_factors)}")

# ── 3. Load and compute yaogu factors ──
print("[3/6] Loading yaogu factors...")
yaogu_names = []
if YAOGU_JSON.exists():
    with open(YAOGU_JSON) as f:
        yaogu_data = json.load(f)
    for fmeta in yaogu_data.get("yaogu_factors", []):
        tree = Expr.from_dict(fmeta["expression"])
        compile_expr(tree, factor_name=fmeta["name"], category=fmeta["category"], register=True)
        yaogu_names.append(fmeta["name"])
    print(f"  Loaded {len(yaogu_names)} yaogu factors: {yaogu_names}")

    # Compute yaogu factors on same data
    yaogu_df = engine.compute(yaogu_names, sim._daily_cache)
    if yaogu_df.index.names[1] is None:
        yaogu_df.index = yaogu_df.index.set_names("symbol", level=1)
    shifted_yg = yaogu_df.unstack().shift(1).stack(future_stack=True)
    yaogu_df = shifted_yg.reorder_levels(["trade_date", "symbol"]).sort_index()

    # Add yaogu columns to factor_df
    sim._factor_df = sim._factor_df.join(yaogu_df, how="left")
    all_factors = baseline_factors + yaogu_names
else:
    print("  No yaogu factors found — running baseline only.")
    all_factors = baseline_factors

print(f"  Total factors: {len(all_factors)}")

# ── 4. Rolling calibration ──
print("[4/6] Rolling-window calibration...")

all_trade_dates = sorted(sim._factor_df.index.get_level_values("trade_date").unique())
trade_dates_dt = [td.date() for td in all_trade_dates]

close = sim._daily_cache["close"].unstack()
fwd_ret_1d = close.pct_change().shift(-1).stack()
fwd_ret_1d.name = "fwd_ret_1d"


def calibrate_weights(factor_df, fwd_ret, cal_dates):
    cal_mask = factor_df.index.get_level_values("trade_date").isin(cal_dates)
    cal_factors = factor_df[cal_mask]
    weights = {}
    for fname in factor_df.columns:
        factor_vals = cal_factors[fname]
        common_idx = factor_vals.dropna().index.intersection(fwd_ret.dropna().index)
        if len(common_idx) < 50:
            continue
        ic = compute_rank_ic(factor_vals.loc[common_idx], fwd_ret.loc[common_idx])
        ic_mean = ic.mean()
        if abs(ic_mean) >= 0.005:
            weights[fname] = ic_mean
    total_w = sum(abs(v) for v in weights.values())
    if total_w <= 0:
        n = max(len(factor_df.columns), 1)
        return {f: 1.0 / n for f in factor_df.columns}
    return {k: v / total_w for k, v in weights.items()}


# Build recalibration schedule
cal_schedule = defaultdict(list)
current_cal_idx = 0
last_cal_month = None
for d in trade_dates_dt:
    month_key = (d.year, d.month)
    if last_cal_month is None or (
        d.year * 12 + d.month - 1 >= last_cal_month[0] * 12 + last_cal_month[1] - 1 + RECALIBRATE_MONTHS
    ):
        current_cal_idx += 1
        last_cal_month = month_key
    cal_schedule[current_cal_idx].append(d)

# Run: baseline-only weights AND full weights
print("  Calibrating baseline-only weights...")
weights_baseline = {}
for cal_idx in sorted(cal_schedule.keys()):
    period_dates = cal_schedule[cal_idx]
    period_start = period_dates[0]
    cal_start_idx = max(0, trade_dates_dt.index(period_start) - ROLLING_WINDOW_DAYS)
    cal_dates = trade_dates_dt[cal_start_idx:trade_dates_dt.index(period_start)]
    cal_dates_pd = [pd.Timestamp(d) for d in cal_dates]

    baseline_df = sim._factor_df[baseline_factors]
    if len(cal_dates) >= 60:
        weights_baseline[cal_idx] = calibrate_weights(baseline_df, fwd_ret_1d, cal_dates_pd)
    else:
        n = len(baseline_factors)
        weights_baseline[cal_idx] = {f: 1.0 / n for f in baseline_factors}

print("  Calibrating baseline+yaogu weights...")
weights_full = {}
for cal_idx in sorted(cal_schedule.keys()):
    period_dates = cal_schedule[cal_idx]
    period_start = period_dates[0]
    cal_start_idx = max(0, trade_dates_dt.index(period_start) - ROLLING_WINDOW_DAYS)
    cal_dates = trade_dates_dt[cal_start_idx:trade_dates_dt.index(period_start)]
    cal_dates_pd = [pd.Timestamp(d) for d in cal_dates]

    if len(cal_dates) >= 60:
        weights_full[cal_idx] = calibrate_weights(sim._factor_df, fwd_ret_1d, cal_dates_pd)
    else:
        n = len(all_factors)
        weights_full[cal_idx] = {f: 1.0 / n for f in all_factors}

# Print yaogu weight evolution
print("\n  Yaogu factor weights over time:")
for cal_idx in sorted(cal_schedule.keys()):
    if cal_idx % 4 == 0 and cal_idx in weights_full:
        w = weights_full[cal_idx]
        yg_weights = {k: v for k, v in w.items() if k in yaogu_names}
        if yg_weights:
            period_start = cal_schedule[cal_idx][0]
            yg_str = ", ".join(f"{k.split('_')[1][:12]}: {v:+.4f}" for k, v in sorted(yg_weights.items(), key=lambda x: -abs(x[1])))
            print(f"    {period_start}: {yg_str}")


# ── 5. Daily cross-section (run both variants) ──
print("[5/6] Computing daily quantile returns (both variants)...")

def compute_daily_quantile_returns(factor_df, weights_history):
    """Compute quantile returns given factor_df and per-period weights."""
    quantile_returns = {q: [] for q in range(N_QUANTILES)}
    dates_used = []

    for i, td in enumerate(trade_dates_dt):
        cal_idx = None
        for cidx, dlist in cal_schedule.items():
            if td in dlist:
                cal_idx = cidx
                break
        if cal_idx is None:
            continue

        weights = weights_history.get(cal_idx)
        if weights is None:
            continue

        try:
            day_factors = factor_df.xs(pd.Timestamp(td), level="trade_date")
        except KeyError:
            continue

        composite = pd.Series(0.0, index=day_factors.index)
        for fname, weight in weights.items():
            if fname not in day_factors.columns:
                continue
            col = day_factors[fname].dropna()
            if len(col) < 5:
                continue
            ranked = col.rank(pct=True)
            composite = composite.add(ranked * weight, fill_value=0.0)

        composite = composite[composite != 0]
        if composite.empty or len(composite) < 100:
            continue

        next_td = td + timedelta(days=1)
        next_td_data = sim._daily_cache[
            sim._daily_cache.index.get_level_values("trade_date") == pd.Timestamp(next_td)
        ]
        if next_td_data.empty:
            next_td2 = td + timedelta(days=2)
            next_td_data = sim._daily_cache[
                sim._daily_cache.index.get_level_values("trade_date") == pd.Timestamp(next_td2)
            ]
        if next_td_data.empty:
            continue

        next_close = next_td_data["close"].groupby("symbol").last()
        td_close = sim._daily_cache[
            sim._daily_cache.index.get_level_values("trade_date") == pd.Timestamp(td)
        ]["close"].groupby("symbol").last()

        common = composite.index.intersection(next_close.index).intersection(td_close.index)
        if len(common) < 100:
            continue

        fwd_ret = (next_close[common] / td_close[common]) - 1.0
        scores = composite[common].rank(pct=True)
        for q in range(N_QUANTILES):
            lo = q / N_QUANTILES
            hi = (q + 1) / N_QUANTILES
            mask = (scores > lo) & (scores <= hi)
            if mask.sum() > 0:
                quantile_returns[q].append(fwd_ret[mask].mean())
            else:
                quantile_returns[q].append(0.0)
        dates_used.append(td)

    return quantile_returns, dates_used


# Baseline only
baseline_df = sim._factor_df[baseline_factors]
qr_base, dates_base = compute_daily_quantile_returns(baseline_df, weights_baseline)
print(f"  Baseline: {len(dates_base)} valid days")

# Baseline + yaogu
qr_full, dates_full = compute_daily_quantile_returns(sim._factor_df, weights_full)
print(f"  Baseline+yaogu: {len(dates_full)} valid days")

# ── 6. Plot comparison ──
print("[6/6] Plotting...")

# Compute cumulative long-short for both
ls_base = np.array(qr_base[N_QUANTILES - 1]) - np.array(qr_base[0])
cum_ls_base = np.cumprod(1.0 + ls_base) - 1.0

ls_full = np.array(qr_full[N_QUANTILES - 1]) - np.array(qr_full[0])
cum_ls_full = np.cumprod(1.0 + ls_full) - 1.0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ── Top-left: Baseline quantile returns ──
ax = axes[0, 0]
cmap = plt.cm.RdYlGn
for q in range(N_QUANTILES):
    cum = np.cumprod(1.0 + np.array(qr_base[q])) - 1.0
    color = cmap(q / (N_QUANTILES - 1))
    alpha = 0.4 + 0.6 * (q / (N_QUANTILES - 1))
    ax.plot(dates_base, cum * 100, color=color, alpha=alpha, linewidth=1.0)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.set_title("Baseline Only — Quantile Returns")
ax.set_ylabel("Cumulative Return (%)")
ax.grid(True, alpha=0.3)

# ── Top-right: Baseline+Yaogu quantile returns ──
ax = axes[0, 1]
for q in range(N_QUANTILES):
    cum = np.cumprod(1.0 + np.array(qr_full[q])) - 1.0
    color = cmap(q / (N_QUANTILES - 1))
    alpha = 0.4 + 0.6 * (q / (N_QUANTILES - 1))
    ax.plot(dates_full, cum * 100, color=color, alpha=alpha, linewidth=1.0)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.set_title("Baseline + Yaogu — Quantile Returns")
ax.set_ylabel("Cumulative Return (%)")
ax.grid(True, alpha=0.3)

# ── Bottom-left: Long-Short comparison ──
ax = axes[1, 0]
ax.plot(dates_base, cum_ls_base * 100, color="blue", linewidth=1.5, label="Baseline")
ax.plot(dates_full, cum_ls_full * 100, color="red", linewidth=1.5, label="Baseline+Yaogu")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.set_title("Long-Short (Q10 − Q1): Baseline vs Baseline+Yaogu")
ax.set_ylabel("Cumulative Return (%)")
ax.legend()
ax.grid(True, alpha=0.3)

# ── Bottom-right: Stats ──
ax = axes[1, 1]
ax.axis("off")

n_years = (dates_base[-1] - dates_base[0]).days / 365.25

# Baseline stats
ls_base_sharpe = np.sqrt(252) * ls_base.mean() / ls_base.std() if ls_base.std() > 0 else 0
base_ann = (1 + cum_ls_base[-1]) ** (1 / n_years) - 1 if cum_ls_base[-1] > -1 else -1

# Full stats
ls_full_sharpe = np.sqrt(252) * ls_full.mean() / ls_full.std() if ls_full.std() > 0 else 0
full_ann = (1 + cum_ls_full[-1]) ** (1 / n_years) - 1 if cum_ls_full[-1] > -1 else -1

stats_lines = [
    "=== Long-Short Performance ===",
    "",
    "BASELINE ONLY:",
    f"  Cumulative: {cum_ls_base[-1]*100:+.1f}%",
    f"  Annualized: {base_ann*100:+.1f}%",
    f"  Sharpe: {ls_base_sharpe:.2f}",
    f"  Days: {len(dates_base)}",
    "",
    "BASELINE + YAOGU:",
    f"  Cumulative: {cum_ls_full[-1]*100:+.1f}%",
    f"  Annualized: {full_ann*100:+.1f}%",
    f"  Sharpe: {ls_full_sharpe:.2f}",
    f"  Days: {len(dates_full)}",
    "",
    f"  Improvement: {(cum_ls_full[-1] - cum_ls_base[-1])*100:+.1f}% cum",
    f"               {(full_ann - base_ann)*100:+.1f}% ann",
    "",
    "=== Yaogu Factor Weights (final period) ===",
]

# Show yaogu weights in final calibration period
final_cal_idx = max(weights_full.keys())
if final_cal_idx in weights_full:
    w = weights_full[final_cal_idx]
    yg_w = {k: v for k, v in w.items() if k in yaogu_names}
    for k, v in sorted(yg_w.items(), key=lambda x: -abs(x[1])):
        short_name = k.replace("yaogu_", "")[:30]
        stats_lines.append(f"  {short_name}: {v:+.4f}")
    if not yg_w:
        stats_lines.append("  (yaogu factors excluded — below noise floor)")

ax.text(0.05, 0.95, "\n".join(stats_lines), transform=ax.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

fig.suptitle("Yaogu Factor Impact — Stratification Backtest (Rolling Calibration)", fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = OUT_DIR / "stratify_yaogu.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# ── Annual comparison ──
print("\nAnnual Long-Short Comparison:")
print(f"{'Year':<6} {'Baseline':>10} {'+Yaogu':>10} {'Δ':>10}")
for year in range(2024, 2027):
    mask = [d.year == year for d in dates_base]
    if sum(mask) > 0:
        y_base = np.prod(1.0 + ls_base[mask]) - 1
        y_full = np.prod(1.0 + ls_full[mask]) - 1
        print(f"{year:<6} {y_base*100:>9.1f}% {y_full*100:>9.1f}% {(y_full-y_base)*100:>+9.1f}%")
