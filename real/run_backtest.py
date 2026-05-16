"""Complete end-to-end backtest with real A-share data via baostock.

Pipeline: data ingest → factor compute → cross-sectional strategy → backtest → metrics → plots
"""

import logging
import warnings
from datetime import date, datetime
from pathlib import Path

import config.chart_style  # noqa: F401 — CJK fonts + Agg backend (must preceed pyplot)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
from data.sources.baostock import BaoStockSource
from data.pipeline import ingest_daily
from data.calendar import get_trading_days
import factor.factors as _  # register all factors
from factor.engine import FactorEngine
from strategy.cross_sectional import CrossSectionalStrategy

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL_COUNT = 50
START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 5, 14)
INITIAL_CASH = 1_000_000
TOP_QUANTILE = 0.2
COMMISSION = 0.00025
STAMP_TAX = 0.001

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("A-Share Quantitative Backtest — Full Pipeline")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Ingest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Data Ingest via baostock ...")
source = BaoStockSource(rate_limit=0.05)
all_stocks = source.list_stocks()

# Pick a diverse set of non-ST stocks from different boards
candidates = all_stocks[~all_stocks["is_st"]].copy()
# Include main_board, chinext, star_market
main = candidates[candidates["board"] == "main_board"]["symbol"].head(30)
chinext = candidates[candidates["board"] == "chinext"]["symbol"].head(15)
star = candidates[candidates["board"] == "star_market"]["symbol"].head(5)
symbols = pd.concat([main, chinext, star]).tolist()

print(f"  Universe: {len(symbols)} stocks ({len(main)} main + {len(chinext)} chinext + {len(star)} star)")
print(f"  Period: {START_DATE} ~ {END_DATE}")

raw_data = ingest_daily(symbols, START_DATE, END_DATE, source=source)
source.close()

n_rows = len(raw_data)
n_syms = raw_data.index.get_level_values("symbol").nunique()
date_range = raw_data.index.get_level_values("trade_date")
print(f"  Fetched: {n_rows:,} rows, {n_syms} symbols, {date_range.min().date()} ~ {date_range.max().date()}")

if raw_data.empty:
    print("  ERROR: No data fetched!")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Factor Computation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Factor Computation ...")
engine = FactorEngine()
factor_names = ["momentum_1m", "momentum_3m", "reversal_5d", "volatility_20d", "turnover_20d"]
factor_df = engine.compute(factor_names, raw_data)
print(f"  Computed: {len(factor_names)} factors, {factor_df.shape[0]:,} values")

# Merge factors with price data
merged = raw_data.join(factor_df, how="left")
# Drop rows without all factors (early period warm-up)
merged = merged.dropna(subset=factor_names)
print(f"  After NaN drop: {len(merged):,} rows (warm-up period removed)")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Factor Validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Factor Validation ...")
from scipy import stats

print(f"  {'Factor':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Sharpe':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for fname in factor_names:
    ser = factor_df[fname].dropna()
    sharpe = ser.mean() / ser.std() * (252**0.5) if ser.std() > 0 else 0
    print(f"  {fname:<20} {ser.mean():>8.4f} {ser.std():>8.4f} {ser.min():>8.4f} {ser.max():>8.4f} {sharpe:>8.3f}")

# Correlation
print(f"\n  Factor Correlation Matrix:")
corr = factor_df.corr()
for line in corr.round(3).to_string().split("\n"):
    print(f"  {line}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Strategy & Backtest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Running Backtest ...")

# Use reversal_5d as primary signal (positive IC in this period)
strategy = CrossSectionalStrategy(
    signal_col="reversal_5d",
    top_quantile=TOP_QUANTILE,
    bottom_quantile=0.0,
    long_only=True,
    n_positions=max(5, int(SYMBOL_COUNT * TOP_QUANTILE)),
)

trading_days = merged.index.get_level_values("trade_date").unique().sort_values()
cash = INITIAL_CASH
positions: dict[str, int] = {}  # symbol -> shares held
equity_curve: list[dict] = []
trades_log: list[dict] = []

print(f"  Trading days: {len(trading_days)}")

for i, today in enumerate(trading_days):
    # Get today's data
    try:
        today_data = merged.xs(today, level="trade_date")
    except KeyError:
        continue
    if today_data.empty:
        continue

    # Previous close prices for valuation (start of day)
    if "pre_close" in today_data.columns:
        prices_start = today_data["pre_close"]
    else:
        prices_start = today_data["close"]

    prices_close = today_data["close"]

    # Mark-to-market at start of day
    total_equity = cash
    for sym, shares in list(positions.items()):
        if sym in prices_start.index and pd.notna(prices_start[sym]):
            total_equity += shares * prices_start[sym]

    # Use yesterday's data for signal (avoid look-ahead bias)
    # Find the previous trading day in our data
    prev_date = None
    for pd_ in trading_days[:i][::-1]:  # scan backwards from today
        if pd_ < today:
            prev_date = pd_
            break

    if prev_date is not None:
        try:
            signal_data = merged.xs(prev_date, level="trade_date")
        except KeyError:
            signal_data = today_data.copy()
    else:
        signal_data = today_data.copy()

    # Strategy generates weights
    universe = today_data.index.tolist()
    try:
        weights = strategy.generate_weights(
            date=today,
            universe=universe,
            data=signal_data,
            prices=prices_close,
            current_positions={sym: float(qty) for sym, qty in positions.items()},
            cash=cash,
        )
    except Exception:
        weights = pd.Series(dtype=float)

    # Execute: sell positions that are no longer in target, buy new ones
    target_symbols = set(weights.index)
    current_symbols = set(positions.keys())

    day_trades = 0
    turnover_value = 0.0

    # Sell symbols not in target
    for sym in current_symbols - target_symbols:
        shares = positions.pop(sym)
        price = prices_start.get(sym, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        proceeds = shares * price
        cost = proceeds * (COMMISSION + STAMP_TAX)
        cash += proceeds - cost
        day_trades += 1
        turnover_value += proceeds

    # Adjust existing positions toward target weights
    target_capital = total_equity * 0.90  # invest 90%, keep 10% buffer
    for sym in target_symbols & current_symbols:
        target_w = weights.get(sym, 0)
        target_value = target_capital * target_w
        current_shares = positions.get(sym, 0)
        current_value = current_shares * prices_start.get(sym, 0)
        diff_value = target_value - current_value

        if abs(diff_value) < 500:  # skip tiny adjustments
            continue

        if diff_value > 0:  # need to buy more
            price = prices_start.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            buy_shares = int(diff_value / price / 100) * 100
            if buy_shares > 0:
                cost = buy_shares * price * (1 + COMMISSION)
                if cost <= cash:
                    cash -= cost
                    positions[sym] = positions.get(sym, 0) + buy_shares
                    day_trades += 1
                    turnover_value += buy_shares * price
        else:  # need to sell some
            price = prices_start.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            sell_shares = min(int(-diff_value / price / 100) * 100, positions.get(sym, 0))
            if sell_shares > 0:
                proceeds = sell_shares * price
                cost = proceeds * (COMMISSION + STAMP_TAX)
                cash += proceeds - cost
                positions[sym] -= sell_shares
                if positions[sym] == 0:
                    del positions[sym]
                day_trades += 1
                turnover_value += proceeds

    # Buy new positions
    for sym in target_symbols - current_symbols:
        target_w = weights.get(sym, 0)
        target_value = target_capital * target_w
        price = prices_start.get(sym, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        buy_shares = int(target_value / price / 100) * 100
        if buy_shares > 0:
            cost = buy_shares * price * (1 + COMMISSION)
            if cost <= cash:
                cash -= cost
                positions[sym] = buy_shares
                day_trades += 1
                turnover_value += buy_shares * price

    # End-of-day mark-to-market
    eod_equity = cash
    for sym, shares in list(positions.items()):
        close_price = prices_close.get(sym, np.nan)
        if pd.notna(close_price) and close_price > 0:
            eod_equity += shares * close_price

    equity_curve.append({
        "date": today,
        "equity": eod_equity,
        "cash": cash,
        "positions_value": eod_equity - cash,
        "n_positions": len(positions),
        "trades": day_trades,
        "turnover": turnover_value,
    })

# ── Build equity DataFrame ──
eq_df = pd.DataFrame(equity_curve)
eq_df["date"] = pd.to_datetime(eq_df["date"])
eq_df = eq_df.set_index("date")
eq_df["daily_return"] = eq_df["equity"].pct_change()
eq_df["cum_return"] = (1 + eq_df["daily_return"].fillna(0)).cumprod() - 1
eq_df["equity_peak"] = eq_df["equity"].expanding().max()
eq_df["drawdown"] = (eq_df["equity"] - eq_df["equity_peak"]) / eq_df["equity_peak"]

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Performance Metrics ...")

final_equity = eq_df["equity"].iloc[-1]
total_return = final_equity / INITIAL_CASH - 1
n_years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
ann_return = (final_equity / INITIAL_CASH) ** (1 / n_years) - 1 if n_years > 0 else 0
daily_returns = eq_df["daily_return"].dropna()
ann_vol = daily_returns.std() * (252**0.5)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0
dn_std = daily_returns[daily_returns < 0].std() * (252**0.5)
sortino = ann_return / dn_std if dn_std > 0 else 0
max_dd = eq_df["drawdown"].min()
calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
win_rate = (daily_returns > 0).mean() * 100
avg_win = daily_returns[daily_returns > 0].mean() * 100
avg_loss = daily_returns[daily_returns < 0].mean() * 100
profit_factor = abs(daily_returns[daily_returns > 0].sum() / daily_returns[daily_returns < 0].sum()) if daily_returns[daily_returns < 0].sum() != 0 else float("inf")
total_trades = eq_df["trades"].sum()
avg_positions = eq_df["n_positions"].mean()

metrics = {
    "Initial Capital": f"{INITIAL_CASH:,.0f} CNY",
    "Final Equity": f"{final_equity:,.0f} CNY",
    "Total Return": f"{total_return*100:.2f}%",
    "Annual Return": f"{ann_return*100:.2f}%",
    "Annual Volatility": f"{ann_vol*100:.2f}%",
    "Sharpe Ratio": f"{sharpe:.3f}",
    "Sortino Ratio": f"{sortino:.3f}",
    "Max Drawdown": f"{max_dd*100:.2f}%",
    "Calmar Ratio": f"{calmar:.3f}",
    "Win Rate": f"{win_rate:.1f}%",
    "Avg Win / Avg Loss": f"{avg_win:.2f}% / {avg_loss:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}",
    "Total Trades": f"{total_trades:,}",
    "Avg Positions": f"{avg_positions:.1f}",
}

for k, v in metrics.items():
    print(f"  {k:<22}: {v}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Generating Plots ...")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# ── Panel 1: Equity Curve ──
ax1 = axes[0]
ax1.plot(eq_df.index, eq_df["equity"] / 1e6, color="#1f77b4", linewidth=1.2)
ax1.fill_between(eq_df.index, eq_df["equity"] / 1e6, INITIAL_CASH / 1e6,
                 where=eq_df["equity"] >= INITIAL_CASH, alpha=0.15, color="green")
ax1.fill_between(eq_df.index, eq_df["equity"] / 1e6, INITIAL_CASH / 1e6,
                 where=eq_df["equity"] < INITIAL_CASH, alpha=0.15, color="red")
ax1.axhline(y=INITIAL_CASH / 1e6, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
ax1.set_ylabel("Equity (M CNY)")
ax1.set_title(f"A-Share Cross-Sectional Strategy  ({START_DATE} ~ {END_DATE})\n"
              f"Universe: {SYMBOL_COUNT} stocks | Signal: reversal_5d | Top {int(TOP_QUANTILE*100)}% long")
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Annotations
ax1.annotate(f"Final: {final_equity/1e6:.2f}M\nReturn: {total_return*100:.1f}%",
             xy=(eq_df.index[-1], eq_df["equity"].iloc[-1] / 1e6),
             xytext=(20, 20), textcoords="offset points",
             fontsize=9, color="#1f77b4",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ── Panel 2: Drawdown ──
ax2 = axes[1]
ax2.fill_between(eq_df.index, 0, eq_df["drawdown"] * 100, color="red", alpha=0.3)
ax2.plot(eq_df.index, eq_df["drawdown"] * 100, color="darkred", linewidth=0.8)
ax2.set_ylabel("Drawdown (%)")
ax2.set_title("Drawdown")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

max_dd_idx = eq_df["drawdown"].idxmin()
if pd.notna(max_dd_idx):
    ax2.annotate(f"Max DD: {max_dd*100:.1f}%",
                 xy=(max_dd_idx, max_dd * 100),
                 xytext=(20, -25), textcoords="offset points",
                 fontsize=9, color="darkred",
                 arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ── Panel 3: Monthly Returns Heatmap-style bar chart ──
ax3 = axes[2]
monthly_returns = eq_df["daily_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
months = monthly_returns.index
colors = ["green" if v >= 0 else "red" for v in monthly_returns.values]
ax3.bar(months, monthly_returns.values, color=colors, alpha=0.7, width=20)
ax3.axhline(y=0, color="gray", linewidth=0.5)
ax3.set_ylabel("Monthly Return (%)")
ax3.set_xlabel("Date")
ax3.set_title("Monthly Returns (Green=Up, Red=Down)")
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

# Monthly stats annotation
positive_months = (monthly_returns > 0).sum()
total_months = len(monthly_returns)
monthly_win_rate = positive_months / total_months * 100 if total_months > 0 else 0
best_month = monthly_returns.max()
worst_month = monthly_returns.min()
ax3.text(0.02, 0.95,
         f"Win Months: {positive_months}/{total_months} ({monthly_win_rate:.0f}%)\n"
         f"Best: {best_month:.1f}%  Worst: {worst_month:.1f}%  "
         f"Avg: {monthly_returns.mean():.1f}%",
         transform=ax3.transAxes, fontsize=9, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plot_path = OUTPUT_DIR / "backtest_report.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot_path}")

# ── Summary ──
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Strategy: Cross-sectional reversal (reversal_5d), long top {int(TOP_QUANTILE*100)}%")
print(f"  Universe: {SYMBOL_COUNT} A-share stocks")
print(f"  Period: {START_DATE} → {END_DATE} ({n_years:.1f} years)")
print(f"  Initial: {INITIAL_CASH:,.0f} CNY")
print(f"  Final: {final_equity:,.0f} CNY ({total_return*100:+.2f}%)")
print(f"  Sharpe: {sharpe:.3f}  |  Sortino: {sortino:.3f}  |  Calmar: {calmar:.3f}")
print(f"  Max DD: {max_dd*100:.2f}%  |  Win Rate: {win_rate:.1f}%  |  PF: {profit_factor:.2f}")
print(f"\n  Plot saved to: {plot_path}")
