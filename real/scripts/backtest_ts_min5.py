"""交易视角回测 ts_min5_lt_ret_60d — 每次买入卖出清晰可查."""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.titlesize": 11})
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import read_daily
from discovery.expr import Expr
from discovery.compiler import compile_expr

EXPR_DICT = {
    "type": "rolling", "op": "ts_min", "window": 5,
    "child": {
        "type": "binary", "op": "lt",
        "left": {"type": "var", "name": "ret_60d"},
        "right": {"type": "const", "value": -0.4885},
    },
    "quantile": 0.25,
}

START_DATE = date(2024, 10, 1)
END_DATE = date(2026, 5, 21)
TRADING_DAYS = 244


def load_data():
    df = read_daily(date(2023, 1, 1), END_DATE)
    close = df["close"].unstack()
    volume = df["volume"].unstack()
    amount = df["amount"].unstack()
    high = df["high"].unstack()
    low = df["low"].unstack()
    derived = {
        "ret_5d": close.pct_change(5).stack(),
        "ret_20d": close.pct_change(20).stack(),
        "ret_60d": close.pct_change(60).stack(),
        "vol_20d": volume.rolling(20, min_periods=5).mean().stack(),
        "vol_60d": volume.rolling(60, min_periods=10).mean().stack(),
        "hl_ratio": ((high - low) / close.clip(lower=1e-8)).stack(),
        "vol_ratio": (volume / volume.shift(1).clip(lower=1e-8)).stack(),
        "amihud": (close.pct_change().abs() / amount.clip(lower=1e-8)).stack(),
    }
    for name, series in derived.items():
        if name not in df.columns:
            df[name] = series
    return df


def main():
    print("Loading data...")
    data = load_data()
    close = data["close"].unstack()

    tree = Expr.from_dict(EXPR_DICT)
    factor_cls = compile_expr(tree, factor_name="ts_min5_lt_ret_60d", register=False)
    fv = factor_cls().compute(data)

    fv = fv.loc[fv.index.get_level_values("trade_date") >= pd.Timestamp(START_DATE)]
    fv_wide = fv.unstack()
    all_dates = sorted(fv_wide.index)

    # ── Detect trades: contiguous blocks of same stock with factor=1 ──
    # Scan day by day, tracking signal state
    trades = []
    holding = False
    current_symbol = None
    entry_date = None
    entry_price = None

    for i, d in enumerate(all_dates):
        factor_day = fv_wide.loc[d]
        active_symbols = list(factor_day[factor_day == 1.0].index)
        signal_symbol = active_symbols[0] if len(active_symbols) > 0 else None

        if signal_symbol is None:
            # No signal → if holding, exit at today's close
            if holding and current_symbol is not None:
                if d in close.index and current_symbol in close.columns:
                    exit_price = close.loc[d, current_symbol]
                    if not pd.isna(exit_price) and entry_price and entry_price > 0:
                        trade_ret = exit_price / entry_price - 1.0
                        holding_days = all_dates.index(d) - all_dates.index(entry_date)
                        trades.append({
                            "symbol": current_symbol,
                            "entry_date": entry_date,
                            "exit_date": d,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "return": trade_ret,
                            "holding_days": holding_days,
                        })
                holding = False
                current_symbol = None
                entry_date = None
                entry_price = None
        else:
            if not holding:
                # New entry
                if d in close.index and signal_symbol in close.columns:
                    px = close.loc[d, signal_symbol]
                    if not pd.isna(px) and px > 0:
                        holding = True
                        current_symbol = signal_symbol
                        entry_date = d
                        entry_price = px
            elif signal_symbol != current_symbol:
                # Switch: exit old, enter new on same day
                if d in close.index and current_symbol in close.columns:
                    exit_px = close.loc[d, current_symbol]
                    if not pd.isna(exit_px) and entry_price and entry_price > 0:
                        holding_days = all_dates.index(d) - all_dates.index(entry_date)
                        trades.append({
                            "symbol": current_symbol,
                            "entry_date": entry_date,
                            "exit_date": d,
                            "entry_price": entry_price,
                            "exit_price": exit_px,
                            "return": exit_px / entry_price - 1.0,
                            "holding_days": holding_days,
                        })
                if d in close.index and signal_symbol in close.columns:
                    px = close.loc[d, signal_symbol]
                    if not pd.isna(px) and px > 0:
                        current_symbol = signal_symbol
                        entry_date = d
                        entry_price = px
                        holding = True

    # Close any open position at last date
    if holding and current_symbol is not None:
        last_date = all_dates[-1]
        if last_date in close.index and current_symbol in close.columns:
            exit_price = close.loc[last_date, current_symbol]
            if not pd.isna(exit_price) and entry_price and entry_price > 0:
                holding_days = all_dates.index(last_date) - all_dates.index(entry_date)
                trades.append({
                    "symbol": current_symbol,
                    "entry_date": entry_date,
                    "exit_date": last_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": exit_price / entry_price - 1.0,
                    "holding_days": holding_days,
                })

    # ── Print trades ──
    print(f"\n{'='*110}")
    print(f"TRADE LOG — {len(trades)} trades ({START_DATE} → {END_DATE})")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Symbol':<8} {'Entry Date':<12} {'Entry¥':>8} {'Exit Date':<12} {'Exit¥':>8} {'Return':>9} {'Days':>5}")
    print("-" * 110)

    total_return = 1.0
    for i, t in enumerate(trades):
        total_return *= (1.0 + t["return"])
        print(f"{i+1:>3} {t['symbol']:<8} {t['entry_date'].strftime('%Y-%m-%d'):<12} "
              f"{t['entry_price']:>8.2f} {t['exit_date'].strftime('%Y-%m-%d'):<12} "
              f"{t['exit_price']:>8.2f} {t['return']:>+8.2%} {t['holding_days']:>5}")

    # ── Build daily equity curve (trade-by-trade, MTM during holding) ──
    daily_equity = pd.Series(1.0, index=all_dates, dtype=float)
    cash = 1.0
    holding = False
    shares = 0.0
    current_symbol = None

    for i, d in enumerate(all_dates):
        factor_day = fv_wide.loc[d]
        active_symbols = list(factor_day[factor_day == 1.0].index)
        signal_symbol = active_symbols[0] if len(active_symbols) > 0 else None

        if signal_symbol is None:
            if holding:
                # Sell at today's close → convert to cash
                sym = current_symbol
                if d in close.index and sym in close.columns:
                    price = close.loc[d, sym]
                    if not pd.isna(price) and price > 0:
                        cash = shares * price
                        shares = 0.0
                holding = False
                current_symbol = None
            daily_equity.iloc[i] = cash
        else:
            if not holding:
                # Buy at today's close
                if d in close.index and signal_symbol in close.columns:
                    price = close.loc[d, signal_symbol]
                    if not pd.isna(price) and price > 0:
                        shares = cash / price
                        cash = 0.0
                        holding = True
                        current_symbol = signal_symbol
            elif signal_symbol != current_symbol:
                # Switch position
                if d in close.index and current_symbol in close.columns:
                    old_price = close.loc[d, current_symbol]
                    if not pd.isna(old_price) and old_price > 0:
                        cash = shares * old_price
                if d in close.index and signal_symbol in close.columns:
                    new_price = close.loc[d, signal_symbol]
                    if not pd.isna(new_price) and new_price > 0:
                        shares = cash / new_price
                        cash = 0.0
                        current_symbol = signal_symbol

            # MTM at today's close
            if holding and d in close.index and current_symbol in close.columns:
                price = close.loc[d, current_symbol]
                if not pd.isna(price) and price > 0:
                    daily_equity.iloc[i] = shares * price
                else:
                    daily_equity.iloc[i] = cash
            else:
                daily_equity.iloc[i] = cash

    # Fill zero values (shouldn't happen much, but forward-fill just in case)
    daily_equity = daily_equity.replace(0.0, np.nan).ffill()

    # ── Market benchmark: equal-weight all stocks ──
    market_daily_ret = close.pct_change().mean(axis=1).reindex(all_dates).fillna(0)
    market_cum = (1.0 + market_daily_ret).cumprod()

    # ── Hybrid: factor signal → stock; no signal → hold market ETF ──
    hybrid_equity = pd.Series(1.0, index=all_dates, dtype=float)
    holding = False
    shares = 0.0
    cash = 1.0
    current_symbol = None

    for i, d in enumerate(all_dates):
        factor_day = fv_wide.loc[d]
        active_symbols = list(factor_day[factor_day == 1.0].index)
        signal_symbol = active_symbols[0] if len(active_symbols) > 0 else None

        if signal_symbol is None:
            if holding:
                # Exit stock → move to market
                if d in close.index and current_symbol in close.columns:
                    price = close.loc[d, current_symbol]
                    if not pd.isna(price) and price > 0:
                        cash = shares * price
                        shares = 0.0
                holding = False
                current_symbol = None
            # Hold equal-weight market
            if i > 0:
                mkt_ret = market_daily_ret.loc[d] if d in market_daily_ret.index else 0.0
                cash *= (1.0 + mkt_ret)
            hybrid_equity.iloc[i] = cash
        else:
            if not holding:
                # Buy stock at close
                if d in close.index and signal_symbol in close.columns:
                    price = close.loc[d, signal_symbol]
                    if not pd.isna(price) and price > 0:
                        shares = cash / price
                        cash = 0.0
                        holding = True
                        current_symbol = signal_symbol
            elif signal_symbol != current_symbol:
                if d in close.index and current_symbol in close.columns:
                    old_price = close.loc[d, current_symbol]
                    if not pd.isna(old_price) and old_price > 0:
                        cash = shares * old_price
                if d in close.index and signal_symbol in close.columns:
                    new_price = close.loc[d, signal_symbol]
                    if not pd.isna(new_price) and new_price > 0:
                        shares = cash / new_price
                        cash = 0.0
                        current_symbol = signal_symbol

            if holding and d in close.index and current_symbol in close.columns:
                price = close.loc[d, current_symbol]
                if not pd.isna(price) and price > 0:
                    hybrid_equity.iloc[i] = shares * price
                else:
                    hybrid_equity.iloc[i] = cash if cash > 0 else hybrid_equity.iloc[i-1] if i > 0 else 1.0
            else:
                hybrid_equity.iloc[i] = cash if cash > 0 else (hybrid_equity.iloc[i-1] if i > 0 else 1.0)

    hybrid_equity = hybrid_equity.replace(0.0, np.nan).ffill()

    # ── Summary ──
    equity_final = daily_equity.iloc[-1]
    market_final = market_cum.iloc[-1]
    hybrid_final = hybrid_equity.iloc[-1]

    print(f"\n{'='*110}")
    print(f"SUMMARY")
    print(f"{'='*110}")
    print(f"  Total trades:               {len(trades)}")
    print(f"  Win rate:                   {sum(1 for t in trades if t['return']>0)/len(trades)*100:.1f}%")
    print(f"  Avg trade return:           {np.mean([t['return'] for t in trades])*100:.2f}%")
    print(f"  Avg holding days:           {np.mean([t['holding_days'] for t in trades]):.1f}")
    print(f"\n  {'Strategy':<35} {'Final Equity':>12} {'Excess vs Market':>18}")
    print(f"  {'-'*65}")
    print(f"  {'Factor only (cash idle)':<35} {equity_final:.4f}x ({equity_final*100-100:+.1f}%)  {(equity_final/market_final - 1)*100:>+15.1f}%")
    print(f"  {'Factor + market ETF idle':<35} {hybrid_final:.4f}x ({hybrid_final*100-100:+.1f}%)  {(hybrid_final/market_final - 1)*100:>+15.1f}%")
    print(f"  {'Market B&H (baseline)':<35} {market_final:.4f}x ({market_final*100-100:+.1f}%)  {'—':>15}")

    # Drawdown (factor only)
    running_max = daily_equity.expanding().max()
    dd = (daily_equity - running_max) / running_max
    max_dd = dd.min()

    # Hybrid drawdown
    hybrid_running_max = hybrid_equity.expanding().max()
    hybrid_dd = (hybrid_equity - hybrid_running_max) / hybrid_running_max

    n_years = (all_dates[-1] - all_dates[0]).days / 365.25

    print(f"\n  Factor max drawdown:        {max_dd*100:.1f}%")
    print(f"  Hybrid max drawdown:        {hybrid_dd.min()*100:.1f}%")
    print(f"  Factor annualized return:   {((equity_final ** (1/n_years) - 1) * 100):.1f}%")
    print(f"  Hybrid annualized return:   {((hybrid_final ** (1/n_years) - 1) * 100):.1f}%")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(daily_equity.index, daily_equity.values, linewidth=1.0, color="#1f77b4", label="Factor only (cash idle)")
    ax.plot(hybrid_equity.index, hybrid_equity.values, linewidth=1.0, color="#2ca02c", label="Factor + market ETF idle")
    ax.plot(market_cum.index, market_cum.values, linewidth=0.8, color="gray", alpha=0.5, label="Market B&H")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"Equity Curve Comparison ({START_DATE} → {END_DATE})")
    ax.set_ylabel("x")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    ax = axes[0, 1]
    ax.fill_between(dd.index, dd.values * 100, 0, color="#1f77b4", alpha=0.4, linewidth=0.5, label=f"Factor only (max={max_dd*100:.1f}%)")
    ax.fill_between(hybrid_dd.index, hybrid_dd.values * 100, 0, color="#2ca02c", alpha=0.4, linewidth=0.5, label=f"Hybrid (max={hybrid_dd.min()*100:.1f}%)")
    ax.set_title(f"Drawdown Comparison")
    ax.set_ylabel("%")
    ax.legend(fontsize=8, loc="lower left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 0]
    trade_labels = [f"#{i+1}\n{t['symbol']}" for i, t in enumerate(trades)]
    trade_returns_pct = [t["return"] * 100 for t in trades]
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in trade_returns_pct]
    bars = ax.bar(range(len(trades)), trade_returns_pct, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    for i, (r, t) in enumerate(zip(trade_returns_pct, trades)):
        ax.text(i, r + (2 if r > 0 else -4), f"{r:+.1f}%", ha="center", fontsize=7)
    ax.set_xticks(range(len(trades)))
    ax.set_xticklabels(trade_labels, fontsize=6)
    ax.set_title(f"Trade Returns ({len(trades)} trades)")
    ax.set_ylabel("Return %")

    ax = axes[1, 1]
    holding_days = [t["holding_days"] for t in trades]
    ax.bar(range(len(trades)), holding_days, color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(trades)))
    ax.set_xticklabels(trade_labels, fontsize=6)
    ax.set_title(f"Holding Days (avg={np.mean(holding_days):.1f}d)")
    ax.set_ylabel("Days")

    fig.suptitle("Trade-Based Backtest: ts_min5_lt_ret_60d | Factor vs Hybrid vs Market", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent.parent / "data" / "results" / "backtest_ts_min5_ops.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nChart saved to: {out_path}")


if __name__ == "__main__":
    main()
