"""Simulation backtest for GP-discovered factors with real trading mechanics.

Usage: .venv/bin/python scripts/simulate_gp_factors.py

Runs A-share backtest with T+1 settlement, commissions (0.025%), stamp tax (0.1%),
slippage (5bps), lot-size rounding (100 shares).  Generates per-factor trade log,
equity curve data, and a markdown report.

Key difference from FactorMimickingPortfolio: this uses the full BacktestEngine
with realistic trading constraints — you see actual buys, sells, and position
adjustments day by day.
"""

from __future__ import annotations

import bisect
import json
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import BacktestConfig
from backtest.engine import BacktestEngine
from strategy.base import Strategy
from discovery.expr import Expr
from discovery.compiler import compile_expr

GP_FACTORS_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "gp_factors.json"
REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "simulate"
LOG_DIR = REPORT_DIR / "trades"

os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")

# ── Backtest configuration ──────────────────────────────────────────────────
BT_CFG = BacktestConfig(
    start_date=date(2025, 10, 17),   # out-of-sample (after GP training)
    end_date=date(2026, 5, 21),
    initial_cash=1_000_000.0,
    commission_rate=0.00025,         # 万2.5
    min_commission=5.0,              # 最低5元
    stamp_tax_rate=0.001,            # 千1（卖出收）
    transfer_fee_rate=0.00001,       # 万0.1
    slippage_bps=5.0,                # 5bps滑点
    max_position_pct=0.10,           # 单票最大10%
    rebalance_freq="weekly",         # 每周调仓
)


class SingleFactorLongOnly(Strategy):
    """Long-only strategy: buy top-N stocks ranked by a single factor.

    Rebalance weekly.  Equal-weight among selected stocks.
    Uses YESTERDAY's factor values to avoid look-ahead bias.
    Sells stocks that drop out of top-N.
    """

    def __init__(
        self,
        factor_name: str,
        factor_values: pd.Series,
        top_n: int = 50,
        ascending: bool = False,
        name: str = "",
    ):
        self._name = name or f"{factor_name}_long"
        self._factor = factor_values
        self._top_n = top_n
        self._ascending = ascending  # True if lower factor = stronger signal
        self._last_rebalance_date: pd.Timestamp | None = None
        self._current_weights: pd.Series = pd.Series(dtype=float)

        # Pre-compute sorted unique dates from factor index for O(1) lookback lookup
        date_level = self._factor.index.names[0]
        self._factor_dates: list[pd.Timestamp] = sorted(
            self._factor.index.get_level_values(date_level).unique()
        )
        # Build a fast lookup: date → Series of factor values for that date
        self._factor_by_date: dict[pd.Timestamp, pd.Series] = {}
        for d in self._factor_dates:
            self._factor_by_date[d] = self._factor.xs(d, level=date_level)

    @property
    def name(self) -> str:
        return self._name

    def generate_weights(
        self,
        date: pd.Timestamp,
        universe: list[str],
        data: pd.DataFrame,
        prices: pd.Series,
        current_positions: dict[str, float],
        cash: float,
    ) -> pd.Series:
        # Determine if rebalance day (weekly, ~5 calendar days)
        if self._last_rebalance_date is None:
            do_rebalance = self._current_weights.empty
        else:
            days_since = (date - self._last_rebalance_date).days
            do_rebalance = days_since >= 5
            # Also retry if last rebalance produced no positions (e.g. no price data)
            if not do_rebalance and self._current_weights.sum() == 0 and days_since >= 1:
                do_rebalance = True

        if not do_rebalance and not self._current_weights.empty:
            return self._current_weights.reindex(universe, fill_value=0.0)

        self._last_rebalance_date = date

        # Find the most recent factor date strictly before `date` to avoid look-ahead
        idx = bisect.bisect_left(self._factor_dates, date) - 1
        if idx < 0:
            self._current_weights = pd.Series(0.0, index=universe)
            return self._current_weights
        lookback_date = self._factor_dates[idx]

        factor_slice = self._factor_by_date.get(lookback_date, pd.Series(dtype=float))
        factor_lagged = factor_slice.reindex(universe)
        valid = factor_lagged.dropna()
        valid_prices = prices.reindex(valid.index).dropna()
        valid = valid.reindex(valid_prices.index).dropna()

        if valid.empty:
            self._current_weights = pd.Series(0.0, index=universe)
            return self._current_weights

        ranked = valid.sort_values(ascending=self._ascending)
        selected = ranked.head(self._top_n)
        n_sel = len(selected)
        if n_sel == 0:
            self._current_weights = pd.Series(0.0, index=universe)
            return self._current_weights

        weight = min(1.0 / n_sel, 0.10)
        self._current_weights = pd.Series(weight, index=selected.index)
        self._current_weights = self._current_weights.reindex(universe, fill_value=0.0)
        return self._current_weights


def load_factor_values(data: pd.DataFrame, entry: dict) -> pd.Series | None:
    """Compile factor and compute values. Returns MultiIndex Series."""
    try:
        tree = Expr.from_dict(entry["expression"])
        factor_cls = compile_expr(tree, factor_name=entry["name"], register=False)
        return factor_cls().compute(data)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def summarize_result(
    name: str, entry: dict, result, fv: pd.Series,
    save_trades: bool = False,
) -> dict:
    """Extract summary metrics and optionally save trade log."""
    fills = result.fills
    equity = result.equity_curve
    daily_ret = result.daily_returns
    benchmark_ret = result.benchmark_returns

    # Basic metrics
    total_return = (1 + daily_ret).prod() - 1 if len(daily_ret) > 0 else 0.0
    ann_return = ((1 + total_return) ** (244 / max(len(daily_ret), 1))) - 1
    ann_vol = daily_ret.std() * np.sqrt(244) if len(daily_ret) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Benchmark comparison
    bench_total = (1 + benchmark_ret).prod() - 1 if len(benchmark_ret) > 0 else 0.0
    excess = total_return - bench_total

    # Trade statistics
    if fills is not None and len(fills) > 0:
        n_buys = len(fills[fills["side"] == "buy"])
        n_sells = len(fills[fills["side"] == "sell"])
        total_commission = fills["commission"].sum() if "commission" in fills.columns else 0
        total_stamp = fills["stamp_tax"].sum() if "stamp_tax" in fills.columns else 0
        total_trades = n_buys + n_sells
    else:
        n_buys = n_sells = total_commission = total_stamp = total_trades = 0

    # Win rate (daily)
    win_rate = (daily_ret > 0).mean() if len(daily_ret) > 0 else 0.0

    # Save trade log
    if save_trades and fills is not None and len(fills) > 0:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace("/", "_")[:60]
        trade_path = LOG_DIR / f"{safe_name}_trades.csv"
        fills.to_csv(trade_path, index=False)
        equity_path = LOG_DIR / f"{safe_name}_equity.csv"
        pd.DataFrame({
            "date": equity.index,
            "equity": equity.values,
            "daily_return": daily_ret.values,
            "benchmark_return": benchmark_ret.values,
        }).to_csv(equity_path, index=False)

    return {
        "name": name,
        "category": entry.get("category", ""),
        "generation": entry.get("generation", 0),
        "depth": entry.get("depth", 0),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "bench_total": bench_total,
        "excess": excess,
        "win_rate": win_rate,
        "n_trades": total_trades,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "commission": total_commission,
        "stamp_tax": total_stamp,
        "final_equity": equity.iloc[-1] if len(equity) > 0 else BT_CFG.initial_cash,
        "equity_curve": equity,
        "daily_returns": daily_ret,
    }


def _build_journal(result, bt_cfg: BacktestConfig) -> list[dict]:
    """Reconstruct daily journal entries from backtest result."""
    fills = result.fills
    equity = result.equity_curve
    daily_ret = result.daily_returns

    # Group fills by date
    fills_by_date: dict[pd.Timestamp, list[dict]] = {}
    if fills is not None and len(fills) > 0:
        for _, row in fills.iterrows():
            d = pd.Timestamp(row["trade_date"])
            fills_by_date.setdefault(d, []).append({
                "symbol": row["symbol"],
                "side": row["side"],
                "shares": int(row["quantity"]),
                "price": float(row["price"]),
                "commission": float(row.get("commission", 0)),
                "stamp_tax": float(row.get("stamp_tax", 0)),
                "transfer_fee": float(row.get("transfer_fee", 0)),
            })

    # Reconstruct positions day by day
    positions: dict[str, float] = {}
    cash = bt_cfg.initial_cash
    journal = []

    for d in equity.index:
        # Apply fills for this day
        day_fills = fills_by_date.get(d, [])
        for f_rec in day_fills:
            px = f_rec["price"]
            qty = f_rec["shares"]
            cost = f_rec["commission"] + f_rec["stamp_tax"] + f_rec["transfer_fee"]
            if f_rec["side"] == "buy":
                cash -= px * qty + cost
                positions[f_rec["symbol"]] = positions.get(f_rec["symbol"], 0) + qty
            else:
                cash += px * qty - cost
                positions[f_rec["symbol"]] = positions.get(f_rec["symbol"], 0) - qty
                if positions[f_rec["symbol"]] <= 0:
                    positions.pop(f_rec["symbol"], None)

        # Remove zero positions
        positions = {s: q for s, q in positions.items() if q > 0}

        pos_list = [{"symbol": s, "shares": int(q)} for s, q in sorted(positions.items())]
        journal.append({
            "date": d.strftime("%Y-%m-%d"),
            "cash": round(cash, 2),
            "equity": round(float(equity.loc[d]), 2),
            "fills": day_fills,
            "positions": pos_list,
        })

    return journal


def _make_journal_json(result, entry: dict, bt_cfg: BacktestConfig) -> dict:
    """Build the full journal.json content for a factor."""
    journal = _build_journal(result, bt_cfg)
    eq = result.equity_curve
    ret = result.daily_returns

    # Compute metrics
    total_ret = (1 + ret).prod() - 1 if len(ret) > 0 else 0.0
    ann_ret = ((1 + total_ret) ** (244 / max(len(ret), 1))) - 1 if total_ret > -1 else -1.0
    ann_vol = float(ret.std() * np.sqrt(244)) if len(ret) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    n_trades = sum(len(e["fills"]) for e in journal)
    n_buys = sum(1 for e in journal for f in e["fills"] if f["side"] == "buy")
    n_sells = n_trades - n_buys

    metrics = {
        "factor": entry["name"],
        "category": entry.get("category", ""),
        "generation": entry.get("generation", 0),
        "start_date": str(bt_cfg.start_date),
        "end_date": str(bt_cfg.end_date),
        "initial_cash": bt_cfg.initial_cash,
        "final_equity": float(eq.iloc[-1]) if len(eq) > 0 else bt_cfg.initial_cash,
        "total_return": float(total_ret),
        "ann_return": float(ann_ret),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_trades": n_trades,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "trading_days": len(journal),
    }

    return {"metrics": metrics, "journal": journal}


# ── Chart generation ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
})


def _generate_equity_position_chart(journal: list[dict], factor_name: str,
                                     bt_cfg: BacktestConfig, out_path: Path) -> None:
    """Dual-panel chart: equity curve (top) + position value & count (bottom)."""
    dates = [pd.Timestamp(e["date"]) for e in journal]
    equities = [e["equity"] for e in journal]
    cash_vals = [e["cash"] for e in journal]
    position_values = [e["equity"] - e["cash"] for e in journal]
    position_counts = [len(e["positions"]) for e in journal]

    if len(dates) < 2:
        return

    norm_equity = np.array(equities) / bt_cfg.initial_cash

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08},
        sharex=True,
    )

    # ── Top: Equity curve ──
    ax1.plot(dates, norm_equity, color="#2c3e50", linewidth=2.5, label="Strategy", zorder=10)
    ax1.fill_between(dates, norm_equity, 1.0,
                     where=norm_equity >= 1.0, alpha=0.10, color="#27ae60")
    ax1.fill_between(dates, norm_equity, 1.0,
                     where=norm_equity < 1.0, alpha=0.10, color="#e74c3c")
    ax1.axhline(y=1.0, color="#7f8c8d", linewidth=0.8, linestyle="--", alpha=0.5)

    final_nav = norm_equity[-1]
    ax1.scatter(dates[-1], final_nav, color="#2c3e50", s=60, zorder=12)
    ax1.annotate(f"NAV {final_nav:.3f}", xy=(dates[-1], final_nav),
                 xytext=(15, 0), textcoords="offset points",
                 fontsize=9, color="#2c3e50", fontweight="bold", ha="left", va="center")

    name_short = factor_name[:45]
    ax1.set_ylabel("NAV (1.0 = Initial)", fontsize=11)
    ax1.set_title(f"{name_short} — Equity Curve  ({bt_cfg.start_date} → {bt_cfg.end_date})",
                  fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.25)

    # ── Bottom: Position composition ──
    ax2.fill_between(dates, position_values, alpha=0.4, color="#3498db", label="Position Value")
    ax2.plot(dates, position_values, color="#2980b9", linewidth=1.5, drawstyle="steps-post")
    ax2.set_ylabel("Position Value (CNY)", fontsize=11, color="#2980b9")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))

    ax2b = ax2.twinx()
    ax2b.plot(dates, position_counts, color="#e67e22", linewidth=1.5,
              marker=".", markersize=6, drawstyle="steps-post", label="# Positions")
    ax2b.set_ylabel("# Positions", fontsize=11, color="#e67e22")
    ax2b.set_ylim(bottom=0)
    if max(position_counts) > 0:
        ax2b.set_ylim(top=max(position_counts) * 1.2)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, framealpha=0.9)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _generate_trade_charts(fills_df: pd.DataFrame, daily_cache: pd.DataFrame,
                            chart_dir: Path) -> None:
    """Generate K-line charts with buy/sell markers for traded stocks."""
    try:
        import mplfinance as mpf
    except ImportError:
        print("  mplfinance not installed, skipping K-line charts")
        return

    if fills_df is None or len(fills_df) == 0:
        return

    a_share_style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(up='red', down='green', edge='inherit', wick='inherit', volume='inherit'),
        gridstyle='-', gridaxis='horizontal', gridcolor='#e0e0e0',
        rc={"font.size": 8},
    )
    BUY_COLOR = "#cc0000"
    SELL_COLOR = "#00aa00"

    # Group fills by symbol
    trades_by_symbol: dict[str, list[dict]] = {}
    for _, row in fills_df.iterrows():
        sym = row["symbol"]
        trades_by_symbol.setdefault(sym, []).append({
            "date": pd.Timestamp(row["trade_date"]),
            "side": row["side"],
            "shares": int(row["quantity"]),
            "price": float(row["price"]),
        })

    all_trade_dates = sorted({t["date"] for trades in trades_by_symbol.values() for t in trades})
    chart_start = all_trade_dates[0] - pd.Timedelta(days=30)
    chart_end = all_trade_dates[-1] + pd.Timedelta(days=10)

    total_syms = len(trades_by_symbol)
    n_cols, n_rows = 2, 2
    per_page = n_cols * n_rows
    syms_list = sorted(trades_by_symbol.keys(), key=lambda s: len(trades_by_symbol[s]), reverse=True)
    total_pages = (total_syms + per_page - 1) // per_page

    print(f"  Generating K-line charts: {total_syms} symbols, {total_pages} pages ...")

    for page_start in range(0, total_syms, per_page):
        page_syms = syms_list[page_start:page_start + per_page]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), squeeze=False)
        fig.suptitle("Trade Charts — GP Factor Simulation", fontsize=15, fontweight="bold", y=0.99)
        fig.text(0.50, 0.975, "Red=Up Green=Down | ▲ Buy   ▼ Sell",
                 ha="center", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

        for idx, sym in enumerate(page_syms):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            try:
                sym_data = daily_cache.xs(sym, level="symbol")
                sym_data = sym_data.loc[(sym_data.index >= chart_start) & (sym_data.index <= chart_end)]
            except KeyError:
                ax.text(0.5, 0.5, f"{sym}\nNo data", transform=ax.transAxes, ha="center", fontsize=11)
                ax.set_title(sym, fontsize=12, fontweight="bold")
                continue

            if sym_data.empty:
                ax.text(0.5, 0.5, f"{sym}\nNo data in range", transform=ax.transAxes, ha="center", fontsize=11)
                ax.set_title(sym, fontsize=12, fontweight="bold")
                continue

            ohlc = sym_data[["open", "high", "low", "close", "volume"]].rename(
                columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            ohlc = ohlc.dropna(subset=["Open", "High", "Low", "Close"])

            if ohlc.empty:
                ax.text(0.5, 0.5, f"{sym}\nNo valid OHLC", transform=ax.transAxes, ha="center", fontsize=11)
                ax.set_title(sym, fontsize=12, fontweight="bold")
                continue

            trades = trades_by_symbol[sym]
            buy_dates = [t["date"] for t in trades if t["side"] == "buy"]
            buy_prices = [t["price"] for t in trades if t["side"] == "buy"]
            sell_dates = [t["date"] for t in trades if t["side"] == "sell"]
            sell_prices = [t["price"] for t in trades if t["side"] == "sell"]

            addplots = []
            if buy_dates:
                bm = pd.DataFrame({"price": buy_prices}, index=pd.DatetimeIndex(buy_dates))
                ba = bm.reindex(ohlc.index)
                addplots.append(mpf.make_addplot(
                    ba["price"].where(ba["price"].notna(), other=np.nan),
                    type="scatter", marker="^", color=BUY_COLOR, markersize=100, ax=ax))
            if sell_dates:
                sm = pd.DataFrame({"price": sell_prices}, index=pd.DatetimeIndex(sell_dates))
                sa = sm.reindex(ohlc.index)
                addplots.append(mpf.make_addplot(
                    sa["price"].where(sa["price"].notna(), other=np.nan),
                    type="scatter", marker="v", color=SELL_COLOR, markersize=100, ax=ax))

            try:
                mpf.plot(ohlc, type="candle", style=a_share_style,
                         addplot=addplots if addplots else None,
                         ax=ax, volume=False, show_nontrading=False)
            except Exception:
                ax.plot(ohlc.index, ohlc["Close"], color="black", linewidth=1.2)
                if buy_dates:
                    ax.scatter(buy_dates, buy_prices, marker="^", color=BUY_COLOR, s=100, zorder=5)
                if sell_dates:
                    ax.scatter(sell_dates, sell_prices, marker="v", color=SELL_COLOR, s=100, zorder=5)

            buy_count = sum(1 for t in trades if t["side"] == "buy")
            sell_count = len(trades) - buy_count
            ax.set_title(f"{sym}  (Buy {buy_count}  Sell {sell_count})", fontsize=12, fontweight="bold")
            ax.set_ylabel("Price", fontsize=9)

        # Hide unused subplots
        for idx2 in range(len(page_syms), n_rows * n_cols):
            r2, c2 = divmod(idx2, n_cols)
            axes[r2][c2].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        page_num = page_start // per_page + 1
        chart_path = chart_dir / f"charts_p{page_num:02d}.png"
        fig.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── Detail charts for top 10 most-traded stocks ──
    top_syms = syms_list[:min(10, len(syms_list))]
    for sym in top_syms:
        try:
            sym_data = daily_cache.xs(sym, level="symbol")
            sym_data = sym_data.loc[(sym_data.index >= chart_start) & (sym_data.index <= chart_end)]
        except KeyError:
            continue
        if sym_data.empty:
            continue

        ohlc = sym_data[["open", "high", "low", "close", "volume"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        ohlc = ohlc.dropna(subset=["Open", "High", "Low", "Close"])
        if ohlc.empty:
            continue

        trades = trades_by_symbol[sym]
        buy_dates = [t["date"] for t in trades if t["side"] == "buy"]
        buy_prices = [t["price"] for t in trades if t["side"] == "buy"]
        sell_dates = [t["date"] for t in trades if t["side"] == "sell"]
        sell_prices = [t["price"] for t in trades if t["side"] == "sell"]

        # Create figure first so axes exist for addplot targeting
        fig_detail, (ax_main, ax_vol) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

        addplots_d = []
        if buy_dates:
            bm = pd.DataFrame({"price": buy_prices}, index=pd.DatetimeIndex(buy_dates))
            ba = bm.reindex(ohlc.index)
            addplots_d.append(mpf.make_addplot(
                ba["price"].where(ba["price"].notna(), other=np.nan),
                type="scatter", marker="^", color=BUY_COLOR, markersize=80, ax=ax_main))
        if sell_dates:
            sm = pd.DataFrame({"price": sell_prices}, index=pd.DatetimeIndex(sell_dates))
            sa = sm.reindex(ohlc.index)
            addplots_d.append(mpf.make_addplot(
                sa["price"].where(sa["price"].notna(), other=np.nan),
                type="scatter", marker="v", color=SELL_COLOR, markersize=80, ax=ax_main))

        # Volume bars
        if "Volume" in ohlc.columns and len(ohlc) > 0:
            vol_colors = ["red" if ohlc["Close"].iloc[i] >= ohlc["Open"].iloc[i] else "green"
                          for i in range(len(ohlc))]
            addplots_d.append(mpf.make_addplot(
                ohlc["Volume"], type="bar", color=vol_colors,
                ax=ax_vol, alpha=0.5, width=0.8))

        try:
            mpf.plot(ohlc, type="candle", style=a_share_style,
                     addplot=addplots_d, ax=ax_main, volume=False,
                     show_nontrading=False)
        except Exception as e:
            ax_main.plot(ohlc.index, ohlc["Close"], color="black", linewidth=1.2)
            if buy_dates:
                ax_main.scatter(buy_dates, buy_prices, marker="^", color=BUY_COLOR, s=80, zorder=5)
            if sell_dates:
                ax_main.scatter(sell_dates, sell_prices, marker="v", color=SELL_COLOR, s=80, zorder=5)

        buy_count = sum(1 for t in trades if t["side"] == "buy")
        sell_count = len(trades) - buy_count
        ax_main.set_title(f"{sym} — Buy {buy_count}  Sell {sell_count}", fontsize=13, fontweight="bold")

        fig_detail.tight_layout()
        fig_detail.savefig(chart_dir / f"detail_{sym}.png", dpi=200, bbox_inches="tight")
        plt.close(fig_detail)

    print(f"  Charts saved to {chart_dir} ({total_syms} symbols)")


def _write_report(result: dict, journal: dict, out_path: Path) -> None:
    """Write markdown report for a single factor."""
    m = journal["metrics"]
    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# GP Factor Simulation — {m['factor']}")
    w()
    w(f"**Period:** {m['start_date']} → {m['end_date']} | **Initial:** ¥{m['initial_cash']:,.0f}")
    w()
    w("## Metrics")
    w()
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Total Return | {m['total_return']*100:+.2f}% |")
    w(f"| Annualized Return | {m['ann_return']*100:.2f}% |")
    w(f"| Annualized Vol | {m['ann_vol']*100:.2f}% |")
    w(f"| Sharpe Ratio | {m['sharpe']:.3f} |")
    w(f"| Max Drawdown | {m['max_drawdown']*100:.2f}% |")
    w(f"| Final Equity | ¥{m['final_equity']:,.0f} |")
    w(f"| Total Trades | {m['total_trades']} ({m['n_buys']} buys / {m['n_sells']} sells) |")
    w(f"| Trading Days | {m['trading_days']} |")
    w()
    w("## Daily Journal (first 30 days)")
    w()
    w("| Date | Equity | Cash | #Pos | Trades |")
    w("|------|--------|------|------|--------|")
    for e in journal["journal"][:30]:
        w(f"| {e['date']} | ¥{e['equity']:,.0f} | ¥{e['cash']:,.0f} | {len(e['positions'])} | {len(e['fills'])} |")

    if len(journal["journal"]) > 30:
        w(f"| ... | ... | ... | ... | ... |")
        w()
        w(f"*(Full journal with {len(journal['journal'])} days in journal.json)*")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    from data.cache import read_daily
    from datetime import datetime

    print("=" * 70)
    print("GP Factor Simulation Backtest (Real Trading Mechanics)")
    print("=" * 70)
    print(f"Period: {BT_CFG.start_date} → {BT_CFG.end_date}")
    print(f"Initial Cash: ¥{BT_CFG.initial_cash:,.0f}")
    print(f"Commission: {BT_CFG.commission_rate*10000:.1f}bp | Stamp: {BT_CFG.stamp_tax_rate*100:.1f}% | Slippage: {BT_CFG.slippage_bps}bp")
    print(f"Rebalance: {BT_CFG.rebalance_freq} | Max Position: {BT_CFG.max_position_pct*100:.0f}%")
    print()

    # Load data
    print("Loading market data...")
    df = read_daily(date(2024, 10, 1), BT_CFG.end_date)

    # Add derived fields
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

    print(f"  {len(df)} rows loaded")
    print()

    # Load GP factors
    if not GP_FACTORS_PATH.exists():
        print(f"ERROR: {GP_FACTORS_PATH} not found")
        sys.exit(1)
    with open(GP_FACTORS_PATH, "r", encoding="utf-8") as f:
        gp_data = json.load(f)

    factors = [f for f in gp_data.get("gp_factors", []) if f.get("accepted", True)]
    print(f"Found {len(factors)} accepted GP factors to backtest")
    print()

    # Preload daily cache for chart range (wider range for K-line context)
    chart_cache = read_daily(date(2025, 9, 1), BT_CFG.end_date)

    all_results: list[dict] = []

    for i, entry in enumerate(factors):
        name = entry["name"]
        print(f"{'─'*60}")
        print(f"[{i+1}/{len(factors)}] {name}")
        print(f"  Category: {entry.get('category','?')} | Gen: {entry.get('generation','?')} | "
              f"Depth: {entry.get('depth','?')} | Complexity: {entry.get('complexity','?')}")

        # Compute factor values
        fv = load_factor_values(df, entry)
        if fv is None:
            print("  SKIP: failed to compile")
            continue

        n_valid = fv.notna().sum()
        fv_std = fv.std()
        print(f"  Valid factor observations: {n_valid} | Std: {fv_std:.4f}")
        if n_valid < 100:
            print("  SKIP: too few valid observations")
            continue
        if fv_std < 1e-10:
            print("  SKIP: near-zero variance (constant factor, cannot rank stocks)")
            continue

        # Determine direction
        ascending = False
        try:
            df_is = read_daily(date(2024, 8, 30), date(2025, 10, 16))
            close_is = df_is["close"].unstack()
            fwd_is = close_is.pct_change(5).shift(-5).stack()
            fv_is = fv.reindex(fwd_is.index).dropna()
            fwd_is = fwd_is.reindex(fv_is.index).dropna()
            from factor.validation import compute_rank_ic
            ic_is = compute_rank_ic(fv_is, fwd_is)
            ic_mean = ic_is.mean()
            ascending = ic_mean < 0
            print(f"  IC mean (IS): {ic_mean:+.4f} → {'lower=better' if ascending else 'higher=better'}")
        except Exception:
            pass

        # Run backtest
        try:
            strategy = SingleFactorLongOnly(
                factor_name=name,
                factor_values=fv,
                top_n=30,
                ascending=ascending,
            )
            engine = BacktestEngine(config=BT_CFG, strategy=strategy)
            bt_result = engine.run()
        except Exception as e:
            print(f"  ERROR in backtest: {e}")
            import traceback
            traceback.print_exc()
            continue

        result_entry = summarize_result(name, entry, bt_result, fv, save_trades=True)
        all_results.append(result_entry)
        journal_data = _make_journal_json(bt_result, entry, BT_CFG)

        print(f"  Return: {result_entry['total_return']*100:.2f}% | "
              f"Sharpe: {result_entry['sharpe']:.2f} | "
              f"MaxDD: {result_entry['max_dd']*100:.2f}% | "
              f"Trades: {result_entry['n_trades']} | "
              f"Final: ¥{result_entry['final_equity']:,.0f}")

        # ── Create per-factor sim directory ──
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        safe_name = name.replace("/", "_")[:40]
        sim_dir = REPORT_DIR.parent / f"sim_{ts}_{safe_name}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        charts_subdir = sim_dir / "charts"
        charts_subdir.mkdir(parents=True, exist_ok=True)

        # Save journal.json
        journal_path = sim_dir / "journal.json"
        with open(journal_path, "w", encoding="utf-8") as f:
            json.dump(journal_data, f, ensure_ascii=False, indent=2, default=str)

        # Generate equity_position chart
        eq_chart_path = sim_dir / "equity_position.png"
        _generate_equity_position_chart(journal_data["journal"], name, BT_CFG, eq_chart_path)

        # Generate K-line trade charts
        _generate_trade_charts(bt_result.fills, chart_cache, charts_subdir)

        # Write markdown report
        report_path = sim_dir / "report.md"
        _write_report(result_entry, journal_data, report_path)

        # Copy trade/equity CSVs into sim dir
        trades_dir = sim_dir / "trades"
        trades_dir.mkdir(exist_ok=True)
        for src in LOG_DIR.glob(f"{safe_name}*"):
            import shutil
            shutil.copy2(src, trades_dir / src.name)

        print(f"  Output: {sim_dir}")

    print()
    print("=" * 70)
    print("Done.")
    print(f"Results in: {REPORT_DIR.parent}/sim_*/")


if __name__ == "__main__":
    main()
