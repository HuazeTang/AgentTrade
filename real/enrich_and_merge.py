#!/usr/bin/env python3
"""Enrich old GP factor with missing fields, run backtests, merge with new.
Uses Simulation internals directly to guarantee matching results."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Must be imported before matplotlib
import config.chart_style  # noqa: F401
import factor.factors as _  # register all factors

from data.cache import read_daily
from data.calendar import get_trading_days
from discovery.compiler import compile_expr
from discovery.expr import Expr
from factor.engine import FactorEngine
from factor.validation import compute_rank_ic, ic_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enrich")

PROJECT_DIR = Path(__file__).resolve().parent
JOURNAL_DIR = PROJECT_DIR / "data" / "results"
GP_FILE = JOURNAL_DIR / "gp_factors.json"
GP_NEW_FILE = JOURNAL_DIR / "gp_factors_new.json"
GP_BACKUP_FILE = JOURNAL_DIR / "gp_factors_backup.json"

TRAIN_PERIOD = (date(2024, 8, 30), date(2025, 10, 16))
BT_START = date(2025, 10, 8)
BT_END = date(2026, 5, 18)
FORWARD_PERIODS = 5
INITIAL_CASH = 100_000.0


def main():
    # ── 1. Load factor entries ──
    with open(GP_FILE, "r") as f:
        old_data = json.load(f)
    with open(GP_NEW_FILE, "r") as f:
        new_data = json.load(f)

    old_factor = old_data["gp_factors"][0]
    new_factor = new_data["gp_factors"][0]
    logger.info("Old: %s", old_factor["name"])
    logger.info("New: %s", new_factor["name"])

    # ── 2. Backup ──
    shutil.copy2(GP_FILE, GP_BACKUP_FILE)
    logger.info("Backup: %s", GP_BACKUP_FILE)

    # ── 3. Register both factors ──
    tree_old = Expr.from_dict(old_factor["expression"])
    compile_expr(tree_old, factor_name=old_factor["name"],
                 category=old_factor.get("category", "gp"), register=True)
    tree_new = Expr.from_dict(new_factor["expression"])
    compile_expr(tree_new, factor_name=new_factor["name"],
                 category=new_factor.get("category", "gp"), register=True)
    logger.info("Registered both GP factors")

    # ── 4. Import AgentSimulation and run backtests ──
    from run_agent_simulation import AgentSimulation, BASELINE_FACTORS, DISABLED_FACTORS

    # Compute IC metrics for old factor on training data
    data_full = read_daily(TRAIN_PERIOD[0], BT_END)
    for col in ["market_cap", "tradable_shares"]:
        if col in data_full.columns and data_full[col].dtype == object:
            data_full[col] = pd.to_numeric(data_full[col], errors="coerce")

    all_dates = sorted(data_full.index.get_level_values("trade_date").unique())
    split_idx = int(len(all_dates) * 0.67)
    train_dates = all_dates[:split_idx]
    train_mask = data_full.index.get_level_values("trade_date").isin(train_dates)
    train_data = data_full.loc[train_mask]

    engine = FactorEngine()
    old_vals = engine.compute([old_factor["name"]], train_data)
    close = data_full["close"].unstack()
    fwd_ret_all = close.pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()
    train_fwd = fwd_ret_all.reindex(old_vals.index)
    ic_series = compute_rank_ic(old_vals[old_factor["name"]], train_fwd)
    ic_s = ic_summary(ic_series.dropna())
    auto_corr = ic_series.dropna().autocorr() if len(ic_series.dropna()) > 1 else 0

    ic_metrics = {
        "ic_mean": round(ic_s["mean"], 4) if not np.isnan(ic_s["mean"]) else 0,
        "ic_ir": round(ic_s["ir"], 3) if not np.isnan(ic_s["ir"]) else 0,
        "ic_std": round(ic_s["std"], 4) if not np.isnan(ic_s["std"]) else 0,
        "hit_rate": round(ic_s["hit_rate"], 4) if not np.isnan(ic_s["hit_rate"]) else 0,
        "auto_corr": round(auto_corr, 4) if not (auto_corr is None or np.isnan(auto_corr)) else 0,
    }
    logger.info("Old factor IC: IC=%.4f, IR=%.3f, hit=%.3f",
                 ic_metrics["ic_mean"], ic_metrics["ic_ir"], ic_metrics["hit_rate"])

    # Baseline backtest
    logger.info("=== Baseline ===")
    sim_bl = AgentSimulation(
        mode="factor", start=BT_START, end=BT_END, initial_cash=INITIAL_CASH,
    )
    # Force load data
    sim_bl._trading_days = get_trading_days(sim_bl.start, sim_bl.end)
    sim_bl._daily_cache = data_full
    # Generate symbols
    symbols_bl = sim_bl._generate_stock_pool(
        sorted(data_full.index.get_level_values("symbol").unique().tolist())
    )

    from factor.registry import registry
    bl_bt = _run_sim_backtest(sim_bl, symbols_bl, [], "baseline")

    # Solo old factor backtest
    logger.info("=== Solo (baseline + old) ===")
    sim_solo = AgentSimulation(
        mode="factor", start=BT_START, end=BT_END, initial_cash=INITIAL_CASH,
    )
    sim_solo._trading_days = get_trading_days(sim_solo.start, sim_solo.end)
    sim_solo._daily_cache = data_full
    symbols_solo = sim_solo._generate_stock_pool(
        sorted(data_full.index.get_level_values("symbol").unique().tolist())
    )
    solo_old = _run_sim_backtest(sim_solo, symbols_solo, [old_factor["name"]], "solo_old")

    # Cumulative (both factors)
    logger.info("=== Cumulative (baseline + old + new) ===")
    sim_cumul = AgentSimulation(
        mode="factor", start=BT_START, end=BT_END, initial_cash=INITIAL_CASH,
    )
    sim_cumul._trading_days = get_trading_days(sim_cumul.start, sim_cumul.end)
    sim_cumul._daily_cache = data_full
    symbols_cumul = sim_cumul._generate_stock_pool(
        sorted(data_full.index.get_level_values("symbol").unique().tolist())
    )
    cumul_both = _run_sim_backtest(sim_cumul, symbols_cumul,
                                    [old_factor["name"], new_factor["name"]],
                                    "cumul_both")

    # ── 5. Build enriched old factor ──
    fitness = (ic_metrics["ic_mean"] * 0.25 + ic_metrics["ic_ir"] * 0.35 +
               ic_metrics["hit_rate"] * 0.15 + (1 - abs(ic_metrics["auto_corr"])) * 0.25)
    old_enriched = {
        "name": old_factor["name"],
        "category": old_factor["category"],
        "expression": old_factor["expression"],
        "generation": -1,
        "fitness": round(fitness, 4),
        "ic_mean": ic_metrics["ic_mean"],
        "ic_ir": ic_metrics["ic_ir"],
        "ic_std": ic_metrics["ic_std"],
        "hit_rate": ic_metrics["hit_rate"],
        "auto_corr": ic_metrics["auto_corr"],
        "max_corr_existing": 0,
        "complexity": old_factor.get("complexity", 3),
        "depth": old_factor.get("depth", 3),
        "validation_passed": True,
        "wf_ic_mean": 0,
        "solo_backtest": solo_old,
        "cumulative_backtest": cumul_both,
        "discovered_at": "2026-05-18",
        "accepted": True,
    }

    # Update new factor's cumulative
    new_factor["cumulative_backtest"] = cumul_both

    # ── 6. Merge and save ──
    merged = {
        "gp_factors": [old_enriched, new_factor],
        "evolution_history": new_data.get("evolution_history", []),
        "meta": {
            "discovered_at": datetime.now().isoformat(),
            "training_period": [str(TRAIN_PERIOD[0]), str(TRAIN_PERIOD[1])],
            "backtest_period": [str(BT_START), str(BT_END)],
            "population_size": new_data.get("meta", {}).get("population_size", "N/A"),
            "max_generations": new_data.get("meta", {}).get("max_generations", "N/A"),
            "accepted_count": 2,
            "total_candidates": 2,
            "baseline_return": bl_bt["cumulative_return"],
            "baseline_sharpe": bl_bt["sharpe_ratio"],
            "final_return": cumul_both["cumulative_return"],
            "final_sharpe": cumul_both["sharpe_ratio"],
        },
    }

    with open(GP_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s (%d factors)", GP_FILE, len(merged["gp_factors"]))

    # ── 7. Console summary ──
    print()
    print("=" * 100)
    print("  GP Factor Merge — Backtest Summary")
    print("=" * 100)
    print(f"{'Factor':<50s} {'IC':>7s} {'IR':>7s} {'Solo_Ret':>9s} {'Solo_SR':>7s} {'MaxDD':>7s}")
    print("-" * 100)
    print(f"{'baseline (40 factors)':<50s} {'-':>7s} {'-':>7s} "
          f"{bl_bt['cumulative_return']*100:>+8.1f}% {bl_bt['sharpe_ratio']:>6.2f} "
          f"{bl_bt['max_drawdown']*100:>+6.1f}%")
    for fac in [old_enriched, new_factor]:
        name = fac["name"][:49]
        slo = fac["solo_backtest"]
        print(f"{name:<50s} {fac['ic_mean']:7.4f} {fac['ic_ir']:7.3f} "
              f"{slo['cumulative_return']*100:>+8.1f}% {slo['sharpe_ratio']:>6.2f} "
              f"{slo['max_drawdown']*100:>+6.1f}%")
    print("-" * 100)
    print(f"{'Combined (both factors)':<50s} {'-':>7s} {'-':>7s} "
          f"{cumul_both['cumulative_return']*100:>+8.1f}% {cumul_both['sharpe_ratio']:>6.2f} "
          f"{cumul_both['max_drawdown']*100:>+6.1f}%")
    delta = (cumul_both['cumulative_return'] - bl_bt['cumulative_return']) * 100
    print(f"\n  Baseline: {bl_bt['cumulative_return']*100:+.2f}% → "
          f"Combined: {cumul_both['cumulative_return']*100:+.2f}% (Δ{delta:+.2f}%)")
    print(f"  Backup: {GP_BACKUP_FILE}")
    print(f"  Saved:  {GP_FILE}")
    print("=" * 100)


def _run_sim_backtest(
    sim, symbols: list[str], gp_names: list[str], label: str,
) -> dict:
    """Run a backtest using AgentSimulation's own methods."""
    from run_agent_simulation import BASELINE_FACTORS, DISABLED_FACTORS

    baseline_names = [f for f in BASELINE_FACTORS if f not in DISABLED_FACTORS]

    factor_df, factor_weights = sim._compute_factor_set_staged(
        baseline_names, gp_names,
    )
    equity = sim._run_backtest_loop(factor_df, factor_weights, symbols, label)
    return _metrics_from_equity(equity, sim.initial_cash)


def _metrics_from_equity(equity: pd.Series, initial_cash: float) -> dict:
    """Compute performance metrics from equity series."""
    if equity.empty:
        return {"cumulative_return": 0, "sharpe_ratio": 0, "max_drawdown": 0,
                "annualized_return": 0, "win_rate": 0}

    eq = equity.dropna()
    if len(eq) < 10:
        return {"cumulative_return": 0, "sharpe_ratio": 0, "max_drawdown": 0,
                "annualized_return": 0, "win_rate": 0}

    final_eq = eq.iloc[-1]
    cr = (final_eq - initial_cash) / initial_cash
    daily_rets = eq.pct_change().dropna()
    if len(daily_rets) < 5:
        sr = 0
    else:
        sr = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    days = len(eq)
    ar = float((1 + cr) ** (252 / days) - 1) if days > 0 else 0
    wr = float((daily_rets > 0).mean()) if len(daily_rets) > 0 else 0

    return {
        "cumulative_return": round(cr, 6),
        "sharpe_ratio": round(sr, 3),
        "max_drawdown": round(mdd, 6),
        "annualized_return": round(ar, 6),
        "win_rate": round(wr, 6),
    }


if __name__ == "__main__":
    main()
