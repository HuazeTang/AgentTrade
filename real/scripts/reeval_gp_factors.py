"""Re-evaluate persisted GP factors with up-to-date metrics including tail capture.

Usage: .venv/bin/python scripts/reeval_gp_factors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from data.cache import read_daily
from discovery.expr import Expr
from discovery.compiler import compile_expr
from discovery.pure_factor import FactorMimickingPortfolio
from factor.engine import FactorEngine

FORWARD_PERIODS = 3
TRADING_DAYS_PER_YEAR = 244


def load_data() -> pd.DataFrame:
    """Load daily data with derived fields (same setup as GP pipeline)."""
    from datetime import date

    df = read_daily(date(2024, 1, 1), date(2026, 5, 21))
    if df.empty:
        raise RuntimeError("No data loaded from cache")

    # Derived fields
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


def compute_fwd_ret(data: pd.DataFrame) -> pd.Series:
    close = data["close"].unstack()
    fwd = close.pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()
    fwd.name = "fwd_ret"
    return fwd


def eval_factor(
    data: pd.DataFrame,
    fwd_ret: pd.Series,
    expr_dict: dict,
    factor_name: str,
) -> dict:
    """Solo evaluate a single GP factor."""
    try:
        tree = Expr.from_dict(expr_dict)
        factor_cls = compile_expr(tree, factor_name=factor_name, register=False)
        fv = factor_cls().compute(data)
    except Exception as e:
        return {"error": f"compile/compute failed: {e}"}

    portfolio = FactorMimickingPortfolio()
    metrics = portfolio.evaluate(fv, fwd_ret)
    return metrics.to_dict()


def eval_cumulative(
    data: pd.DataFrame,
    fwd_ret: pd.Series,
    prior_names: list[str],
    prior_exprs: list[dict],
    new_name: str,
    new_expr: dict,
) -> dict:
    """Evaluate cumulative (prior accepted + new factor) composite."""
    portfolio = FactorMimickingPortfolio()

    # Compile all factors
    all_vals: dict[str, pd.Series] = {}
    all_names = prior_names + [new_name]
    all_exprs = prior_exprs + [new_expr]

    for name, expr_dict in zip(all_names, all_exprs):
        try:
            tree = Expr.from_dict(expr_dict)
            factor_cls = compile_expr(tree, factor_name=name, register=False)
            fv = factor_cls().compute(data)
            if fv.notna().sum() > 0:
                all_vals[name] = fv
        except Exception:
            pass

    if not all_vals:
        return {"error": "no valid factors in composite"}

    metrics = portfolio.evaluate_composite(
        {n: 1.0 for n in all_vals}, all_vals, fwd_ret
    )
    return metrics.to_dict()


def main():
    gp_file = DATA_DIR.parent / "results" / "gp_factors.json"
    if not gp_file.exists():
        print(f"File not found: {gp_file}")
        sys.exit(1)

    with open(gp_file) as f:
        gp_data = json.load(f)

    factors = gp_data.get("gp_factors", [])
    if not factors:
        print("No GP factors found in file.")
        return

    print(f"Loading market data...")
    data = load_data()
    fwd_ret = compute_fwd_ret(data)
    print(f"Loaded {len(data)} rows, {data.index.get_level_values('trade_date').nunique()} dates, "
          f"{data.index.get_level_values('symbol').nunique()} symbols")

    # Split into accepted / rejected for cumulative eval
    accepted = [f for f in factors if f.get("accepted", False)]
    rejected = [f for f in factors if not f.get("accepted", False)]

    print(f"\n{'='*130}")
    print(f"Re-evaluating {len(factors)} factors ({len(accepted)} accepted, {len(rejected)} rejected)")
    print(f"{'='*130}")

    # Evaluate each factor solo + cumulative
    results = []
    for entry in factors:
        name = entry["name"]
        expr = entry["expression"]
        accepted_flag = entry.get("accepted", False)

        print(f"\nEvaluating: {name[:70]}...")
        solo = eval_factor(data, fwd_ret, expr, name)

        # Cumulative: combine with all accepted factors that came before this one
        cum_result = None
        if accepted_flag:
            # This factor is part of the cumulative context
            prior_accepted = [a for a in factors
                            if a.get("accepted") and a["name"] != name]
            if prior_accepted:
                prior_names = [a["name"] for a in prior_accepted]
                prior_exprs = [a["expression"] for a in prior_accepted]
                cum_result = eval_cumulative(
                    data, fwd_ret, prior_names, prior_exprs, name, expr
                )
        else:
            # Rejected factor: cumulative with all accepted factors
            if accepted:
                prior_names = [a["name"] for a in accepted]
                prior_exprs = [a["expression"] for a in accepted]
                cum_result = eval_cumulative(
                    data, fwd_ret, prior_names, prior_exprs, name, expr
                )

        results.append({
            "name": name,
            "category": entry.get("category", "?"),
            "gen": entry.get("generation", "?"),
            "accepted": accepted_flag,
            "solo": solo,
            "cum": cum_result,
        })

    # ── Print Summary ──────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("FACTOR EVALUATION SUMMARY (with tail capture metrics)")
    print(f"{'='*140}")

    # Solo metrics header
    hdr = (f"{'Factor':<48} {'Type':<10} {'Gen':>3} {'Acc':>3} "
           f"{'IC':>6} {'IC_IR':>6} {'SR':>7} {'MaxDD':>7} "
           f"{'Win%':>6} "
           f"{'Decile':>7} {'UpTail':>7} {'DnTail':>7} {'TailW%':>6} "
           f"{'UpConc':>6} {'DnConc':>6}")
    print(hdr)
    print("-" * 140)

    for r in results:
        solo = r["solo"]
        cum = r["cum"]

        def _f(v, width=6):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'-':>{width}}"
            if isinstance(v, float):
                if abs(v) >= 100:
                    return f"{v:>{width}.1f}"
                if abs(v) >= 1:
                    return f"{v:>{width}.3f}"
                return f"{v:>{width}.4f}"
            return f"{str(v):>{width}}"

        short_name = r["name"][:45]
        ic = solo.get("ic_mean")
        ic_ir = solo.get("ic_ir")
        sr = solo.get("sharpe_ratio")
        dd = solo.get("max_drawdown")
        wr = solo.get("win_rate")
        decile = solo.get("top_decile_spread_capture")
        up_cap = solo.get("market_upside_tail_capture")
        dn_cap = solo.get("market_downside_tail_capture")
        tail_wr = solo.get("market_tail_win_rate")
        up_conc = solo.get("upside_concentration")
        dn_conc = solo.get("downside_concentration")

        line = (f"{short_name:<48} {r['category']:<10} {str(r['gen']):>3} "
                f"{'Y' if r['accepted'] else 'N':>3} "
                f"{_f(ic, 6)} {_f(ic_ir, 6)} {_f(sr, 7)} {_f(dd, 7)} "
                f"{_f(wr, 6)} "
                f"{_f(decile, 7)} {_f(up_cap, 7)} {_f(dn_cap, 7)} {_f(tail_wr, 6)} "
                f"{_f(up_conc, 6)} {_f(dn_conc, 6)}")
        print(line)

        # Cumulative line if available
        if cum and "error" not in cum:
            cum_sr = cum.get("sharpe_ratio")
            cum_dd = cum.get("max_drawdown")
            cum_wr = cum.get("win_rate")
            cum_decile = cum.get("top_decile_spread_capture")
            print(f"  {'[cumulative]':>45}                               "
                  f"{_f(cum_sr, 7)} {_f(cum_dd, 7)} {_f(cum_wr, 6)} "
                  f"{_f(cum_decile, 7)}")

    # ── Tail Capture Interpretation Guide ──────────────────────────────
    print(f"\n{'─'*80}")
    print("TAIL CAPTURE INTERPRETATION:")
    print(f"  Decile:  fraction of achievable cross-sectional spread captured (1.0=perfect)")
    print(f"  UpTail:  FMP return / market return on top-10% market days (>0=participates)")
    print(f"  DnTail:  FMP return / market return on bottom-10% market days (<0=hedges)")
    print(f"  TailW%:  fraction of all tail days (up+down) where FMP > 0")
    print(f"  UpConc:  top 10% daily returns / all positive returns (<0.25=smooth alpha)")
    print(f"  DnConc:  bottom 10% daily returns / all negative returns (<0.25=smooth alpha)")


if __name__ == "__main__":
    main()
