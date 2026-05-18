"""Compare rebalancing on different weekdays (Mon-Fri) with baseline+GP."""
import sys
import time
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

import run_agent_simulation as sim_mod

JOURNAL_DIR = Path(__file__).resolve().parent / "data" / "results"
GP_FACTORS_SRC = JOURNAL_DIR / "gp_factors.json"

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Use existing GP factors (no rediscovery)
sim_mod.REBALANCE_FREQ = "weekly"

# Patch _is_rebalance_day to check actual weekday
original_is_rebalance = sim_mod.AgentSimulation._is_rebalance_day

def make_rebalance_fn(target_weekday: int):
    """Return an _is_rebalance_day that triggers on a specific weekday (0=Mon)."""
    def _is_rebalance_weekday(self, day_index: int) -> bool:
        if day_index >= len(self._trading_days):
            return True
        td = self._trading_days[day_index]
        return td.dayofweek == target_weekday
    return _is_rebalance_weekday

def run_for_weekday(weekday: int) -> dict:
    """Run baseline+GP backtest for a given weekday."""
    name = WEEKDAY_NAMES[weekday]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_dir = JOURNAL_DIR / f"sim_{ts}_wd{weekday}"

    sim_mod.AgentSimulation._is_rebalance_day = make_rebalance_fn(weekday)

    sim = sim_mod.AgentSimulation(
        start=sim_mod.TRADING_PERIOD[0],
        end=sim_mod.TRADING_PERIOD[1],
        mode="factor",
        output_dir=sim_dir,
        use_gp=False,   # don't re-run GP discovery
    )
    sim.load_gp = True  # load persisted GP factors instead

    # Copy gp_factors.json so _load_persisted_gp_factors can find it
    dest = sim.output_dir / "gp_factors.json"
    if not dest.exists():
        shutil.copy(GP_FACTORS_SRC, dest)

    try:
        import os
        import io
        # Suppress daily trade print() output during simulation
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = sim.run()
        finally:
            sys.stdout = old_stdout

        if "error" in result:
            return {"weekday": name, "error": result["error"]}

        # Equity series stored as attributes, not in result dict
        equity_combined = getattr(sim, "_gp_equity", None)
        equity_baseline = getattr(sim, "_baseline_equity", None)

        if equity_combined is None:
            return {"weekday": name, "error": "no combined equity"}

        ret = (equity_combined.iloc[-1] / equity_combined.iloc[0] - 1) * 100
        peak = equity_combined.cummax()
        dd = ((equity_combined - peak) / peak * 100).min()
        excess = equity_combined - equity_combined.iloc[0]
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

        # Count rebalance entries
        n_rebalances = sum(
            1 for e in sim._journal
            if e.get("reasoning") and "non-rebalance" not in str(e.get("reasoning", ""))
        )

        baseline_ret = None
        if equity_baseline is not None:
            baseline_ret = (equity_baseline.iloc[-1] / equity_baseline.iloc[0] - 1) * 100

        return {
            "weekday": name,
            "return_pct": round(ret, 2),
            "max_dd_pct": round(dd, 2),
            "sharpe": round(sharpe, 3),
            "n_rebalances": n_rebalances,
            "equity_final": round(float(equity_combined.iloc[-1]), 2),
            "baseline_return_pct": round(baseline_ret, 2) if baseline_ret else None,
        }
    except Exception as e:
        import traceback
        return {"weekday": name, "error": f"{e}\n{traceback.format_exc()}"}
    finally:
        sim_mod.AgentSimulation._is_rebalance_day = original_is_rebalance


def main():
    print("=" * 78)
    print("  Weekday Rebalance Comparison — Baseline+GP, SL=500, ¥10,000")
    print("=" * 78)

    results = []
    for wd in range(5):
        print(f"\n--- Testing {WEEKDAY_NAMES[wd]} (weekday={wd}) ---")
        t0 = time.time()
        r = run_for_weekday(wd)
        elapsed = time.time() - t0
        if "error" in r:
            print(f"  ERROR: {r['error'][:200]}")
        else:
            print(f"  Return={r['return_pct']}%  MaxDD={r['max_dd_pct']}%  "
                  f"Sharpe={r['sharpe']}  Rebalances={r['n_rebalances']}  "
                  f"Baseline={r.get('baseline_return_pct')}%  "
                  f"({elapsed:.0f}s)")
        results.append(r)

    print("\n" + "=" * 78)
    print(f"  {'Weekday':<12} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>8} {'#Rebal':>8} {'BaseRet%':>9}")
    print("  " + "-" * 57)
    for r in results:
        if "error" in r:
            print(f"  {r['weekday']:<12} {'ERROR: ' + r['error'][:60]}")
        else:
            print(f"  {r['weekday']:<12} {r['return_pct']:>8.2f} {r['max_dd_pct']:>8.2f} "
                  f"{r['sharpe']:>8.3f} {r['n_rebalances']:>8} "
                  f"{r.get('baseline_return_pct', 'N/A'):>9}")

    # Highlight best
    valid = [r for r in results if "error" not in r]
    if valid:
        best_ret = max(valid, key=lambda r: r["return_pct"])
        best_sharpe = max(valid, key=lambda r: r["sharpe"])
        print(f"\n  Best return: {best_ret['weekday']} ({best_ret['return_pct']}%)")
        print(f"  Best Sharpe: {best_sharpe['weekday']} ({best_sharpe['sharpe']})")

    print("=" * 78)


if __name__ == "__main__":
    main()
