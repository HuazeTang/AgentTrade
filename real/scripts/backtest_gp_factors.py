"""Standalone single-factor backtest for GP-discovered factors.

Usage: .venv/bin/python scripts/backtest_gp_factors.py

Loads accepted GP factors from gp_factors.json, re-compiles them, and runs
detailed single-factor backtests: IC analysis, factor mimicking portfolio,
quantile returns, walk-forward validation.  Writes a markdown report.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from data.cache import read_daily
from discovery.expr import Expr
from discovery.compiler import compile_expr
from discovery.pure_factor import FactorMimickingPortfolio, walk_forward_validate
from factor.validation import compute_rank_ic, ic_summary, quantile_returns

FORWARD_PERIODS = 5
TRADING_DAYS_PER_YEAR = 244
GP_FACTORS_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "gp_factors.json"
REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "gp_factor_backtest_report.md"

os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")


def load_data() -> pd.DataFrame:
    """Load daily data with derived fields (same setup as GP pipeline)."""
    df = read_daily(date(2024, 1, 1), date(2026, 5, 21))
    if df.empty:
        raise RuntimeError("No data loaded from cache")

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


def compute_forward_returns(data: pd.DataFrame) -> pd.Series:
    close = data["close"].unstack()
    fwd = close.pct_change(FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()
    fwd.name = "fwd_ret"
    return fwd


def _bar_chart(values: list[float], labels: list[str], width: int = 40,
               unit: str = "", max_bar: float | None = None) -> str:
    """ASCII bar chart helper."""
    if not values:
        return ""
    max_val = max_bar if max_bar is not None else max(max(abs(v) for v in values), 1e-8)
    lines = []
    for label, val in zip(labels, values):
        bar_len = int(abs(val) / max_val * width) if max_val > 0 else 0
        bar = "█" * min(bar_len, width)
        val_str = f"{val:.4f}{unit}"
        lines.append(f"  {label:<30s} {val_str:>12s}  {bar}")
    return "\n".join(lines)


def main():
    print("Loading market data...")
    data = load_data()
    fwd_ret = compute_forward_returns(data)
    print(f"  {len(data)} rows, {data.index.get_level_values(0).nunique()} dates")

    # Load GP factors
    if not GP_FACTORS_PATH.exists():
        print(f"ERROR: {GP_FACTORS_PATH} not found")
        sys.exit(1)
    with open(GP_FACTORS_PATH, "r", encoding="utf-8") as f:
        gp_data = json.load(f)

    factors = gp_data.get("gp_factors", [])
    accepted = [f for f in factors if f.get("accepted", True)]
    print(f"Loaded {len(factors)} total, {len(accepted)} accepted GP factors")

    # Load existing factor values for context
    import factor.factors as _  # register base factors
    from factor.registry import registry
    from factor.engine import FactorEngine
    engine = FactorEngine()
    base_names = [n for n in registry.list_all()
                  if not n.startswith("gp_") and not n.startswith("llm_")]
    print(f"Computing {len(base_names)} baseline factors for context...")
    base_fv = engine.compute(base_names[:20], data)  # top 20 for correlation

    # Use out-of-sample period (after GP training window)
    # GP trained on 2024-08-30 to 2025-10-16. OOS: 2025-10-17 to 2026-05-21
    oos_start = "2025-10-17"
    fwd_ret_oos = fwd_ret.loc[fwd_ret.index.get_level_values(0) >= oos_start]

    # Prepare portfolio evaluator
    portfolio = FactorMimickingPortfolio(
        total_leverage=1.0, rebalance_freq="daily", long_only=False, use_ranks=True,
    )

    report_lines = []
    def w(line: str = "") -> None:
        report_lines.append(line)

    w("# GP Factor Single-Factor Backtest Report")
    w()
    w(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Forward periods:** {FORWARD_PERIODS} days")
    w(f"**In-sample:** 2024-08-30 to 2025-10-16 (270 days, GP training)")
    w(f"**Out-of-sample:** {oos_start} to 2026-05-21 (~145 days)")
    w(f"**Factors evaluated:** {len(accepted)} accepted GP factors")
    w()

    # ── Summary table ──
    w("## 1. Summary")
    w()
    w("| # | Factor | Gen | IC Mean | IC IR | Hit % | Solo SR | Max DD | Cum Ret | Solo SR (OOS) | WF Pass |")
    w("|---|--------|-----|---------|-------|-------|---------|--------|---------|---------------|---------|")

    results: list[dict] = []
    for i, entry in enumerate(accepted):
        name = entry["name"]
        print(f"\n{'='*60}")
        print(f"Backtesting [{i+1}/{len(accepted)}]: {name}")
        print(f"{'='*60}")

        # Re-compile factor
        try:
            tree = Expr.from_dict(entry["expression"])
            factor_cls = compile_expr(tree, factor_name=name, register=False)
        except Exception as e:
            print(f"  ERROR compiling: {e}")
            continue

        # Compute factor values (in-sample + OOS)
        fv = factor_cls().compute(data)
        fv_is = fv.loc[fv.index.get_level_values(0) < oos_start]
        fv_oos = fv.loc[fv.index.get_level_values(0) >= oos_start]

        valid_is = fv_is.notna().sum()
        valid_oos = fv_oos.notna().sum()
        print(f"  Valid obs: IS={valid_is}, OOS={valid_oos}")

        if valid_is < 30:
            print("  SKIP: too few valid IS observations")
            continue

        # ── In-sample evaluation ──
        common_is = fv_is.dropna().index.intersection(fwd_ret.dropna().index)
        fv_is_aligned = fv_is.loc[common_is]
        fwd_is_aligned = fwd_ret.loc[common_is]

        pfm_is = portfolio.evaluate(fv_is_aligned, fwd_is_aligned)

        # ── OOS evaluation ──
        pfm_oos = None
        if valid_oos >= 30:
            common_oos = fv_oos.dropna().index.intersection(fwd_ret_oos.dropna().index)
            if len(common_oos) >= 30:
                fv_oos_aligned = fv_oos.loc[common_oos]
                fwd_oos_aligned = fwd_ret_oos.loc[common_oos]
                pfm_oos = portfolio.evaluate(fv_oos_aligned, fwd_oos_aligned)

        # ── Walk-forward ──
        wf_result = None
        try:
            common_all = fv.dropna().index.intersection(fwd_ret.dropna().index)
            if len(common_all) >= 200:
                wf_result = walk_forward_validate(
                    fv.loc[common_all], fwd_ret.loc[common_all],
                    window_size=180, step_size=60, min_windows=2,
                    portfolio=portfolio,
                )
        except Exception as e:
            print(f"  WF validation failed: {e}")

        # ── Quantile returns ──
        q_returns = None
        try:
            q_df = quantile_returns(fv_is_aligned, fwd_is_aligned, n_quantiles=5)
            # q_df has columns: trade_date, quantile, return
            q_returns = q_df.groupby("quantile")["return"].mean().to_dict()
        except Exception as e:
            print(f"  Quantile analysis failed: {e}")

        # ── IC time series ──
        ic_series = None
        try:
            ic_series = compute_rank_ic(fv_is_aligned, fwd_is_aligned)
        except Exception as e:
            print(f"  IC computation failed: {e}")

        # ── Correlation with existing factors ──
        max_corr_name = ""
        max_corr_val = 0.0
        if base_fv is not None and len(common_is) > 50:
            try:
                fv_df_is = pd.DataFrame({name: fv_is_aligned})
                for col in base_fv.columns:
                    aligned = base_fv[col].loc[base_fv[col].index.intersection(common_is)]
                    if len(aligned) < 50:
                        continue
                    corr = fv_is_aligned.corr(aligned)
                    if abs(corr) > abs(max_corr_val):
                        max_corr_val = corr
                        max_corr_name = col
            except Exception:
                pass

        oos_sr = pfm_oos.sharpe_ratio if pfm_oos else float("nan")
        wf_passed = wf_result.get("passed", False) if wf_result else False

        w(f"| {i+1} | `{name[:35]}` | {entry.get('generation','?')} | "
          f"{pfm_is.ic_mean:+.4f} | {pfm_is.ic_ir:.2f} | {pfm_is.ic_hit_rate*100:.0f}% | "
          f"{pfm_is.sharpe_ratio:.2f} | {pfm_is.max_drawdown*100:.1f}% | {pfm_is.cumulative_return*100:.1f}% | "
          f"{oos_sr:.2f} | {'Y' if wf_passed else 'N'} |")

        results.append({
            "name": name,
            "entry": entry,
            "pfm_is": pfm_is,
            "pfm_oos": pfm_oos,
            "wf_result": wf_result,
            "q_returns": q_returns,
            "ic_series": ic_series,
            "max_corr_name": max_corr_name,
            "max_corr_val": max_corr_val,
            "fv_is_aligned": fv_is_aligned,
            "fwd_is_aligned": fwd_is_aligned,
        })

    w()

    # ── Detailed per-factor sections ──
    w("## 2. Detailed Factor Analysis")
    w()

    for i, r in enumerate(results):
        name = r["name"]
        entry = r["entry"]
        pfm = r["pfm_is"]
        pfm_oos = r["pfm_oos"]
        wf = r["wf_result"]
        qr = r["q_returns"]

        w(f"### 2.{i+1}. `{name}`")
        w()
        w(f"**Category:** {entry.get('category', 'unknown')} | "
          f"**Generation:** {entry.get('generation', '?')} | "
          f"**Depth:** {entry.get('depth', '?')} | "
          f"**Complexity:** {entry.get('complexity', '?')}")
        w()
        w(f"**Expression:** `{entry.get('expression_hint', 'N/A')}`")
        w()

        # IC Metrics
        w("#### IC Analysis (In-Sample)")
        w()
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| IC Mean (Rank) | {pfm.ic_mean:+.4f} |")
        w(f"| IC Std | {pfm.ic_std:.4f} |")
        w(f"| IC IR | {pfm.ic_ir:.2f} |")
        w(f"| IC Hit Rate | {pfm.ic_hit_rate*100:.1f}% |")
        w(f"| IC Dispersion | {pfm.ic_dispersion or 0:.4f} |")
        w()

        # IC stability over time
        if r["ic_series"] is not None and len(r["ic_series"]) > 60:
            ic_s = r["ic_series"]
            ic_monthly = ic_s.groupby(pd.Grouper(freq="ME")).mean()
            w("**IC by Month:**")
            w()
            w("| Month | IC Mean |")
            w("|-------|---------|")
            for dt, val in ic_monthly.items():
                if pd.notna(val):
                    w(f"| {dt.strftime('%Y-%m')} | {val:+.4f} |")
            w()

        # Factor Mimicking Portfolio
        w("#### Factor Mimicking Portfolio (In-Sample)")
        w()
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| Sharpe Ratio | {pfm.sharpe_ratio:.3f} |")
        w(f"| Annualized Return | {pfm.annualized_return*100:.2f}% |")
        w(f"| Volatility (ann.) | {pfm.volatility*100:.2f}% |")
        w(f"| Max Drawdown | {pfm.max_drawdown*100:.2f}% |")
        w(f"| Cumulative Return | {pfm.cumulative_return*100:.2f}% |")
        w(f"| Win Rate (daily) | {pfm.win_rate*100:.1f}% |")
        w(f"| Mean Daily Return | {pfm.mean_daily_return*100:.4f}% |")
        w()

        # Tail capture
        w("#### Tail Event Capture")
        w()
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| Top Decile Spread Capture | {pfm.top_decile_spread_capture or 0:.2f} |")
        w(f"| Market Upside Tail Capture | {pfm.market_upside_tail_capture or 0:.2f} |")
        w(f"| Market Downside Tail Capture | {pfm.market_downside_tail_capture or 0:.2f} |")
        w(f"| Market Tail Win Rate | {(pfm.market_tail_win_rate or 0)*100:.1f}% |")
        w(f"| Upside Concentration | {pfm.upside_concentration or 0:.2f} |")
        w(f"| Downside Concentration | {pfm.downside_concentration or 0:.2f} |")
        w()

        # OOS
        if pfm_oos:
            w("#### Out-of-Sample (OOS) Performance")
            w()
            w(f"**Period:** {oos_start} to 2026-05-21")
            w()
            w("| Metric | IS | OOS |")
            w("|--------|-----|-----|")
            w(f"| IC Mean | {pfm.ic_mean:+.4f} | {pfm_oos.ic_mean:+.4f} |")
            w(f"| IC IR | {pfm.ic_ir:.2f} | {pfm_oos.ic_ir:.2f} |")
            w(f"| Sharpe Ratio | {pfm.sharpe_ratio:.3f} | {pfm_oos.sharpe_ratio:.3f} |")
            w(f"| Max Drawdown | {pfm.max_drawdown*100:.1f}% | {pfm_oos.max_drawdown*100:.1f}% |")
            w(f"| Cum Return | {pfm.cumulative_return*100:.1f}% | {pfm_oos.cumulative_return*100:.1f}% |")
            w(f"| Win Rate | {pfm.win_rate*100:.1f}% | {pfm_oos.win_rate*100:.1f}% |")
            w()

        # Walk-forward
        if wf:
            w("#### Walk-Forward Validation")
            w()
            w(f"**Passed:** {'YES' if wf.get('passed', False) else 'NO'}")
            w(f"**Windows:** {wf.get('n_windows', 0)}")
            w(f"**Mean Test Sharpe:** {wf.get('mean_test_sharpe', 0):.3f}")
            w(f"**Std Test Sharpe:** {wf.get('sharpe_std', 0):.3f}")
            w(f"**Min Test Sharpe:** {wf.get('min_test_sharpe', 0):.3f}")
            w()

        # Quantile analysis
        if qr is not None:
            w("#### Quantile Returns (5-Quantile Long-Short)")
            w()
            w("| Quantile | Mean Return (bp/d) |")
            w("|----------|-------------------|")
            for q, val in qr.items():
                w(f"| Q{q} | {val*10000:.2f} |")
            w()

        # Correlation
        if r["max_corr_name"]:
            w(f"**Max correlation with existing:** `{r['max_corr_name']}` ({r['max_corr_val']:.3f})")
            w()

        w("---")
        w()

    # ── Factor comparison ──
    w("## 3. Factor Comparison")
    w()
    w("### IC IR Rankings")
    names_short = [r["name"][:40] for r in results]
    ic_irs = [r["pfm_is"].ic_ir for r in results]
    w("```")
    w(_bar_chart(ic_irs, names_short, width=50, unit=""))
    w("```")
    w()

    w("### Sharpe Ratio Rankings")
    sharpes = [r["pfm_is"].sharpe_ratio for r in results]
    w("```")
    w(_bar_chart(sharpes, names_short, width=50, unit=""))
    w("```")
    w()

    w("### OOS Stability (IS Sharpe vs OOS Sharpe)")
    oos_srs = [r["pfm_oos"].sharpe_ratio if r["pfm_oos"] else float("nan") for r in results]
    w()
    w("| Factor | IS SR | OOS SR | SR Decay |")
    w("|--------|-------|--------|----------|")
    for i, r in enumerate(results):
        is_sr = r["pfm_is"].sharpe_ratio
        oos_sr = r["pfm_oos"].sharpe_ratio if r["pfm_oos"] else float("nan")
        decay = (is_sr - oos_sr) / max(is_sr, 0.01) * 100 if not np.isnan(oos_sr) else float("nan")
        w(f"| `{names_short[i]}` | {is_sr:.2f} | {oos_sr:.2f} | {decay:.0f}% |")
    w()

    # Write report
    report_content = "\n".join(report_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_content, encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")
    print(f"  Lines: {len(report_lines)}")
    print("Done.")


if __name__ == "__main__":
    main()
