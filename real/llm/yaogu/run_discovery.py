"""Entry point for LLM-driven 妖股 factor discovery.

Usage:
    python -m llm.yaogu.run_discovery [--period 2018-2026] [--output data/results/yaogu_factors.json]

Pipeline:
    1. Load daily OHLCV data
    2. Extract 妖股 case pairs (positive + matched negatives)
    3. LLM generates expression tree factors via contrastive analysis
    4. Compile and validate factors
    5. Save to JSON (gp_factors.json compatible format)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run_agent_simulation import (
    AgentSimulation, get_trading_days, BASELINE_FACTORS, DISABLED_FACTORS,
)
from data.cache import read_daily
from llm.yaogu.case_extractor import YaoguCaseExtractor
from llm.yaogu.generator import YaoguFactorGenerator, GeneratedFactor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ──
DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2026, 5, 25)
OUTPUT_DIR = Path("data/results")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM 妖股 Factor Discovery")
    parser.add_argument("--start", type=str, default=str(DEFAULT_START),
                        help="Backtest start date")
    parser.add_argument("--end", type=str, default=str(DEFAULT_END),
                        help="Backtest end date")
    parser.add_argument("--output", type=str, default="data/results/yaogu_factors.json",
                        help="Output JSON path")
    parser.add_argument("--n-factors", type=int, default=5,
                        help="Number of factors to request from LLM")
    parser.add_argument("--min-cum-ret", type=float, default=0.30,
                        help="Minimum forward cumulative return for 妖股")
    parser.add_argument("--forward-window", type=int, default=10,
                        help="Forward window (days) to check for launch")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("=" * 60)
    print("  LLM 妖股 Factor Discovery")
    print(f"  Period: {start} ~ {end}")
    print(f"  Forward window: {args.forward_window}d")
    print(f"  Min cumulative return: {args.min_cum_ret*100:.0f}%")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/4] Loading data...")
    sim = AgentSimulation(start=start, end=end, mode="factor")
    sim._trading_days = get_trading_days(sim.start, sim.end)
    sim._daily_cache = read_daily(start - timedelta(days=400), end)

    all_syms = sorted(sim._daily_cache.index.get_level_values("symbol").unique().tolist())
    symbols = sim._generate_stock_pool(all_syms)
    pool_mask = sim._daily_cache.index.get_level_values("symbol").isin(symbols)
    sim._daily_cache = sim._daily_cache[pool_mask]
    sim._daily_cache = sim._add_derived_features(sim._daily_cache)

    print(f"  Symbols: {len(symbols)}, rows: {len(sim._daily_cache)}")

    # ── 2. Extract case pairs ──
    print("\n[2/4] Extracting 妖股 case pairs...")
    extractor = YaoguCaseExtractor(
        forward_window=args.forward_window,
        min_cum_ret=args.min_cum_ret,
    )
    pairs = extractor.extract(sim._daily_cache, symbols)

    if not pairs:
        print("  No 妖股 cases found. Try adjusting thresholds or date range.")
        return

    summary = YaoguCaseExtractor.compute_summary(pairs)
    print(f"  Unique 妖股: {summary['n_positives']}")
    print(f"  Matched pairs: {summary['n_pairs']}")
    print(f"  Pos avg fwd ret: {summary['pos_avg_forward_ret']*100:.1f}%")
    print(f"  Neg avg fwd ret: {summary['neg_avg_forward_ret']*100:.1f}%")
    print(f"  Top discriminating features:")
    for k, diff in summary["top_discriminating"][:5]:
        print(f"    {k}: {diff:+.4f}")

    # ── 3. LLM generation ──
    print("\n[3/4] Generating factors via LLM...")
    existing = [f for f in BASELINE_FACTORS if f not in DISABLED_FACTORS]
    generator = YaoguFactorGenerator(n_factors_per_round=args.n_factors)

    if not generator.configured:
        print("  ERROR: LLM client not configured.")
        print("  Set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable.")
        return

    factors = generator.generate(pairs, existing)

    if not factors:
        print("  No factors generated successfully.")
        return

    print(f"\n  Generated {len(factors)} factors:")
    for gf in factors:
        nodes = gf.tree.node_count()
        depth = gf.tree.depth()
        print(f"    {gf.name} (category={gf.category}, nodes={nodes}, depth={depth})")

    # ── 4. Save ──
    print(f"\n[4/4] Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save in gp_factors.json compatible format
    output_data = {
        "yaogu_factors": [
            {
                "name": gf.name,
                "category": gf.category,
                "expression": gf.expression_json,
                "ic_mean": gf.ic_mean,
                "ic_ir": gf.ic_ir,
                "validation_passed": gf.validation_passed,
                "generation": -2,  # -2 = LLM-generated (vs -1 = GP seed, >=0 = GP gen)
                "discovered_at": date.today().isoformat(),
                "accepted": False,  # needs backtest validation
            }
            for gf in factors
        ],
        "meta": {
            "source": "llm_yaogu_discovery",
            "n_case_pairs": summary["n_pairs"],
            "n_positives": summary["n_positives"],
            "top_discriminating": summary["top_discriminating"][:5],
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(factors)} factors to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
