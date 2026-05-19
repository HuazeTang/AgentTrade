# GP Factor Discovery — Agent Guide

This document explains how to discover new alpha factors using genetic programming, evaluate them, and persist them so they accumulate across sessions.

## Quick Start

```bash
# Single command: evolve → validate → backtest → save
python run_agent_simulation.py --gp-discover --mode factor \
    --start 2025-10-01 --end 2026-05-18 --cash 100000 \
    --gp-population 200 --gp-generations 25
```

All flags are optional. Defaults: population=200, generations=25, cash=10000, start=2025-10-01, end=2026-04-30.

## What Happens

When you run `--gp-discover`, the pipeline runs these stages sequentially:

### 1. GP Evolution (training phase)

- Splits data 67/33: training period is the first 67% of trading days, backtest is user-specified `--start` to `--end`
- Evolves a population of expression trees using crossover, mutation, tournament selection
- Fitness = weighted combination of IC mean, IC IR, stability, hit rate, minus parsimony penalty
- Existing baseline factors (19 registered factors) are pre-computed and injected as terminals alongside raw price fields (open, high, low, close, volume, amount, etc.)
- **Previously accepted GP factors** from `data/results/gp_factors.json` are also injected as terminals, enabling iterative accumulation
- Early stopping after `early_stop_generations` (default 10) of no improvement

### 2. Validation

Top candidates from evolution are validated against:

- IC significance (t-test on rank IC > 0)
- IC IR minimum (≥ 0.3)
- Hit rate (≥ 0.55)
- Low auto-correlation (≤ 0.95)
- Low max correlation with existing factors (≤ 0.85)
- Walk-forward IC mean (≥ 2% after orthogonal regression)
- Orthogonal filter (Gram-Schmidt style, min residual IR ≥ 0.10) — drops redundant factors
- Category diversity — max 1 factor per category, max 5 total (configurable via `max_new` in code)

### 3. Per-Factor Backtesting

Each validated factor goes through TWO backtests:

**Solo backtest:** baseline 19 factors + only this GP factor. Measures each factor's individual contribution.

**Cumulative backtest:** baseline + GP factors accepted so far (forward feature selection). Measures incremental improvement.

Backtest uses the same strategy as `--mode factor`: MAX_POSITIONS=1, weekly rebalance on Wednesday, 0.1% commission, 0.1% stamp tax on sells, T+1 settlement.

### 4. Acceptance Decision

A factor is accepted if its cumulative Sharpe ratio ≥ previous cumulative Sharpe − 0.01 (small tolerance). This means: accepted as long as it doesn't degrade performance.

### 5. Console Output — Ablation Table

```
=========================================================================================================
                      GP Factor Discovery & Backtest Results
            Training: 2024-08-30 ~ 2025-09-03    Backtest: 2025-10-01 ~ 2026-05-18
=========================================================================================================
Factor                                    Gen      IC  IC_IR  Solo_Ret  Solo_SR Cumul_Ret Cumul_SR    MaxDD Acc
---------------------------------------------------------------------------------------------------------
baseline (19 factors)                      -        -      -   +27.6%    0.96         -         -   -55.3%   -
+ gp_ts_min120_ts_delay120_sign_...        0   0.1018  0.983   +27.6%    0.96    +27.6%     0.96   -55.3%   Y
---------------------------------------------------------------------------------------------------------
Final: 1 GP factors accepted. Baseline: +27.6% → Combined: +27.6% (Δ+0.0%)
Saved: data/results/gp_factors.json (1 active factors, usable as terminals next GP run)
=========================================================================================================
```

### 6. gp_factors.json Output

Saved to `data/results/gp_factors.json` (fixed path). Format:

```json
{
  "gp_factors": [
    {
      "name": "gp_ts_min120_ts_delay120_sign_cs_group_zscore_board_high_ebf099a1",
      "category": "leader",
      "expression": { "type": "rolling", "op": "ts_min", "window": 120, "child": { ... } },
      "generation": 0,
      "fitness": 0.5055,
      "ic_mean": 0.1018,
      "ic_ir": 0.983,
      "ic_std": 0.1035,
      "hit_rate": 0.811,
      "auto_corr": 1.0,
      "max_corr_existing": 0.3379,
      "complexity": 5,
      "depth": 5,
      "validation_passed": true,
      "wf_ic_mean": 0.0831,
      "solo_backtest": {
        "cumulative_return": 0.2759,
        "sharpe_ratio": 0.957,
        "max_drawdown": -0.5526,
        "annualized_return": 0.6896,
        "win_rate": 0.507
      },
      "cumulative_backtest": { ... },
      "discovered_at": "2026-05-19",
      "accepted": true
    }
  ],
  "evolution_history": [
    {
      "generation": 0,
      "best_fitness": 0.1771,
      "mean_fitness": -0.1152,
      "median_fitness": -0.1356,
      "worst_fitness": -0.1635,
      "std_fitness": 0.0688,
      "best_ic": 0.0381,
      "mean_ic": 0.0177,
      "best_ir": 0.2471,
      "mean_ir": -0.0432,
      "best_depth": 2,
      "mean_depth": 2.78,
      "best_nodes": 2,
      "mean_nodes": 3.17,
      "valid_count": 54,
      "total_count": 100,
      "hall_of_fame_size": 10,
      "stall_count": 0,
      "elapsed_seconds": 0.0
    }
  ],
  "meta": {
    "discovered_at": "2026-05-19T15:50:11",
    "training_period": ["2024-08-30", "2025-10-16"],
    "backtest_period": ["2025-10-08", "2026-05-18"],
    "population_size": 100,
    "max_generations": 10,
    "accepted_count": 1,
    "total_candidates": 1,
    "baseline_return": 0.2759,
    "baseline_sharpe": 0.957,
    "final_return": 0.2759,
    "final_sharpe": 0.957
  }
}
```

Only `accepted: true` factors are auto-loaded next time. Rejected factors remain in the file but with `accepted: false` — they serve as a negative log.

## Iterative Factor Accumulation

This is the key design: **factors build on each other across sessions.**

- Run 1 discovers factor A → saved with `accepted: true` → becomes a terminal
- Run 2 evolves with factor A available as a building block → can discover factor B that references A
- Run 3 evolves with A + B available → and so on

Accepted factors are also automatically loaded by `--mode factor` and `--recommend`, so they improve those modes immediately without re-running GP.

To reset: delete `data/results/gp_factors.json`.

## Using Discovered Factors (No Re-discovery)

```bash
# Backtest with all accepted GP factors auto-loaded
python run_agent_simulation.py --mode factor --start 2025-10-01 --end 2026-05-18

# Stock recommendation with GP factors
python run_agent_simulation.py --recommend --recommend-cash 100000 --recommend-top 15
```

No extra flags needed — the code checks for `data/results/gp_factors.json` and loads it automatically.

## Key Files

| File | Role |
|------|------|
| `run_agent_simulation.py` | Main entry point. `Simulation.__init__` takes `gp_discover`, `gp_population`, `gp_generations`. `_run_gp_pipeline()` orchestrates everything. |
| `discovery/gp.py` | `GPEngine` — population evolution, crossover, mutation, tournament selection. `Individual` NamedTuple. `history_to_dict()` for serialization. |
| `discovery/expr.py` | Expression tree types: `VarExpr`, `UnaryOp`, `BinaryOp`, `RollingOp`, `CrossSectionalOp`, `TimeSeriesOp`. Also defines operator sets and terminal fields. |
| `discovery/compiler.py` | `compile_expr()` — compiles an expression tree into a registered Factor class. `compile_and_validate()` — compiles then validates against Factor base class. |
| `discovery/operators.py` | Operator implementations (rolling mins, z-scores, cross-sectional ranks, time-series delays, etc.) |
| `discovery/validate.py` | `FactorValidator` — runs IC, IR, hit rate, auto-corr, correlation, walk-forward checks. `orthogonal_filter()` — Gram-Schmidt residual IR filtering. |
| `factor/engine.py` | `FactorEngine.compute()` — computes named factors over daily DataFrame. |
| `factor/registry.py` | `registry` — global Factor class registry. `registry.register()` / `registry.register_all()`. |
| `data/results/gp_factors.json` | Persistent factor pool. Auto-loaded by `--mode factor`, `--recommend`, and `--gp-discover`. |

## Internal Architecture (How the Code Works)

### Factor Computation Order

GP factors can reference baseline factor columns. To handle this, `_compute_factor_set_staged()` runs in two phases:

1. Compute baseline factors first → inject their columns into the raw data DataFrame
2. Compute GP factors on the enriched DataFrame (so they see baseline columns as input)

This prevents the "GP factor depends on baseline columns that don't exist yet" error.

### GP Terminals Setup

Inside `_run_gp_pipeline()`, the terminal set for GP evolution is:

```
TERMINAL_FIELDS (raw price fields: open, high, low, close, volume, amount, etc.)
+ baseline factor names (e.g. momentum_20, volatility_60, rsi_14, ...)
+ prior accepted GP factor names (from gp_factors.json)
```

Factor values are pre-computed and injected as columns into `gp_data`, so VarExpr nodes can reference them. The `extended_terminals` list tells the random expression generator which terminal names are valid.

### Validation Pipeline

```
FactorValidator.validate() runs these checks:
  1. compute_rank_ic() → Rank IC, IC IR, IC std, hit rate
  2. ic t-test → IC significantly > 0?
  3. auto_corr → stability check
  4. max corr with existing factors → uniqueness check
  5. Walk-forward IC → out-of-sample predictive power
  6. Orthogonal regression → residual IC after regressing out existing factors
Result.passed = all checks pass
```

### Backtest Loop

`_run_backtest_loop()` simulates day-by-day:
- Each trading day: compute composite score = sum(rank-normalized factor values × IC_IR-calibrated weights)
- Sell if current position's rank drops below SELL_RANK_LIMIT (5)
- Buy top-ranked stock if no position (subject to MKT_DD_THRESHOLD, which is currently disabled)
- Apply T+1 settlement, 0.1% commission, 0.1% stamp tax on sells, lot-size rounding (100 shares)

## Configuration Points

In `run_agent_simulation.py` top-level constants:

```python
MAX_POSITIONS = 1           # single-position strategy
REBALANCE_FREQ = "weekly"   # rebalance on Wednesdays
SELL_RANK_LIMIT = 5         # sell when rank falls below 5 (validates signal decay)
FORWARD_PERIODS = 5         # 5-day forward return for IC computation
TAKE_PROFIT_PCT = 0.25      # 25% take-profit
STOP_LOSS_PCT = None        # stop-loss disabled
```

In `_run_gp_pipeline()`, `GPConfig`:

```python
GPConfig(
    population_size=200,      # --gp-population
    max_generations=25,       # --gp-generations
    tournament_size=7,
    crossover_prob=0.7,
    mutation_prob=0.5,
    elite_count=10,
    max_depth=7,
    max_complexity=40,
    early_stop_generations=10,
    parsimony_penalty=0.0003,
    ic_mean_weight=0.25,
    ic_ir_weight=0.35,
    stability_weight=0.25,
    hit_rate_weight=0.15,
)
```

## Practical Tips

1. **Start small for iteration speed**: `--gp-population 100 --gp-generations 10` runs in ~5-10 minutes. Increase for production runs.
2. **Check the ablation table first**: If all factors show Δ+0.0%, the GP isn't finding incremental alpha. Try a different backtest period or increase generations.
3. **evolution_history tells you if GP is learning**: best_fitness and mean_fitness should trend up across generations. If they plateau, increase population or generations.
4. **accepted: false factors aren't lost**: They stay in gp_factors.json for audit trail. You can manually flip `accepted: true` if you disagree with the auto-decision.
5. **Factor pool can be curated**: Edit gp_factors.json to remove poor factors or add hand-crafted ones. The expression format is the serialized Expr tree.
6. **Data requirement**: Needs `data/cache/daily/` populated (run `python download_all_main.py` first if empty). Required columns per stock: open, high, low, close, volume, amount.

## Other Useful Commands

```bash
# Compare weekday rebalancing (Wed is optimal for A-shares)
python compare_weekdays.py

# Full research report with GP discovery + charts
python run_agent_report.py

# Single stock deep-dive analysis (charts saved to data/results/sim_*/)
python run_agent_simulation.py --analyze 603629

# LLM vs Factor comparison
python run_agent_simulation.py --mode compare --model qwen-max
```
