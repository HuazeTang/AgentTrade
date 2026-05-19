# GP Factor Discovery — Agent Runbook

## Quick Start

```bash
# Full pipeline: LLM seed → evolve → validate → backtest → save
python run_agent_simulation.py --gp-discover \
    --start 2025-10-08 --end 2026-05-18 --cash 100000 \
    --gp-population 200 --gp-generations 25

# Fast smoke test (5-10 min)
python run_agent_simulation.py --gp-discover \
    --gp-population 50 --gp-generations 5

# Factor backtest (auto-loads gp_factors.json)
python run_agent_simulation.py --mode factor \
    --start 2025-10-08 --end 2026-05-18 --cash 100000

# Stock recommendation (auto-loads GP factors)
python run_agent_simulation.py --recommend --recommend-cash 100000 --recommend-top 15
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gp-discover` | off | Run GP evolution + per-factor backtest + save |
| `--gp-population` | 200 | GP population size |
| `--gp-generations` | 25 | GP max generations |
| `--llm-seed` | on | Use LLM to seed GP initial population (requires API key) |
| `--no-llm-seed` | — | Disable LLM seeding |
| `--mode factor` | — | Backtest with baseline + persisted GP factors |
| `--recommend` | — | Output tomorrow's top-N buy candidates |
| `--start/--end` | 2025-10-01/2026-04-30 | Backtest date range |
| `--cash` | 10000 | Initial cash |

## Pipeline Stages (--gp-discover)

### 0. LLM Seed Generation (new)
If `--llm-seed` is on and an API key is configured (DEEPSEEK_API_KEY/QWEN_API_KEY/OPENAI_API_KEY in `.env`):
1. Computes per-factor IC/IR/stability metrics on training data
2. Identifies dead/decaying/weak factors and correlation clusters
3. Prompts LLM to propose 5 new factor formulas addressing specific weaknesses
4. Parses expression hints into Expr trees, compiles to Factor classes
5. Injects seeds into GP initial population (replaces first N random individuals)

Without API key → logged as "LLM seeding skipped" and proceeds normally.

### 1. GP Evolution
- Splits data 67/33: first 67% of trading days = training
- 36 operators across 6 categories (arithmetic, rolling, cross-sectional, time-series, grouped cross-sectional, conditional)
- **New non-linear operators:** `ts_skew`, `ts_kurt`, `ts_corr`, `ts_quantile`, `ts_ema`, `ts_prod`, `cs_regression_residual`, `if_then`
- Previously accepted GP factors from `data/results/gp_factors.json` are injected as terminals
- **Strategy co-evolution:** each individual also carries a `StrategyGene` with `sell_rank_limit` (2-20), evolved alongside the factor expression via mutation and crossover

### 2. Validation
Five checks: IC significance, IC IR ≥ 0.15, auto-corr ≤ 0.95, max corr with existing ≤ 0.7, walk-forward IC ≥ 0.005. Then orthogonal filter (Gram-Schmidt) + category diversity (max 1 per category).

### 3. Per-Factor Backtesting
- **Solo:** baseline 22 factors + single GP factor (runs in parallel via ThreadPoolExecutor for >1 factor)
- **Cumulative:** forward feature selection — add factors one by one, keep those that don't degrade cumulative Sharpe
- Strategy gene's `sell_rank_limit` is used during backtest

### 4. Acceptance Decision
Factor accepted if cumulative Sharpe ≥ previous cumulative Sharpe − 0.01.

### 5. Console Output — Ablation Table

```
=========================================================================================================
                      GP Factor Discovery & Backtest Results
            Training: 2024-08-30 ~ 2025-09-03    Backtest: 2025-10-08 ~ 2026-05-18
=========================================================================================================
Factor                                    Gen      IC  IC_IR  Solo_Ret  Solo_SR Cumul_Ret Cumul_SR    MaxDD Acc
---------------------------------------------------------------------------------------------------------
baseline (22 factors)                      -        -      -   +27.6%    0.96         -         -   -55.3%   -
+ gp_ts_skew30_pct_change5_close_xxx       3   0.0850  0.720   +30.1%    1.05    +30.1%     1.05   -52.1%   Y
+ gp_if_then_momentum_reversal_xxx         7   0.0720  0.650   +28.5%    0.99    +33.2%     1.15   -48.3%   Y
---------------------------------------------------------------------------------------------------------
Final: 2 GP factors accepted. Baseline: +27.6% → Combined: +33.2% (Δ+5.6%)
Saved: data/results/gp_factors.json (2 active factors, usable as terminals next GP run)
=========================================================================================================
```

### 6. gp_factors.json Output

Saved to `data/results/gp_factors.json`. Each factor now includes `strategy_gene`:

```json
{
  "gp_factors": [
    {
      "name": "gp_ts_skew30_pct_change5_close_a1b2c3d4",
      "category": "momentum",
      "expression": { "type": "rolling", "op": "ts_skew", "window": 30, "child": {...} },
      "generation": 3,
      "fitness": 0.3200,
      "ic_mean": 0.0850,
      "ic_ir": 0.720,
      "strategy_gene": { "sell_rank_limit": 7 },
      "solo_backtest": { "cumulative_return": 0.301, "sharpe_ratio": 1.05, "max_drawdown": -0.521 },
      "cumulative_backtest": { "cumulative_return": 0.301, "sharpe_ratio": 1.05, "max_drawdown": -0.521 },
      "discovered_at": "2026-05-19",
      "accepted": true
    }
  ],
  "meta": {
    "accepted_count": 2,
    "total_candidates": 3,
    "baseline_return": 0.276,
    "baseline_sharpe": 0.957,
    "final_return": 0.332,
    "final_sharpe": 1.15
  }
}
```

## New Features (May 2026)

### 36 Operators (was 28)

| Category | Operators |
|----------|-----------|
| Arithmetic (unary) | `neg`, `abs`, `log`, `sqrt`, `sign` |
| Arithmetic (binary) | `add`, `sub`, `mul`, `div`, `max`, `min`, `gt`, `lt` |
| Rolling (single-child) | `ts_sum`, `ts_mean`, `ts_std`, `ts_min`, `ts_max`, `ts_delay`, `ts_rank`, **`ts_skew`***, **`ts_kurt`***, **`ts_quantile`***, **`ts_ema`***, **`ts_prod`*** |
| Rolling (binary) | **`ts_corr`*** |
| Cross-sectional | `cs_rank`, `cs_zscore`, `cs_scale`, **`cs_regression_residual`*** |
| Grouped CS | `cs_group_mean`, `cs_group_zscore` |
| Time-series | `pct_change`, `delta`, `ts_lag` |
| Conditional (ternary) | **`if_then`*** |

*New operators

### Strategy Co-Evolution
- `StrategyGene` with `sell_rank_limit ∈ [2, 20]` evolves alongside factor expression
- Lower limit (3-5): faster rotation, good for short-term signals
- Higher limit (12-15): hold longer, good for slow-moving factors
- `--mode factor` automatically loads the strategy_gene from `gp_factors.json`

### Parallel Backtesting
- Solo backtests run in parallel via `ThreadPoolExecutor` (4 workers by default)
- Located in `backtest/parallel.py` — `BacktestTask` + `run_parallel_backtests()`

### LLM Seed Generation
- Analyzes baseline factor weaknesses via IC trends and correlation clusters
- Proposes targeted factor formulas, parsed into Expr trees
- Injected as seed individuals into GP initial population
- Disable with `--no-llm-seed`; requires API key in `.env`

## Key Files

| File | Role |
|------|------|
| `run_agent_simulation.py` | Main entry point. `_run_gp_pipeline()` orchestrates everything. |
| `discovery/gp.py` | `GPEngine`, `Individual`, `StrategyGene`, evolution loop |
| `discovery/expr.py` | Expression tree types (8 classes including `TernaryOp`), operator sets |
| `discovery/operators.py` | `OperatorMeta` definitions for all 36 operators |
| `discovery/compiler.py` | `compile_expr()` — compiles tree to Factor class |
| `discovery/validate.py` | `FactorValidator`, `orthogonal_filter()` |
| `discovery/llm_seed.py` | **New** — `LLMSeedGenerator` (analyze + propose + compile) |
| `backtest/parallel.py` | **New** — `BacktestTask` + `run_parallel_backtests()` |
| `factor/engine.py` | `FactorEngine.compute()` — topological sort + computation |
| `data/results/gp_factors.json` | Persistent factor pool with strategy_genes |

## Position Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITIONS` | 1 | Single-position strategy |
| `SELL_RANK_LIMIT` | 5 | Sell when rank drops below this (overridden by strategy_gene) |
| `MAX_POSITION_PCT` | 0.95 | Max equity per position |
| `REBALANCE_FREQ` | "weekly" | Rebalance on Wednesdays |
| `FORWARD_PERIODS` | 5 | 5-day forward return for IC |
| `TAKE_PROFIT_PCT` | 0.25 | 25% take-profit trigger |

## Practical Tips

1. **Start fast**: `--gp-population 50 --gp-generations 5` runs in ~3-5 min
2. **Check the ablation table**: if all Δ+0.0%, GP isn't finding alpha — try a different date range or increase generations
3. **evolution_history** in gp_factors.json shows if GP is learning (fitness trending up)
4. **Disable LLM** if no API key or for faster runs: `--no-llm-seed`
5. **Reset factor pool**: delete `data/results/gp_factors.json`
6. **Manually edit gp_factors.json**: flip `accepted: true/false`, adjust `sell_rank_limit`, or add hand-crafted factors

## Data Requirements

Needs `data/cache/daily/` populated. Run `python download_all_main.py` first if empty.
Required columns: open, high, low, close, volume, amount, pre_close, turnover, tradable_shares, market_cap, board.
