# GP Engine Architecture

## Overview

GP factor discovery with two-stage fitness: fast IC evaluation (parallel, all individuals) + real backtest validation (parallel, top-N elites per generation).

## Flow

```
evolve() per-generation:
  ┌─────────────────────────────────────────────────────┐
  │ 1. Tournament select + crossover + mutate          │
  │ 2. IC evaluate 400 individuals (ThreadPool, 8w)    │  ~15s
  │ 3. Sort by IC fitness                              │
  │ 4. backtest_callback(population, gen)              │
  │    ├─ Take top-5 valid elites                      │
  │    ├─ Batch-compile + compute factor values        │
  │    ├─ Run 5 solo backtests (ThreadPool, 4w)        │  ~2s
  │    └─ Blend: fitness = 0.7×IC + 0.3×Sharpe_norm   │
  │ 5. Re-sort population (backtest may re-rank)       │
  │ 6. Update hall-of-fame, log, check early stop      │
  └─────────────────────────────────────────────────────┘

Post-GP per-generation:
  ┌─────────────────────────────────────────────────────┐
  │ 1. Validate top candidates (walk-forward IC)       │
  │ 2. Orthogonal + diversity filter                   │
  │ 3. Solo backtests (ThreadPool, 4w)                 │  并行
  │ 4. Cumulative backtests (ThreadPool, 4w)           │  并行
  │    └─ Pre-compute all GP factor values once        │
  │    └─ Each task subsets columns from cache         │
  │    └─ If mid-list rejection: re-run affected        │
  │ 5. Second-pass retest rejected with full set        │  并行
  │ 6. Save gp_factors.json, generate report           │
  └─────────────────────────────────────────────────────┘
```

## Key Design Decisions

### IC fitness (fast, all 400)
- IC mean (0.4) + IC IR (0.2) + stability (0.2) + hit rate (0.2)
- Hard-reject: auto_corr > 0.95, IC std < 0.005, depth <= 1
- ThreadPoolExecutor (8 workers) — Factor classes from exec() not pickleable, but pandas releases GIL

### Backtest fitness (slow, top-5 only)
- Runs real AgentSimulation backtest loop (broker, universe filter, T+1, price limits)
- Sharpe ratio normalized: `(sharpe + 1) / 4` clamped to [0, 1]
- Blended: `fitness = 0.7 * ic_fitness + 0.3 * norm_sharpe`
- Catches IC false positives (near-constant signals with high IC but poor trading)

### Cumulative backtest parallelization
- Pre-compute baseline + all GP factor values in one batch
- BacktestTask.factor_cache: subset columns instead of recomputing from scratch
- First pass: run all cumulative combinations in parallel (assume all accepted)
- If mid-list factor rejected: re-run remaining factors with corrected set
- Second pass (retest): also parallel

### Thread safety
- ThreadPoolExecutor (not ProcessPoolExecutor) — compiled Factor classes are not pickleable
- Each worker creates a fresh AgentSimulation instance (isolated accountant, broker, journal)
- Market data is read-only across all workers

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_workers` | 8 | IC evaluation thread pool size |
| `backtest_top_n` | 5 | Number of elites to backtest per generation |
| `backtest_weight` | 0.3 | Weight of backtest Sharpe in blended fitness |
| `backtest.parallel max_workers` | 4 | Solo/cumulative backtest thread pool size |

## Log Output

```
Gen   3: best_fitness=0.2341 IC=0.0450 IR=1.230 depth=4 nodes=8 (gp_ts_std_volume_abc123) btSR=1.234 [18.2s]
```

`btSR` only appears when backtest_callback ran (non-zero Sharpe).
