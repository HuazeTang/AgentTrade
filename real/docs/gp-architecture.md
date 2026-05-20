# A Two-Phase Framework for Factor Discovery via Genetic Programming with Pure Factor Evaluation

## Abstract

We present a genetic programming (GP) framework for discovering alpha factors in the
A-share equity market. The key design principle is the **separation of factor quality
assessment from trading strategy optimization**. Factor candidates are evaluated in a
*factor mimicking portfolio* (FMP) framework that strips away all trading frictions
(position limits, T+1 settlement, stop-loss, transaction costs), isolating each
factor's pure predictive power. Strategy search is deferred to a separate second phase,
ensuring that the factor library remains uncontaminated by strategy-specific overfitting.

## 1. Architecture Overview

The system operates in two phases:

- **Phase A — Factor Discovery (GP)**: Evolves factor expressions via genetic
  programming. Each individual is evaluated by (i) its Information Coefficient (IC)
  on in-sample data, and (ii) the risk-adjusted return of a factor mimicking portfolio
  constructed from cross-sectionally standardized factor values. No trading simulation
  is involved.
- **Phase B — Strategy Search (future)**: Starting from the verified factor library,
  searches for optimal trading rules (position sizing, rebalancing frequency,
  take-profit/stop-loss thresholds). This phase treats factors as fixed inputs and
  optimizes only the execution layer.

```
┌─────────────────────────────────────────────────────┐
│                    Phase A: GP Factor Discovery      │
│                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ GP evolve │───>│ IC fitness   │───>│ Pure FMP  │  │
│  │ (crossover│    │ (parallel,   │    │ callback  │  │
│  │  mutation)│    │  all inds)   │    │ (top-N)   │  │
│  └──────────┘    └──────────────┘    └───────────┘  │
│                         │                  │         │
│                         └──── blended ─────┘         │
│                                fitness               │
│                                                     │
│  Post-GP: Validation → Ortho Filter → Walk-Forward  │
│           Pure Solo → Pure Cumulative → Accept/Reject│
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              Phase B: Strategy Search (future)       │
│                                                     │
│  Factor Library ──> Strategy Optimizer ──> Trading   │
│                     (grid/RL/GP)           Rules     │
└─────────────────────────────────────────────────────┘
```

## 2. Genetic Programming Engine

### 2.1 Individual Representation

Each individual is a syntax tree where leaf nodes are market data fields (OHLCV) or
existing factor values, and internal nodes are arithmetic, rolling-window,
cross-sectional, and time-series operators (see Appendix A for the full operator set).

```python
Individual = (tree, factor_name, factor_cls, fitness,
              ic_mean, ic_ir, hit_rate, auto_corr,
              complexity, depth, generation)
```

Notably, `Individual` does **not** carry any strategy parameters (e.g., sell thresholds,
position limits). This enforces the principle that factor quality is assessed
independently of trading rules.

### 2.2 Initialization

The initial population is generated via *ramped half-and-half*: half the individuals
are full trees (all branches reach `max_depth`), half are grown with random stopping.
Tree depths are ramped from 2 to `max_depth` to ensure structural diversity.
LLM-proposed seed factors can optionally replace the first *k* individuals.

### 2.3 Genetic Operators

| Operator | Probability | Description |
|----------|------------|-------------|
| Subtree crossover | 0.70 | Swap random non-root subtrees between two parents |
| Subtree mutation | 0.20 × `mutation_prob` | Replace a random subtree with a new random tree |
| Window mutation | 0.15 × `mutation_prob` | Change the window/period of a rolling or time-series operator |
| Operator mutation | 0.15 × `mutation_prob` | Replace an operator with a different one of the same arity |
| Constant mutation | 0.10 × `mutation_prob` | Perturb a constant node by ±20% |
| Unary mutation | 0.10 × `mutation_prob` | Insert or remove a unary operator (e.g., log, sqrt) |
| Wrap-rolling mutation | 0.30 × `mutation_prob` | Wrap a subtree with a rolling or time-series operator |

Selection is tournament selection with size *k* = 7. Elitism preserves the top *e* = 10
individuals each generation.

### 2.4 IC-based Fitness Evaluation

All individuals are evaluated in parallel via `ThreadPoolExecutor` (threads, not
processes, since compiled `Factor` classes are not pickleable). The fitness of
individual *i* is:

$$F_{IC}(i) = w_1 \cdot |\overline{IC}| + w_2 \cdot \min(IR, 5) + w_3 \cdot IC_{min\_period} + w_4 \cdot HR$$

where:

| Symbol | Weight | Description |
|--------|--------|-------------|
| $\overline{IC}$ | $w_1 = 0.25$ | Mean rank IC (Spearman) across all trading days |
| $IR$ | $w_2 = 0.35$ | Information Ratio: $\overline{IC} / \sigma(IC)$, capped at 5 |
| $IC_{min\_period}$ | $w_3 = 0.25$ | Minimum sub-period IC (stability across regimes) |
| $HR$ | $w_4 = 0.15$ | Hit rate: fraction of days with $IC > 0$ |

A parsimony penalty $\lambda \cdot \text{complexity}$ ($\lambda = 0.0003$) is subtracted.

**Hard rejection criteria** (individual is assigned fitness = −999):
- Depth ≤ 1 (trivial raw field or constant)
- IC standard deviation < 0.005 (near-constant signal generating spuriously high IR)
- Auto-correlation > 0.95 (effectively constant factor)
- Compilation or computation failure

## 3. Pure Factor Evaluation (Factor Mimicking Portfolio)

### 3.1 Motivation

Traditional backtest-based evaluation confounds factor quality with strategy
parameters (position sizing, stop-loss thresholds, rebalancing frequency). A factor
with excellent predictive power may receive a poor backtest Sharpe simply because the
strategy parameters were suboptimal — and vice versa. This conflation biases the GP
toward factors that happen to fit the specific trading rules rather than factors with
genuine predictive power.

### 3.2 Factor Mimicking Portfolio Construction

Given a factor value $f_{t,i}$ for stock $i$ at date $t$, the mimicking portfolio
weight is:

$$z_{t,i} = 2 \cdot \left( \text{rank}_{pct}(f_{t,\cdot})_i - 0.5 \right)$$

$$w_{t,i} = \frac{z_{t,i}}{\sum_j |z_{t,j}|} \cdot L$$

where $\text{rank}_{pct}$ is the cross-sectional percentile rank (0 to 1), $z_{t,i} \in [-1, 1]$ is
the centered rank score, and $L = 1.0$ is the total leverage (sum of absolute weights).

Using rank-percentile normalization rather than z-score avoids Gaussian assumptions and
is robust to outliers. The portfolio is dollar-neutral (long-short) with daily
rebalancing.

The daily portfolio return is:

$$r_t^{FMP} = \sum_i w_{t,i} \cdot r_{t \to t+1, i}$$

where $r_{t \to t+1, i}$ is stock $i$'s forward return from date $t$ to $t+1$ (the
caller is responsible for ensuring no look-ahead bias, e.g., shifting factor values so
that date $t$'s factor uses only information available at date $t-1$).

### 3.3 Pure Factor Metrics

For each factor, we compute:

| Metric | Description |
|--------|-------------|
| IC Mean | Mean daily rank IC (Spearman) |
| IC IR | IC Mean / IC Std |
| IC Hit Rate | Fraction of days with positive IC |
| Sharpe Ratio | $\bar{r} / \sigma(r) \cdot \sqrt{252}$, annualized |
| Max Drawdown | Maximum peak-to-trough decline |
| Cumulative Return | $(1 + r_1)(1 + r_2)...(1 + r_T) - 1$ |
| Annualized Return | $\bar{r} \cdot 252$ |
| Volatility | $\sigma(r) \cdot \sqrt{252}$ |
| Win Rate | Fraction of days with positive return |
| IC Dispersion | Standard deviation of IC across equal-duration sub-periods |

### 3.4 Multi-Factor Composite

For evaluating a set of $K$ factors together (cumulative evaluation), we construct a
weighted composite. Each factor's values are rank-normalized, then combined via
IC-calibrated weights $v_k$:

$$z_{t,i}^{composite} = \sum_{k=1}^K v_k \cdot z_{t,i}^{(k)}$$

where $v_k$ are calibrated on the training period via `_calibrate_factor_weights_standalone`
(IC × IC_IR weighting). The composite score is then converted to portfolio weights using
the same rank-percentile normalization.

## 4. Per-Generation Pure Factor Callback

During GP evolution, after IC evaluation and ranking, a callback evaluates the top-$N$
individuals ($N = 5$) via factor mimicking portfolios. The callback maintains a
running set of *provisionally accepted* factors across generations.

For each elite candidate $c$, it computes the cumulative FMP Sharpe with all
provisionally accepted factors plus $c$:

$$\Delta SR_c = SR(\text{accepted} \cup \{c\}) - SR(\text{accepted})$$

The blended fitness is:

$$F_{blend}(c) = (1 - \alpha) \cdot F_{IC}(c) + \alpha \cdot \text{clip}\left(\frac{\Delta SR_c + 0.3}{0.8}, 0, 1\right)$$

where $\alpha = 0.3$ is the blend weight. If $\Delta SR_c > 0.001$, candidate $c$ is
provisionally accepted and added to the running set for subsequent candidates.

This ensures the GP's fitness function incorporates *actual portfolio performance* of
the factor set, not just the correlation structure of individual factors.

## 5. Post-GP Validation Pipeline

After GP evolution completes, the top candidates undergo multi-stage validation:

### 5.1 Walk-Forward Validation

Rolling-window out-of-sample test with configurable window and step sizes:

```
For window i from 0 to N:
  Train: [start + i·step, start + i·step + window)
         → Compute IC metrics
  Test:  [start + i·step + window, start + (i+1)·step + window)
         → Construct FMP, compute OOS Sharpe
```

A factor passes walk-forward if `mean_test_sharpe > 0` and `min_test_sharpe > −0.5`
across at least `min_windows` out-of-sample periods.

### 5.2 Orthogonal Filter

Candidates are tested for orthogonality against existing factors. A candidate is
rejected if its *residual IC ratio* — the IC of the candidate's residuals after
regressing out existing factors — falls below 0.10 of its raw IC.

### 5.3 Pure Solo Evaluation

Each validated candidate is evaluated individually via FMP, producing `pure_solo`
metrics (Sharpe, max drawdown, cumulative return, etc.).

### 5.4 Pure Cumulative Evaluation

Candidates are stacked sequentially. For the $j$-th candidate, an IC-weighted composite
of all previously accepted factors plus the candidate is constructed, and the FMP
Sharpe is computed:

- **First candidate**: accepted if cumulative Sharpe ≥ 0.3
- **Subsequent candidates**: accepted if cumulative Sharpe ≥ previous cumulative Sharpe + 0.02

This greedy forward selection ensures each accepted factor contributes incremental
improvement to the factor set.

### 5.5 Persistence

Accepted factors are persisted to `gp_factors.json` with full metadata (expression tree,
IC metrics, pure solo/cumulative metrics, walk-forward results). The file is **merged**
across runs — previously discovered factors are preserved, and re-discovered factors
keep their best fitness. Prior GP factors are also injected as terminals in subsequent
GP runs, enabling iterative factor accumulation.

## 6. Parallelization Strategy

| Stage | Mechanism | Workers |
|-------|-----------|---------|
| GP population evaluation | `ThreadPoolExecutor` | `min(max_workers, cpu_count)` |
| Per-generation pure FMP callback | Sequential (acceptance dependency chain) | 1 |
| Post-GP solo evaluation | Sequential (each factor evaluated independently) | 1 |

Threads rather than processes are used throughout because compiled `Factor` classes
(via Python's `exec()`) are not pickleable. Pandas/NumPy release the GIL during
computation, yielding effective parallelism for IC evaluation.

## 7. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 400 | Number of individuals per generation |
| `max_generations` | 30 | Maximum generations before forced stop |
| `tournament_size` | 7 | Tournament selection pressure |
| `crossover_prob` | 0.7 | Probability of crossover per parent pair |
| `mutation_prob` | 0.5 | Probability of mutation per child |
| `elite_count` | 10 | Number of elites preserved each generation |
| `max_depth` | 7 | Maximum tree depth |
| `max_complexity` | 40 | Maximum node count |
| `early_stop_generations` | 10 | Stop if no improvement for N generations |
| `parsimony_penalty` | 0.0003 | Fitness penalty per node |
| `max_workers` | 8 | Thread pool size for IC evaluation |
| `pure_factor_top_n` | 5 | Number of elites to evaluate via FMP per generation |
| `pure_factor_blend_weight` | 0.3 | Weight of pure Sharpe in blended fitness ($\alpha$) |

## 8. Key Design Decisions

### 8.1 Rank-percentile normalization over z-score
Rank-percentile normalization is non-parametric and robust to outliers, avoiding the
distributional assumptions of z-score normalization. Each stock receives a weight in
$[-1, 1]$ regardless of the factor's distribution shape.

### 8.2 IC fitness + pure FMP blend over full simulation
Full trading simulation introduces confounding variables (T+1 settlement, price limits,
lot-size rounding, broker commissions, stamp tax) that obscure the factor's actual
predictive power. By evaluating factors in a frictionless FMP framework, the GP
selects for genuine alpha rather than strategy fit.

### 8.3 Greedy forward acceptance over global optimization
Cumulative acceptance uses greedy forward selection rather than combinatorial search
over all subsets. While suboptimal in theory, this scales linearly with the number of
candidates and produces interpretable acceptance decisions (each factor's marginal
contribution is explicit).

### 8.4 Thread-level parallelism over process-level
Process-based parallelism would require serializing compiled `Factor` classes, which
is impossible due to `exec()`-based compilation. Thread-based parallelism works because
the compute-intensive operations (pandas correlations, NumPy operations) release the GIL.

## Appendix A: GP Operator Set

### Arithmetic Operators
`+`, `−`, `×`, `÷`, `sqrt`, `log`, `abs`, `square`, `cube`, `sign`, `neg`, `inv`,
`log1p`, `min`, `max`, `clip`

### Rolling Window Operators (windows: 5, 10, 20, 60, 120)
`mean`, `std`, `max`, `min`, `sum`, `skew`, `kurt`, `median`, `mad`, `zscore`,
`prod`, `ptp`, `quantile`

### Time Series Operators (periods: 5, 10, 20, 60, 120)
`ts_delta`, `ts_pct`, `ts_delay`, `ts_corr`, `ts_rank`, `ts_mean`, `ts_std`,
`ts_max`, `ts_min`, `ts_zscore`, `ts_regression_residual`

### Cross-Sectional Operators
`cs_rank`, `cs_zscore`, `cs_scale`, `cs_regression_residual`

### Grouped Cross-Sectional Operators (groups: industry)
`cs_group_mean`, `cs_group_zscore`

### Ternary Operator
`if_else(a, b, c)`: returns `b` where `a >= 0`, else `c` (element-wise)

### Terminals (leaf nodes)
OHLCV fields, turnover, amount, market_cap, pe_ratio, pb_ratio, industry, plus any
baseline or previously accepted GP factor values.

---

*Generated by Claude Code, May 2026.*
