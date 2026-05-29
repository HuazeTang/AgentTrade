"""Prompt templates for contrastive 妖股 factor generation.

The LLM receives: (1) matched pairs of 妖股 vs 假启动, (2) operator catalog,
(3) expression tree JSON schema. It outputs expression trees that capture
discriminating features between real launches and false alarms.
"""

from __future__ import annotations

from discovery.operators import operator_registry


def _terminal_catalog() -> str:
    return """## Terminal Variables (leaf nodes in expression tree)

| Variable | Type | Description |
|----------|------|-------------|
| `close` | var | Daily close price (后复权) |
| `open` | var | Daily open price |
| `high` | var | Daily high price |
| `low` | var | Daily low price |
| `pre_close` | var | Previous day close |
| `volume` | var | Daily trading volume (shares) |
| `amount` | var | Daily trading amount (yuan) |
| `turnover` | var | Daily turnover rate (%) |
| `ret_5d` | var | 5-day price return (pre-computed derived) |
| `ret_20d` | var | 20-day price return (pre-computed derived) |
| `ret_60d` | var | 60-day price return (pre-computed derived) |
| `vol_20d` | var | 20-day realized volatility |
| `vol_60d` | var | 60-day realized volatility |
| `hl_ratio` | var | High/Low ratio (intraday range proxy) |

All variables are MultiIndex Series indexed by (trade_date, symbol)."""


def _expression_schema() -> str:
    return """## Expression Tree JSON Schema

Each factor is an expression tree in JSON format. Node types:

### Leaf nodes
```json
{"type": "var", "name": "close"}
{"type": "const", "value": 1.0}
```

### Unary operators: neg, abs, log, sqrt, sign
```json
{"type": "unary", "op": "log", "child": {"type": "var", "name": "volume"}}
```

### Binary operators: add, sub, mul, div, max, min, gt, lt
```json
{"type": "binary", "op": "div", "left": {"type": "var", "name": "close"}, "right": {"type": "var", "name": "pre_close"}}
```

### Rolling window operators: ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_delay, ts_rank
```json
{"type": "rolling", "op": "ts_mean", "window": 20, "child": {"type": "var", "name": "volume"}}
```

### Cross-sectional operators: cs_rank, cs_zscore, cs_scale
```json
{"type": "cs", "op": "cs_rank", "child": {"type": "rolling", "op": "ts_std", "window": 20, "child": {"type": "var", "name": "close"}}}
```

### Time-series operators: pct_change, delta, ts_lag
```json
{"type": "ts", "op": "pct_change", "periods": 5, "child": {"type": "var", "name": "close"}}
```

### Group cross-sectional: cs_group_mean, cs_group_zscore (group_field: "board")
```json
{"type": "cs_group", "op": "cs_group_zscore", "group_field": "board", "child": {"type": "var", "name": "close"}}
```

### IMPORTANT RULES
1. All window parameters must be integers >= 1
2. Rolling windows: use 5, 10, 20, 30, 60, 120
3. ts_delay periods: use 1, 5, 10, 21
4. pct_change/delta periods: use 1, 5, 10, 21
5. Always end with cs_rank or cs_zscore for cross-sectional comparability
6. Tree complexity (node count) should be 3-20
7. Tree depth should be 2-5
8. Do NOT use derived variables as leaf nodes in rolling operators unless they are simple scalars"""


def _factor_examples() -> str:
    return """## Expression Tree Examples

### Example 1: Volume surge relative to history
Detects stocks where recent volume is unusually high vs their own 20-day average:
```json
{"type": "binary", "op": "div", "left": {"type": "rolling", "op": "ts_mean", "window": 5, "child": {"type": "var", "name": "volume"}}, "right": {"type": "rolling", "op": "ts_mean", "window": 20, "child": {"type": "var", "name": "volume"}}}
```

### Example 2: Narrow-range consolidation breakout
Detects stocks in a tight range (low amplitude) followed by upward break:
```json
{"type": "binary", "op": "mul", "left": {"type": "unary", "op": "neg", "child": {"type": "rolling", "op": "ts_std", "window": 20, "child": {"type": "ts", "op": "pct_change", "periods": 1, "child": {"type": "var", "name": "close"}}}}, "right": {"type": "ts", "op": "pct_change", "periods": 5, "child": {"type": "var", "name": "close"}}}
```
This multiplies: (negative of 20d return volatility) × (5d return). High when stock was flat but recently surging.

### Example 3: Upward bias with volume confirmation
Detects persistent buying pressure confirmed by volume:
```json
{"type": "binary", "op": "mul", "left": {"type": "rolling", "op": "ts_mean", "window": 5, "child": {"type": "binary", "op": "gt", "left": {"type": "var", "name": "close"}, "right": {"type": "var", "name": "pre_close"}}}, "right": {"type": "binary", "op": "div", "left": {"type": "rolling", "op": "ts_mean", "window": 5, "child": {"type": "var", "name": "volume"}}, "right": {"type": "rolling", "op": "ts_mean", "window": 20, "child": {"type": "var", "name": "volume"}}}}
```
Multiplies: (5d up-day ratio) × (5d/20d volume ratio). High when both buying pressure and volume surge are present."""


SYSTEM_PROMPT = f"""You are an elite quantitative researcher at a hedge fund specializing in A-share momentum-leader (妖股) detection.

## Investment Philosophy: 小亏大赚 (Small Losses, Big Wins)

The goal is NOT to predict every price movement. The goal is to find ASYMMETRIC opportunities:
- When we're WRONG: the stock doesn't launch → small loss (stop-loss or flat return)
- When we're RIGHT: the stock launches (连板) → big win (30%+ in days)

This means the factor must FILTER OUT false positives — stocks that "look ready" but don't actually launch. The key is DISCRIMINATING features.

{_terminal_catalog()}

{operator_registry.to_llm_prompt()}

{_expression_schema()}

{_factor_examples()}

## Factor Design Principles for 妖股 Detection

1. **Volume-price coordination**: Volume precedes price. Look for volume surge BEFORE price breakout.
2. **Compression → Expansion**: Low volatility precedes high volatility. Tight ranges resolve into trends.
3. **Sector resonance**: Individual stocks that lead their sector are more likely to become 妖股.
4. **Persistent buying**: Not just one up day — consecutive up days with volume confirmation.
5. **False positive filters**: Many stocks have similar pre-launch patterns. Find what DISCRIMINATES.

Output format: JSON array of expression tree objects. No explanation, just the JSON."""


def build_contrastive_prompt(
    summary: dict,
    top_pairs: list,
    existing_factors: list[str],
    n_factors: int = 5,
) -> str:
    """Build a contrastive user prompt with matched pair statistics and examples."""

    # ── Summary statistics ──
    lines = [
        "# 妖股 Factor Discovery — Contrastive Analysis",
        "",
        "## Aggregate Statistics",
        f"- Positive cases (妖股): {summary.get('n_positives', 0)} unique stocks",
        f"- Matched pairs total: {summary.get('n_pairs', 0)}",
        f"- Positive avg forward return: {summary['pos_avg_forward_ret']*100:.1f}%",
        f"- Negative avg forward return: {summary['neg_avg_forward_ret']*100:.1f}%",
        "",
        "### Positive vs Negative: Group Averages",
        "",
        "| Metric | 妖股 (Pos) | 假启动 (Neg) | Diff |",
        "|--------|-----------|-------------|------|",
    ]

    pos_avg = summary.get("pos_avg_metrics", {})
    neg_avg = summary.get("neg_avg_metrics", {})
    for k, diff in summary.get("top_discriminating", [])[:10]:
        pv = pos_avg.get(k, 0) or 0
        nv = neg_avg.get(k, 0) or 0
        lines.append(f"| {k} | {pv:.4f} | {nv:.4f} | {diff:+.4f} |")

    # ── Top discriminating features ──
    lines.extend([
        "",
        "## Top Discriminating Features (Pos - Neg, by magnitude)",
        "",
    ])
    for i, (k, diff) in enumerate(summary.get("top_discriminating", [])[:8]):
        direction = "妖股 HIGHER" if diff > 0 else "妖股 LOWER"
        lines.append(f"{i+1}. **{k}**: {diff:+.4f} ({direction})")

    # ── Example pairs ──
    lines.extend([
        "",
        "## Illustrative Case Pairs (妖股 vs 假启动)",
        "",
        "Each pair shows two stocks on the SAME day with SIMILAR pre-launch patterns.",
        "One launched, one didn't. Study the DIFFERENCES.",
        "",
    ])

    for i, pair in enumerate(top_pairs):
        lines.append(f"### Pair {i+1}: {pair.symbol_pos} (LAUNCHED {pair.pos_forward_ret*100:.0f}%) vs {pair.symbol_neg} (flat {pair.neg_forward_ret*100:.0f}%)")
        lines.append(f"Date: {pair.launch_date}")
        lines.append("")
        lines.append("| Metric | 妖股 | 假启动 | Δ |")
        lines.append("|--------|------|--------|---|")
        for k in sorted(pair.pos_metrics.keys()):
            pv = pair.pos_metrics.get(k, 0) or 0
            nv = pair.neg_metrics.get(k, 0) or 0
            dv = pair.diff_metrics.get(k, 0) or 0
            lines.append(f"| {k} | {pv:.4f} | {nv:.4f} | {dv:+.4f} |")
        lines.append("")

    # ── Existing factors ──
    lines.append("## Existing Factors (DO NOT DUPLICATE)")
    for f in existing_factors:
        lines.append(f"- {f}")

    # ── Task ──
    lines.extend([
        "",
        f"## Task",
        f"Based on the contrastive analysis above, design {n_factors} expression tree factors.",
        "Each factor should capture at least ONE discriminating feature between 妖股 and 假启动.",
        "",
        "**Critical requirements:**",
        "1. The factor must rank 妖股 HIGH and 假启动 LOW on the same day",
        "2. Use the EXACT JSON expression tree format from the schema",
        "3. Every tree must end with a cross-sectional normalization (cs_rank or cs_zscore)",
        "4. Do NOT replicate existing factors — create genuinely new signals",
        "5. Keep trees focused: 3-15 nodes, depth 2-4",
        "",
        "Output ONLY a JSON array of expression tree objects. No markdown, no explanation.",
    ])

    return "\n".join(lines)
