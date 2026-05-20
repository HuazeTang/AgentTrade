"""LLM-assisted seed factor generation for GP discovery.

Analyzes baseline factor weaknesses via IC trends and correlation clusters,
then asks an LLM to propose hand-crafted factor formulas as seed individuals
for the GP initial population.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from discovery.expr import (
    Expr, VarExpr, ConstExpr, UnaryOp, BinaryOp,
    RollingOp, CrossSectionalOp, TimeSeriesOp, TernaryOp,
    ROLLING_OPS, ROLLING_OPS_BINARY, ROLLING_WINDOWS, UNARY_OPS, BINARY_OPS,
    CS_OPS, TS_OPS, TS_PERIODS, TERMINAL_FIELDS,
)
from discovery.operators import operator_registry

if TYPE_CHECKING:
    from agent.llm_client import LLMClient
    from discovery.gp import Individual

logger = logging.getLogger(__name__)


class LLMSeedGenerator:
    """Generate seed individuals from LLM-suggested factor ideas."""

    # Simple expression_hint parser patterns
    _ROLLING_PATTERN = re.compile(
        r'(ts_\w+)\((\w+),\s*(\d+)\)'
    )
    _UNARY_PATTERN = re.compile(
        r'(neg|abs|log|sqrt|sign)\((\w+)\)'
    )
    _BINARY_PATTERN = re.compile(
        r'(add|sub|mul|div|max|min|gt|lt)\((\w+),\s*(\w+)\)'
    )
    _CS_PATTERN = re.compile(
        r'(cs_\w+)\((\w+)\)'
    )
    _TS_PATTERN = re.compile(
        r'(pct_change|delta|ts_lag)\((\w+),\s*(\d+)\)'
    )

    def __init__(self, model: str | None = None):
        self.model = model

    # ── Main pipeline ──────────────────────────────────────────────────────

    def analyze_baseline_weakness(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.Series,
        llm: "LLMClient",
    ) -> dict:
        """Compute per-factor IC stats and feed diagnosis to LLM.

        Returns a structured diagnosis dict with:
          - per_factor: {name: {ic_mean, ic_ir, trend_slope, status}}
          - correlation_clusters: list of factor groups with high mutual correlation
          - summary: natural-language summary from LLM
        """
        from factor.validation import compute_rank_ic, ic_summary, factor_correlation

        per_factor = {}
        for col in factor_df.columns:
            try:
                ic_series = compute_rank_ic(factor_df[col], forward_returns)
                ic_s = ic_summary(ic_series.dropna())
                # Simple trend: linear slope of IC over time
                ic_clean = ic_series.dropna()
                if len(ic_clean) >= 20:
                    x = np.arange(len(ic_clean))
                    slope, _ = np.polyfit(x, ic_clean.values, 1)
                else:
                    slope = 0.0

                status = "healthy"
                if abs(ic_s.get("mean", 0)) < 0.01:
                    status = "dead"
                elif slope < -0.0005:
                    status = "decaying"
                elif abs(ic_s.get("mean", 0)) < 0.02:
                    status = "weak"

                per_factor[col] = {
                    "ic_mean": round(ic_s.get("mean", 0), 4),
                    "ic_ir": round(ic_s.get("ir", 0), 3),
                    "hit_rate": round(ic_s.get("hit_rate", 0), 3),
                    "trend_slope": round(slope, 6),
                    "status": status,
                }
            except Exception:
                per_factor[col] = {"ic_mean": 0, "ic_ir": 0, "hit_rate": 0,
                                   "trend_slope": 0, "status": "error"}

        # Correlation clusters (simple: pairs with corr > 0.5)
        corr_clusters = []
        try:
            corr_matrix = factor_correlation(factor_df)
            if corr_matrix is not None and not corr_matrix.empty:
                high_corr_pairs = []
                seen = set()
                for i, col_i in enumerate(corr_matrix.columns):
                    for j, col_j in enumerate(corr_matrix.columns):
                        if i >= j:
                            continue
                        pair_key = (min(col_i, col_j), max(col_i, col_j))
                        if pair_key in seen:
                            continue
                        seen.add(pair_key)
                        if abs(corr_matrix.loc[col_i, col_j]) > 0.5:
                            high_corr_pairs.append({
                                "factors": [col_i, col_j],
                                "correlation": round(corr_matrix.loc[col_i, col_j], 3),
                            })
                corr_clusters = high_corr_pairs[:10]  # top 10
        except Exception:
            pass

        # LLM diagnosis
        diagnosis = {
            "per_factor": per_factor,
            "correlation_clusters": corr_clusters,
            "total_factors": len(per_factor),
            "dead_count": sum(1 for v in per_factor.values() if v["status"] == "dead"),
            "decaying_count": sum(1 for v in per_factor.values() if v["status"] == "decaying"),
        }

        return diagnosis

    def propose_factors(
        self,
        diagnosis: dict,
        existing_names: list[str],
        llm: "LLMClient",
        n_ideas: int = 5,
    ) -> list[dict]:
        """Ask LLM to propose factor formulas addressing diagnosed weaknesses.

        Returns list of {name, intuition, category, expression_hint}.
        """
        if not llm.configured:
            logger.info("LLM seeding skipped (no API key)")
            return []

        # Build concise prompt
        weak_factors = {
            name: info for name, info in diagnosis.get("per_factor", {}).items()
            if info["status"] in ("dead", "decaying", "weak")
        }
        healthy_factors = {
            name: info for name, info in diagnosis.get("per_factor", {}).items()
            if info["status"] == "healthy"
        }

        prompt = f"""You are a quantitative researcher designing stock selection factors for the Chinese A-share market.

## Current Factor Health
Dead/decaying factors (need replacement):
{json.dumps(weak_factors, indent=2, ensure_ascii=False)}

Healthy factors (preserve):
{json.dumps(healthy_factors, indent=2, ensure_ascii=False)}

Correlation clusters (redundant pairs):
{json.dumps(diagnosis.get('correlation_clusters', []), indent=2, ensure_ascii=False)}

## Available Data Fields
OHLCV: close, high, low, open, volume, amount, turnover, pre_close
Derived: ret_5d, ret_20d, ret_60d, vol_20d, vol_60d, hl_ratio, amihud, vol_ratio

## Available Operators
{operator_registry.to_llm_prompt()}

## Task
Propose exactly {n_ideas} new factor formulas that:
1. Address the specific weaknesses above (replace dead/decaying factors with uncorrelated alternatives)
2. Capture non-linear relationships, tail risk, or conditional patterns the existing factors miss
3. Use the expression_hint format below (machine-parseable shorthand)

Output a JSON array:
[{{"name": "snake_case_name", "intuition": "1-2 sentence economic rationale",
   "category": "momentum|value|quality|volatility|liquidity|leader|volume_price|risk|composite",
   "expression_hint": "op(field, param) or op(op(field,p), op(field2,p))"}}]

Expression hint syntax:
- Variable: field_name (e.g. close, volume, ret_20d)
- Unary: neg(x), abs(x), log(x), sqrt(x), sign(x)
- Binary: add(a,b), sub(a,b), mul(a,b), div(a,b), max(a,b), min(a,b), gt(a,b), lt(a,b)
- Rolling: ts_mean(x,20), ts_std(x,20), ts_skew(x,60), ts_kurt(x,30), ts_corr(x,y,20), ts_quantile(x,20,0.25), ts_ema(x,10), ts_prod(x,20), ts_sum(x,20), ts_min(x,20), ts_max(x,20), ts_delay(x,5), ts_rank(x,20)
- Cross-sectional: cs_rank(x), cs_zscore(x), cs_regression_residual(x)
- Conditional: if_then(cond, then_val, else_val)
- Time-series: pct_change(x,5), delta(x,5)
- Constants: 0.5, -1.0, etc.
- Nesting: ts_mean(div(close, volume), 20)

IMPORTANT: Return ONLY the JSON array, no markdown fences, no explanation."""

        result = llm.chat_json(prompt, expected_keys=None)

        if isinstance(result, list):
            proposals = result
        elif "error" in result:
            logger.warning("LLM seed proposal failed: %s", result.get("error"))
            return []
        else:
            proposals = result.get("ideas", result.get("factors", [result]))

        if not isinstance(proposals, list):
            proposals = [proposals]

        logger.info("LLM proposed %d factor ideas", len(proposals))
        return proposals[:n_ideas]

    def compile_seeds(
        self,
        proposals: list[dict],
    ) -> list["Individual"]:
        """Parse expression_hint strings → Expr trees → compile → Individuals.

        Skips unparseable proposals with a warning.
        """
        from discovery.compiler import compile_expr, compile_and_validate

        seeds = []
        for prop in proposals:
            hint = prop.get("expression_hint", "")
            try:
                tree = self._parse_hint(hint)
            except Exception as e:
                logger.warning("Failed to parse hint '%s': %s", hint, e)
                continue

            name = prop.get("name", f"llm_seed_{len(seeds)}")
            category = prop.get("category", "llm")
            try:
                factor_cls = compile_expr(tree, factor_name=name,
                                          category=category, register=True)
                from discovery.gp import Individual
                ind = Individual(
                    tree=tree, factor_name=name, factor_cls=factor_cls,
                    fitness=0.0, ic_mean=0.0, ic_ir=0.0, hit_rate=0.5,
                    auto_corr=0.5, complexity=tree.complexity(),
                    depth=tree.depth(), generation=-1,
                )
                seeds.append(ind)
                logger.info("Compiled LLM seed: %s (%s)", name, hint)
            except Exception as e:
                logger.warning("Failed to compile seed '%s': %s", name, e)

        return seeds

    # ── Expression hint parser ─────────────────────────────────────────────

    def _parse_hint(self, hint: str) -> Expr:
        """Parse a simple expression shorthand into an Expr tree.

        Examples:
          close → VarExpr("close")
          ts_mean(close, 20) → RollingOp("ts_mean", 20, VarExpr("close"))
          div(close, volume) → BinaryOp("div", VarExpr("close"), VarExpr("volume"))
          cs_rank(ts_std(ret_5d, 60)) → CrossSectionalOp("cs_rank", RollingOp(...))
          if_then(gt(close, ts_mean(close,20)), ret_5d, neg(ret_20d)) → TernaryOp(...)
        """
        hint = hint.strip()

        # Try if_then first (most complex pattern)
        if hint.startswith("if_then("):
            return self._parse_ternary(hint)

        # Try binary ops
        for op in BINARY_OPS:
            prefix = f"{op}("
            if hint.startswith(prefix):
                return self._parse_binary(op, hint)

        # Try rolling ops (single-child, then binary like ts_corr)
        for op in ROLLING_OPS + ROLLING_OPS_BINARY:
            prefix = f"{op}("
            if hint.startswith(prefix):
                return self._parse_rolling(op, hint)

        # Try unary ops
        for op in UNARY_OPS:
            prefix = f"{op}("
            if hint.startswith(prefix):
                return self._parse_unary(op, hint)

        # Try cross-sectional ops
        for op in CS_OPS:
            prefix = f"{op}("
            if hint.startswith(prefix):
                return self._parse_cs(op, hint)

        # Try time-series ops
        for op in TS_OPS:
            prefix = f"{op}("
            if hint.startswith(prefix):
                return self._parse_ts(op, hint)

        # Try constant
        try:
            val = float(hint)
            return ConstExpr(val)
        except ValueError:
            pass

        # Assume variable
        return VarExpr(hint)

    def _find_matching_paren(self, s: str, start: int) -> int:
        """Find the matching closing paren for an opening paren at `start`."""
        depth = 1
        i = start + 1
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        return i - 1  # position of matching ')'

    def _split_args(self, args_str: str) -> list[str]:
        """Split comma-separated args, respecting nested parens."""
        parts = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())
        return parts

    def _parse_ternary(self, hint: str) -> Expr:
        """Parse if_then(cond, then, else)."""
        inner = hint[len("if_then("):-1]
        parts = self._split_args(inner)
        if len(parts) != 3:
            raise ValueError(f"if_then needs 3 args, got {len(parts)}: {hint}")
        from discovery.expr import TernaryOp
        return TernaryOp(
            "if_then",
            self._parse_hint(parts[0]),
            self._parse_hint(parts[1]),
            self._parse_hint(parts[2]),
        )

    def _parse_binary(self, op: str, hint: str) -> Expr:
        """Parse op(left, right)."""
        inner = hint[len(f"{op}("):-1]
        parts = self._split_args(inner)
        if len(parts) != 2:
            raise ValueError(f"Binary op needs 2 args, got {len(parts)}: {hint}")
        return BinaryOp(op, self._parse_hint(parts[0]), self._parse_hint(parts[1]))

    def _parse_rolling(self, op: str, hint: str) -> Expr:
        """Parse rolling_op(child, window) or ts_quantile(child, w, q)."""
        inner = hint[len(f"{op}("):-1]
        parts = self._split_args(inner)

        if op == "ts_quantile" and len(parts) >= 3:
            return RollingOp(
                op, int(parts[1]), self._parse_hint(parts[0]),
                quantile=float(parts[2]),
            )
        elif op == "ts_corr" and len(parts) >= 3:
            return RollingOp(
                op, int(parts[2]), self._parse_hint(parts[0]),
                right=self._parse_hint(parts[1]),
            )
        elif len(parts) >= 2:
            return RollingOp(op, int(parts[1]), self._parse_hint(parts[0]))
        else:
            raise ValueError(f"Rolling op needs 2+ args, got {len(parts)}: {hint}")

    def _parse_unary(self, op: str, hint: str) -> Expr:
        """Parse unary_op(child)."""
        inner = hint[len(f"{op}("):-1]
        return UnaryOp(op, self._parse_hint(inner.strip()))

    def _parse_cs(self, op: str, hint: str) -> Expr:
        """Parse cs_op(child)."""
        inner = hint[len(f"{op}("):-1]
        return CrossSectionalOp(op, self._parse_hint(inner.strip()))

    def _parse_ts(self, op: str, hint: str) -> Expr:
        """Parse ts_op(child, periods)."""
        inner = hint[len(f"{op}("):-1]
        parts = self._split_args(inner)
        if len(parts) >= 2:
            return TimeSeriesOp(op, int(parts[1]), self._parse_hint(parts[0]))
        raise ValueError(f"TS op needs 2 args, got {len(parts)}: {hint}")
