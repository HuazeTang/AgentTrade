"""LLM-driven 妖股 factor generator.

Pipeline: case extraction → contrastive prompt → LLM → parse JSON expression trees
→ compile Factor classes → validate via IC/IR.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from agent.llm_client import LLMClient
from discovery.expr import Expr
from discovery.compiler import compile_expr
from discovery.operators import operator_registry
from llm.yaogu.case_extractor import YaoguCaseExtractor, CasePair
from llm.yaogu.prompts import SYSTEM_PROMPT, build_contrastive_prompt

logger = logging.getLogger(__name__)


@dataclass
class GeneratedFactor:
    """A successfully compiled factor from LLM output."""
    name: str
    category: str
    expression_json: dict
    tree: Expr
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    validation_passed: bool = False
    compile_error: str = ""


class YaoguFactorGenerator:
    """Generate 妖股-detection factors via LLM contrastive analysis."""

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        n_factors_per_round: int = 5,
    ):
        self._client = client or LLMClient(model=model, temperature=temperature)
        self.n_factors_per_round = n_factors_per_round

    @property
    def configured(self) -> bool:
        return self._client.configured

    # ── Main API ───────────────────────────────────────────────────────────

    def generate(
        self,
        pairs: list[CasePair],
        existing_factors: list[str],
    ) -> list[GeneratedFactor]:
        """Generate factors from contrastive case pairs.

        Args:
            pairs: Matched 妖股/假启动 pairs from YaoguCaseExtractor.
            existing_factors: Names of already-registered factors to avoid.

        Returns:
            List of successfully compiled GeneratedFactor objects.
        """
        if not self.configured:
            logger.error("LLM client not configured. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
            return []

        if not pairs:
            logger.warning("No case pairs provided — cannot generate factors.")
            return []

        # Build summary + select top illustrative pairs
        summary = YaoguCaseExtractor.compute_summary(pairs)
        top_pairs = YaoguCaseExtractor.top_pairs(pairs, n=5)
        user_prompt = build_contrastive_prompt(
            summary, top_pairs, existing_factors, self.n_factors_per_round
        )

        logger.info("Calling LLM with %d pairs, %d existing factors...",
                     summary["n_pairs"], len(existing_factors))

        try:
            response = self._client.chat(
                user_prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return []

        # Parse JSON expressions from response
        expressions = _extract_json_array(response.text)
        if not expressions:
            logger.error("Failed to parse any expression trees from LLM response.")
            logger.debug("Raw response: %s", response.text[:500])
            return []

        logger.info("LLM returned %d expression trees", len(expressions))

        # Compile each expression
        results = []
        for i, expr_dict in enumerate(expressions):
            gf = self._compile_one(expr_dict, i, existing_factors)
            results.append(gf)
            if gf.compile_error:
                logger.warning("  [%d] FAILED: %s", i, gf.compile_error)
            else:
                logger.info("  [%d] %s (category=%s, nodes=%d, depth=%d)",
                            i, gf.name, gf.category,
                            gf.tree.node_count(), gf.tree.depth())

        succeeded = [r for r in results if not r.compile_error]
        logger.info("Compiled %d/%d factors successfully", len(succeeded), len(results))
        return succeeded

    def _compile_one(
        self, expr_dict: dict, idx: int, existing: list[str],
    ) -> GeneratedFactor:
        """Compile a single expression tree dict into a Factor class."""
        # Validate basic structure
        if "type" not in expr_dict:
            return GeneratedFactor(
                name="", category="", expression_json=expr_dict,
                tree=None, compile_error="Missing 'type' field"
            )

        # Generate a unique name from the expression
        name = _generate_factor_name(expr_dict, idx)

        try:
            tree = Expr.from_dict(expr_dict)
        except Exception as e:
            return GeneratedFactor(
                name=name, category="", expression_json=expr_dict,
                tree=None, compile_error=f"Expr.from_dict failed: {e}"
            )

        # Basic sanity checks on tree
        if tree.node_count() > 30:
            return GeneratedFactor(
                name=name, category="", expression_json=expr_dict,
                tree=tree, compile_error=f"Tree too complex ({tree.node_count()} nodes, max 30)"
            )
        if tree.depth() > 6:
            return GeneratedFactor(
                name=name, category="", expression_json=expr_dict,
                tree=tree, compile_error=f"Tree too deep (depth {tree.depth()}, max 6)"
            )

        # Infer category from tree structure
        category = _infer_category(tree)

        try:
            compiled = compile_expr(
                tree,
                factor_name=name,
                category=category,
                register=True,
            )
        except Exception as e:
            return GeneratedFactor(
                name=name, category=category, expression_json=expr_dict,
                tree=tree, compile_error=f"compile_expr failed: {e}"
            )

        return GeneratedFactor(
            name=name,
            category=category,
            expression_json=expr_dict,
            tree=tree,
        )


# ── JSON parsing ───────────────────────────────────────────────────────

def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM response, handling markdown fences."""
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "type" in result:
            return [result]
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    # Find ```json ... ``` blocks
    import re
    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
    for block in json_blocks:
        try:
            result = json.loads(block.strip())
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "type" in result:
                return [result]
        except json.JSONDecodeError:
            continue

    # Try finding JSON array pattern
    array_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
    if array_match:
        try:
            result = json.loads(array_match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try individual JSON objects
    objects = re.findall(r'\{[^{}]*"type"\s*:\s*"[^"]+"[^{}]*\}', text)
    results = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if "type" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return results


# ── Factor naming ───────────────────────────────────────────────────────

def _generate_factor_name(expr_dict: dict, idx: int) -> str:
    """Generate a stable, descriptive factor name from expression dict."""
    # Extract operator sequence for naming
    ops = _extract_op_sequence(expr_dict)
    ops_short = "_".join(ops[:4])  # first 4 ops
    expr_str = json.dumps(expr_dict, sort_keys=True)
    h = hashlib.sha256(expr_str.encode()).hexdigest()[:8]
    return f"yaogu_{ops_short}_{h}"


def _extract_op_sequence(node: dict) -> list[str]:
    """Extract operator sequence from expression tree dict."""
    ops = []
    t = node.get("type", "")
    if t == "var":
        ops.append(node.get("name", "v")[:8])
    elif t == "const":
        ops.append(f"c{node.get('value', 0)}")
    elif t == "unary":
        ops.append(node.get("op", "u")[:4])
        if "child" in node:
            ops.extend(_extract_op_sequence(node["child"]))
    elif t == "binary":
        ops.append(node.get("op", "b")[:4])
        if "left" in node:
            ops.extend(_extract_op_sequence(node["left"]))
        if "right" in node:
            ops.extend(_extract_op_sequence(node["right"]))
    elif t == "rolling":
        op = node.get("op", "r")
        w = node.get("window", "")
        ops.append(f"{op}{w}")
        if "child" in node:
            ops.extend(_extract_op_sequence(node["child"]))
    elif t == "cs":
        ops.append(node.get("op", "cs")[:8])
        if "child" in node:
            ops.extend(_extract_op_sequence(node["child"]))
    elif t == "cs_group":
        ops.append(f"{node.get('op', 'cg')[:8]}")
        if "child" in node:
            ops.extend(_extract_op_sequence(node["child"]))
    elif t == "ts":
        op = node.get("op", "ts")
        p = node.get("periods", "")
        ops.append(f"{op}{p}")
        if "child" in node:
            ops.extend(_extract_op_sequence(node["child"]))
    return ops


def _infer_category(tree: Expr) -> str:
    """Infer factor category from expression tree structure."""
    # Look at leaf variables and operators to guess category
    fields = tree.required_fields()

    # Check if primarily volume-based
    vol_fields = {"volume", "amount", "turnover"}
    if fields & vol_fields and not fields - vol_fields - {"close"}:
        return "volume_price"

    # Check if uses high/low (technical pattern)
    if {"high", "low"} & fields:
        return "pattern"

    # Check for trend/leader signals
    if "ret_5d" in fields or "ret_20d" in fields:
        return "leader"

    return "composite"
