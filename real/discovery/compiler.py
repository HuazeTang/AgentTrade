"""Expression tree to Factor class compiler.

Compiles an Expr tree into a registered Factor class whose compute()
evaluates the generated pandas expression in a controlled namespace.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from discovery.expr import (
    Expr, VarExpr, ConstExpr, UnaryOp, BinaryOp,
    RollingOp, CrossSectionalOp, GroupedCrossSectionalOp, TimeSeriesOp,
    collect_all_nodes,
)
from discovery.operators import operator_registry, OperatorMeta
from factor.base import Factor, FactorMeta
from factor.registry import registry

logger = logging.getLogger(__name__)


# Safe namespace for expression evaluation.
# Only controlled objects are exposed - no __builtins__, no os, no sys.
_EVAL_NAMESPACE = {"np": np, "pd": pd}


def compile_expr(
    tree: Expr,
    factor_name: str | None = None,
    category: str | None = None,
    description: str | None = None,
    register: bool = True,
) -> type[Factor]:
    """Compile an expression tree into a Factor class.

    Args:
        tree: The expression tree to compile.
        factor_name: Optional name (auto-generated if None).
        category: Optional factor category (auto-detected if None).
        description: Optional human-readable description.
        register: Whether to register the factor in the global registry.

    Returns:
        A Factor subclass ready for computation.
    """
    name = factor_name or _generate_name(tree)
    cat = category or _infer_category(tree)
    desc = description or _generate_description(tree)
    source = tree.to_source()

    # Build compute() source code.
    # The expression uses 'data' as the MultiIndex DataFrame variable.
    compute_src = (
        f"def compute(self, data):\n"
        f"    import numpy as np\n"
        f"    result = {source}\n"
        f"    return result.rename('{name}').sort_index()\n"
    )

    # Compile and exec in controlled namespace.
    local_ns: dict = {}
    try:
        exec(compile(compute_src, "<factor_expr>", "exec"), _EVAL_NAMESPACE, local_ns)
    except SyntaxError as e:
        raise ValueError(f"Expression compiled to invalid Python: {source}") from e

    compute_func = local_ns["compute"]

    # Build FactorMeta
    meta = FactorMeta(
        name=name,
        category=cat,
        description=desc,
        version="1.0.0-gp",
        lookback_days=_estimate_lookback(tree),
    )

    # Collect required fields from leaf nodes
    req_fields = sorted(tree.required_fields())

    # Create Factor subclass dynamically
    factor_cls = type(name, (Factor,), {
        "meta": meta,
        "compute": compute_func,
        "required_fields": property(lambda self: req_fields),
        "__module__": __name__,
    })

    if register:
        try:
            registry.register(factor_cls)
        except ValueError:
            logger.debug("Factor '%s' already registered, skipping", name)

    return factor_cls


def compile_and_validate(
    tree: Expr,
    data: pd.DataFrame,
    factor_name: str | None = None,
) -> tuple[type[Factor], pd.Series]:
    """Compile an expression tree and immediately compute on test data.

    Returns (FactorClass, computed_series). Raises on compilation,
    computation failure, or near-zero variance output.
    """
    factor_cls = compile_expr(tree, factor_name=factor_name, register=False)
    factor = factor_cls()
    result = factor.compute(data)

    # Reject near-constant outputs (zero variance → NaN IC)
    std = result.std()
    if np.isnan(std) or std < 1e-10:
        raise ValueError(f"Near-zero variance output (std={std:.2g})")

    return factor_cls, result


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_name(tree: Expr) -> str:
    """Generate a stable, descriptive factor name from the tree structure."""
    structure = _tree_structure_string(tree)
    # Short hash for uniqueness
    h = hashlib.sha256(structure.encode()).hexdigest()[:8]
    # Build a readable prefix from the tree
    prefix = _name_prefix(tree)
    return f"gp_{prefix}_{h}"


def _name_prefix(tree: Expr) -> str:
    """Extract a short descriptive prefix from the tree root."""
    if isinstance(tree, VarExpr):
        return tree.name
    elif isinstance(tree, ConstExpr):
        return f"c{tree.value:g}".replace("-", "n").replace(".", "p")
    elif isinstance(tree, UnaryOp):
        return f"{tree.op}_{_name_prefix(tree.child)}"
    elif isinstance(tree, BinaryOp):
        return f"{tree.op}_{_name_prefix(tree.left)}_{_name_prefix(tree.right)}"
    elif isinstance(tree, RollingOp):
        return f"{tree.op}{tree.window}_{_name_prefix(tree.child)}"
    elif isinstance(tree, CrossSectionalOp):
        return f"{tree.op}_{_name_prefix(tree.child)}"
    elif isinstance(tree, GroupedCrossSectionalOp):
        return f"{tree.op}_{tree.group_field}_{_name_prefix(tree.child)}"
    elif isinstance(tree, TimeSeriesOp):
        return f"{tree.op}{tree.periods}_{_name_prefix(tree.child)}"
    return "expr"


def _tree_structure_string(tree: Expr) -> str:
    """Canonical string representation of tree structure for hashing."""
    if isinstance(tree, VarExpr):
        return f"V:{tree.name}"
    elif isinstance(tree, ConstExpr):
        return f"C:{tree.value:.4f}"
    elif isinstance(tree, UnaryOp):
        return f"U:{tree.op}({_tree_structure_string(tree.child)})"
    elif isinstance(tree, BinaryOp):
        return f"B:{tree.op}({_tree_structure_string(tree.left)},{_tree_structure_string(tree.right)})"
    elif isinstance(tree, RollingOp):
        return f"R:{tree.op}:{tree.window}({_tree_structure_string(tree.child)})"
    elif isinstance(tree, CrossSectionalOp):
        return f"X:{tree.op}({_tree_structure_string(tree.child)})"
    elif isinstance(tree, GroupedCrossSectionalOp):
        return f"G:{tree.op}:{tree.group_field}({_tree_structure_string(tree.child)})"
    elif isinstance(tree, TimeSeriesOp):
        return f"T:{tree.op}:{tree.periods}({_tree_structure_string(tree.child)})"
    return "?"


def _infer_category(tree: Expr) -> str:
    """Guess factor category from dominant operator types."""
    cats = _collect_op_categories(tree)

    # Priority: if we have rolling + pct_change/delta on close → momentum
    has_rolling = any(c == "rolling" for c in cats)
    has_ts = any(c == "time_series" for c in cats)
    has_cs = any(c == "cross_sectional" for c in cats)
    has_cs_group = any(isinstance(n, GroupedCrossSectionalOp) for n in collect_all_nodes(tree))
    has_arith = any(c == "arithmetic" for c in cats)

    # Check if leaves are predominantly price-based
    fields = tree.required_fields()
    price_like = {"close", "open", "high", "low"}
    volume_like = {"volume", "amount", "turnover"}

    if has_cs_group:
        return "leader"
    if has_ts and fields & price_like:
        return "momentum"
    if has_ts and fields & volume_like:
        return "liquidity"
    if has_cs:
        return "value"
    # Check for comparison ops → trend/leader category
    has_comparison = any(
        isinstance(n, BinaryOp) and n.op in ("gt", "lt")
        for n in collect_all_nodes(tree)
    )
    if has_comparison and has_rolling:
        return "trend"
    if has_rolling and fields & {"close", "high", "low"}:
        return "volatility"
    if has_rolling and fields & volume_like:
        return "liquidity"
    return "composite"


def _collect_op_categories(tree: Expr) -> list[str]:
    """Walk the tree and collect operator categories."""
    cats: list[str] = []
    if isinstance(tree, (UnaryOp, BinaryOp)):
        op = operator_registry.get(tree.op) if tree.op in operator_registry._operators else None
        if op:
            cats.append(op.category)
    elif isinstance(tree, RollingOp):
        cats.append("rolling")
    elif isinstance(tree, CrossSectionalOp):
        cats.append("cross_sectional")
    elif isinstance(tree, GroupedCrossSectionalOp):
        cats.append("cross_sectional")
    elif isinstance(tree, TimeSeriesOp):
        cats.append("time_series")

    for child in tree.children():
        cats.extend(_collect_op_categories(child))
    return cats


def _generate_description(tree: Expr) -> str:
    """Generate a human-readable description of the expression."""
    return f"GP-generated factor: {tree!r}"


def _estimate_lookback(tree: Expr) -> int:
    """Estimate maximum lookback window in days."""
    max_window = 0
    if isinstance(tree, RollingOp):
        max_window = max(max_window, tree.window)
    elif isinstance(tree, TimeSeriesOp):
        max_window = max(max_window, tree.periods)
    for child in tree.children():
        max_window = max(max_window, _estimate_lookback(child))
    return max_window
