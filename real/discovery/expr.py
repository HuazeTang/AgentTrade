"""Expression tree for factor formulas.

An Expr tree represents a factor formula that can be compiled to a Factor class.
The tree is evolvable: GP can mutate, crossover, and select on it.

All operations work on a MultiIndex DataFrame (trade_date, symbol).
Leaf nodes (VarExpr) reference columns in the data.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Expr(ABC):
    """Abstract base for expression tree nodes."""

    @abstractmethod
    def to_source(self) -> str:
        """Generate valid pandas expression string evaluable in factor compute()."""
        ...

    @abstractmethod
    def complexity(self) -> int:
        """Total node count in subtree (used for parsimony penalty)."""
        ...

    @abstractmethod
    def required_fields(self) -> set[str]:
        """All VarExpr leaf names this tree depends on."""
        ...

    @abstractmethod
    def depth(self) -> int:
        """Maximum depth from this node to deepest leaf."""
        ...

    @abstractmethod
    def node_count(self) -> int:
        """Number of nodes in this subtree."""
        ...

    @abstractmethod
    def clone(self) -> Expr:
        """Deep copy of the tree."""
        ...

    @abstractmethod
    def children(self) -> list[Expr]:
        """Return direct child nodes (for mutation/crossover)."""
        ...

    @abstractmethod
    def replace_child(self, old: Expr, new: Expr) -> Expr:
        """Return a new tree with `old` child replaced by `new`."""
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON persistence."""
        ...

    @staticmethod
    def from_dict(d: dict) -> Expr:
        """Deserialize from a dict produced by to_dict()."""
        t = d["type"]
        if t == "var":
            return VarExpr(d["name"])
        if t == "const":
            return ConstExpr(d["value"])
        if t == "unary":
            return UnaryOp(d["op"], Expr.from_dict(d["child"]))
        if t == "binary":
            return BinaryOp(d["op"], Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "rolling":
            return RollingOp(d["op"], d["window"], Expr.from_dict(d["child"]))
        if t == "cs":
            return CrossSectionalOp(d["op"], Expr.from_dict(d["child"]))
        if t == "cs_group":
            return GroupedCrossSectionalOp(d["op"], d["group_field"], Expr.from_dict(d["child"]))
        if t == "ts":
            return TimeSeriesOp(d["op"], d["periods"], Expr.from_dict(d["child"]))
        raise ValueError(f"Unknown expr type: {t}")


# ═══════════════════════════════════════════════════════════════════════════════
# Leaf nodes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VarExpr(Expr):
    """Reference a column in the data DataFrame (e.g., 'close', 'volume')."""
    name: str

    def to_source(self) -> str:
        return f'data["{self.name}"]'

    def complexity(self) -> int:
        return 1

    def required_fields(self) -> set[str]:
        return {self.name}

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1

    def clone(self) -> Expr:
        return VarExpr(self.name)

    def children(self) -> list[Expr]:
        return []

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        return self  # leaf has no children

    def __repr__(self) -> str:
        return f"Var({self.name})"

    def to_dict(self) -> dict:
        return {"type": "var", "name": self.name}


@dataclass
class ConstExpr(Expr):
    """Literal constant."""
    value: float

    def to_source(self) -> str:
        return repr(self.value)

    def complexity(self) -> int:
        return 1

    def required_fields(self) -> set[str]:
        return set()

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1

    def clone(self) -> Expr:
        return ConstExpr(self.value)

    def children(self) -> list[Expr]:
        return []

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        return self

    def __repr__(self) -> str:
        return f"Const({self.value:g})"

    def to_dict(self) -> dict:
        return {"type": "const", "value": self.value}


# ═══════════════════════════════════════════════════════════════════════════════
# Unary node
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnaryOp(Expr):
    """Unary operator: neg, abs, log, sqrt, sign, inverse."""
    op: str
    child: Expr

    _FUNCTIONS = {
        "neg": "(-{x})",
        "abs": "({x}).abs()",
        "log": "np.log(({x}).clip(lower=1e-10))",
        "sqrt": "np.sqrt(({x}).clip(lower=0))",
        "sign": "np.sign({x})",
        "inverse": "(-({x}))",
    }

    def to_source(self) -> str:
        inner = self.child.to_source()
        template = self._FUNCTIONS.get(self.op, "({x})")
        return template.format(x=inner)

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def required_fields(self) -> set[str]:
        return self.child.required_fields()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def clone(self) -> Expr:
        return UnaryOp(self.op, self.child.clone())

    def children(self) -> list[Expr]:
        return [self.child]

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        if self.child is old:
            return UnaryOp(self.op, new)
        return UnaryOp(self.op, self.child.replace_child(old, new))

    def __repr__(self) -> str:
        return f"{self.op}({self.child!r})"

    def to_dict(self) -> dict:
        return {"type": "unary", "op": self.op, "child": self.child.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════════
# Binary node
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BinaryOp(Expr):
    """Binary operator: add, sub, mul, div, max, min."""
    op: str
    left: Expr
    right: Expr

    _SYMBOLS = {
        "add": "+", "sub": "-", "mul": "*", "div": "/",
    }

    def to_source(self) -> str:
        l = self.left.to_source()
        r = self.right.to_source()
        if self.op in self._SYMBOLS:
            return f"({l} {self._SYMBOLS[self.op]} {r})"
        elif self.op == "max":
            return f"np.maximum({l}, {r})"
        elif self.op == "min":
            return f"np.minimum({l}, {r})"
        elif self.op == "gt":
            return f"(({l}) > ({r})).astype(float)"
        elif self.op == "lt":
            return f"(({l}) < ({r})).astype(float)"
        return f"({l} + {r})"  # fallback

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def required_fields(self) -> set[str]:
        return self.left.required_fields() | self.right.required_fields()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def clone(self) -> Expr:
        return BinaryOp(self.op, self.left.clone(), self.right.clone())

    def children(self) -> list[Expr]:
        return [self.left, self.right]

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        if self.left is old:
            return BinaryOp(self.op, new, self.right)
        if self.right is old:
            return BinaryOp(self.op, self.left, new)
        return BinaryOp(self.op, self.left.replace_child(old, new), self.right.replace_child(old, new))

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"

    def to_dict(self) -> dict:
        return {"type": "binary", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling window operations (operate along time axis within each symbol)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RollingOp(Expr):
    """Time-series rolling operations: ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_rank, ts_delay.

    These unstack → operate on time axis → stack.
    """
    op: str
    window: int
    child: Expr

    _IMPLS = {
        "ts_sum":   ".rolling({w}, min_periods=max(1,{w}//2)).sum()",
        "ts_mean":  ".rolling({w}, min_periods=max(1,{w}//2)).mean()",
        "ts_std":   ".rolling({w}, min_periods=max(1,{w}//2)).std()",
        "ts_min":   ".rolling({w}, min_periods=max(1,{w}//2)).min()",
        "ts_max":   ".rolling({w}, min_periods=max(1,{w}//2)).max()",
        "ts_delay": ".shift({w})",
        "ts_rank":  ".rolling({w}, min_periods=max(1,{w}//2)).apply(lambda x: (x.rank().iloc[-1]-1)/(len(x)-1) if len(x)>1 else 0.5, raw=False)",
    }

    def to_source(self) -> str:
        inner = self.child.to_source()
        impl = self._IMPLS.get(self.op, self._IMPLS["ts_mean"])
        unstacked = f"({inner}).unstack()"
        operated = unstacked + impl.format(w=self.window)
        return f"({operated}).stack()"

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def required_fields(self) -> set[str]:
        return self.child.required_fields()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def clone(self) -> Expr:
        return RollingOp(self.op, self.window, self.child.clone())

    def children(self) -> list[Expr]:
        return [self.child]

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        if self.child is old:
            return RollingOp(self.op, self.window, new)
        return RollingOp(self.op, self.window, self.child.replace_child(old, new))

    def __repr__(self) -> str:
        return f"{self.op}({self.window}, {self.child!r})"

    def to_dict(self) -> dict:
        return {"type": "rolling", "op": self.op, "window": self.window, "child": self.child.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-sectional operations (operate across symbols within each date)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossSectionalOp(Expr):
    """Cross-sectional operations: cs_rank, cs_zscore, cs_scale.

    These unstack → operate across columns (axis=1) → stack.
    """
    op: str
    child: Expr

    _IMPLS = {
        "cs_rank":   ".rank(axis=1, pct=True)",
        "cs_zscore": ".pipe(lambda df: df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).clip(lower=1e-10), axis=0))",
        "cs_scale":  ".pipe(lambda df: (df - df.min(axis=1)) / (df.max(axis=1) - df.min(axis=1)).clip(lower=1e-10))",
    }

    def to_source(self) -> str:
        inner = self.child.to_source()
        impl = self._IMPLS.get(self.op, self._IMPLS["cs_rank"])
        return f"(({inner}).unstack(){impl}).stack()"

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def required_fields(self) -> set[str]:
        return self.child.required_fields()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def clone(self) -> Expr:
        return CrossSectionalOp(self.op, self.child.clone())

    def children(self) -> list[Expr]:
        return [self.child]

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        if self.child is old:
            return CrossSectionalOp(self.op, new)
        return CrossSectionalOp(self.op, self.child.replace_child(old, new))

    def __repr__(self) -> str:
        return f"{self.op}({self.child!r})"

    def to_dict(self) -> dict:
        return {"type": "cs", "op": self.op, "child": self.child.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════════
# Grouped cross-sectional operations (grouped by a data column, e.g. sector)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GroupedCrossSectionalOp(Expr):
    """Cross-sectional operations grouped by a categorical field.

    cs_group_mean: within-group mean — separates stock-specific from group-level signal.
    cs_group_zscore: within-group z-score — leader vs peers within same sector.
    """
    op: str
    group_field: str  # e.g. "board" — column in data used for grouping
    child: Expr

    _IMPLS = {
        "cs_group_mean":   ".groupby(g).transform('mean')",
        "cs_group_zscore": ".groupby(g).transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))",
    }

    def to_source(self) -> str:
        inner = self.child.to_source()
        impl = self._IMPLS.get(self.op, self._IMPLS["cs_group_mean"])
        # Transpose so symbols become rows, groupby, then transpose back
        return (
            f"((lambda u, g: u.T{impl}.T)"
            f"(({inner}).unstack(),"
            f" data['{self.group_field}'].groupby(level='symbol').first()))"
            f".stack()"
        )

    def complexity(self) -> int:
        return 2 + self.child.complexity()

    def required_fields(self) -> set[str]:
        return self.child.required_fields() | {self.group_field}

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def clone(self) -> Expr:
        return GroupedCrossSectionalOp(self.op, self.group_field, self.child.clone())

    def children(self) -> list[Expr]:
        return [self.child]

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        if self.child is old:
            return GroupedCrossSectionalOp(self.op, self.group_field, new)
        return GroupedCrossSectionalOp(self.op, self.group_field,
                                       self.child.replace_child(old, new))

    def __repr__(self) -> str:
        return f"{self.op}({self.group_field}, {self.child!r})"

    def to_dict(self) -> dict:
        return {"type": "cs_group", "op": self.op, "group_field": self.group_field, "child": self.child.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════════
# Time-series pointwise operations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimeSeriesOp(Expr):
    """Pointwise time-series operations: pct_change, delta, ts_lag.

    pct_change and delta work on unstacked data (date x symbol).
    """
    op: str
    periods: int
    child: Expr

    _IMPLS = {
        "pct_change": ".pct_change(periods={p})",
        "delta":      ".diff(periods={p})",
        "ts_lag":     ".shift({p})",
    }

    def to_source(self) -> str:
        inner = self.child.to_source()
        impl = self._IMPLS.get(self.op, self._IMPLS["pct_change"])
        return f"(({inner}).unstack(){impl.format(p=self.periods)}).stack()"

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def required_fields(self) -> set[str]:
        return self.child.required_fields()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def clone(self) -> Expr:
        return TimeSeriesOp(self.op, self.periods, self.child.clone())

    def children(self) -> list[Expr]:
        return [self.child]

    def replace_child(self, old: Expr, new: Expr) -> Expr:
        if self.child is old:
            return TimeSeriesOp(self.op, self.periods, new)
        return TimeSeriesOp(self.op, self.periods, self.child.replace_child(old, new))

    def __repr__(self) -> str:
        return f"{self.op}({self.periods}, {self.child!r})"

    def to_dict(self) -> dict:
        return {"type": "ts", "op": self.op, "periods": self.periods, "child": self.child.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════════
# Tree generation helpers (used by GP initialisation and mutation)
# ═══════════════════════════════════════════════════════════════════════════════

# Terminal set: data fields available at leaves
TERMINAL_FIELDS = [
    "close", "volume", "amount", "turnover", "high", "low", "open", "pre_close",
    # Derived momentum/volatility terminals (pre-computed in data)
    "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "vol_60d",
    "hl_ratio", "amihud", "vol_ratio",
]

# Fields available for group-by in cross-sectional grouped ops
GROUP_FIELDS = ["board"]

# Operators available for each node type
UNARY_OPS = ["neg", "abs", "log", "sqrt", "sign"]
BINARY_OPS = ["add", "sub", "mul", "div", "max", "min", "gt", "lt"]
ROLLING_OPS = ["ts_sum", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_delay", "ts_rank"]
CS_OPS = ["cs_rank", "cs_zscore"]
CS_GROUP_OPS = ["cs_group_mean", "cs_group_zscore"]
TS_OPS = ["pct_change", "delta"]

# Typical windows for rolling ops
ROLLING_WINDOWS = [5, 10, 20, 30, 60, 120]
# Typical periods for TS ops
TS_PERIODS = [1, 5, 10, 21, 63]


def random_expr(
    max_depth: int = 4,
    terminals: list[str] | None = None,
    method: str = "grow",
) -> Expr:
    """Generate a random expression tree.

    Args:
        max_depth: maximum tree depth.
        terminals: list of field names for leaf nodes.
        method: 'grow' (random shapes) or 'full' (all branches to max_depth).
    """
    fields = terminals or TERMINAL_FIELDS
    return _random_tree(1, max_depth, fields, method)


def _random_tree(
    current_depth: int,
    max_depth: int,
    terminals: list[str],
    method: str,
) -> Expr:
    """Recursive random tree generator."""
    # At max depth, must use terminal
    if current_depth >= max_depth:
        return _random_terminal(terminals)

    # At depths 1..max_depth-1, probabilistically choose node type
    # 7 choices: rolling, ts, binary, unary, cs, cs_group, terminal
    if method == "full":
        node_type_weights = [0.26, 0.26, 0.18, 0.10, 0.08, 0.08, 0.04]
    else:  # grow
        if current_depth == 1:
            node_type_weights = [0.28, 0.24, 0.14, 0.10, 0.08, 0.08, 0.08]
        else:
            node_type_weights = [0.19, 0.15, 0.14, 0.10, 0.08, 0.08, 0.26]

    choice = random.choices(
        ["rolling", "ts", "binary", "unary", "cs", "cs_group", "terminal"],
        weights=node_type_weights,
        k=1,
    )[0]

    if choice == "terminal":
        return _random_terminal(terminals)
    elif choice == "rolling":
        op = random.choice(ROLLING_OPS)
        w = random.choice(ROLLING_WINDOWS)
        return RollingOp(op, w, _random_tree(current_depth + 1, max_depth, terminals, method))
    elif choice == "ts":
        op = random.choice(TS_OPS)
        p = random.choice(TS_PERIODS)
        return TimeSeriesOp(op, p, _random_tree(current_depth + 1, max_depth, terminals, method))
    elif choice == "binary":
        op = random.choice(BINARY_OPS)
        left = _random_tree(current_depth + 1, max_depth, terminals, method)
        right = _random_tree(current_depth + 1, max_depth, terminals, method)
        return BinaryOp(op, left, right)
    elif choice == "unary":
        op = random.choice(UNARY_OPS)
        return UnaryOp(op, _random_tree(current_depth + 1, max_depth, terminals, method))
    elif choice == "cs":
        op = random.choice(CS_OPS)
        return CrossSectionalOp(op, _random_tree(current_depth + 1, max_depth, terminals, method))
    elif choice == "cs_group":
        op = random.choice(CS_GROUP_OPS)
        gf = random.choice(GROUP_FIELDS)
        return GroupedCrossSectionalOp(op, gf, _random_tree(current_depth + 1, max_depth, terminals, method))

    return _random_terminal(terminals)


def _random_terminal(terminals: list[str]) -> Expr:
    """Randomly choose a VarExpr or ConstExpr, with bias toward price/momentum terminals."""
    if random.random() < 0.7:
        # Bias toward price-like terminals (70% price, 30% volume/other)
        price_terminals = {"close", "high", "low", "open", "ret_5d", "ret_20d", "ret_60d"}
        price_in_set = [t for t in terminals if t in price_terminals]
        other_in_set = [t for t in terminals if t not in price_terminals]
        if price_in_set and random.random() < 0.7:
            return VarExpr(random.choice(price_in_set))
        elif other_in_set:
            return VarExpr(random.choice(other_in_set))
        else:
            return VarExpr(random.choice(terminals))
    else:
        return ConstExpr(round(random.uniform(-1, 1), 2))


def collect_all_nodes(root: Expr) -> list[Expr]:
    """Collect all nodes in the tree via BFS."""
    nodes = [root]
    for child in root.children():
        nodes.extend(collect_all_nodes(child))
    return nodes
