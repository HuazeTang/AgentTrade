"""Operator definitions and registry for factor expression trees.

Each operator has metadata (arity, category, financial meaning) separate from
its pandas implementation in expr.py. The registry is the single source of truth
for what operators GP can use and what the LLM sees in prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class OperatorMeta:
    """Metadata for a single operator in the expression language.

    Attributes:
        name: Unique operator identifier (e.g. 'ts_mean', 'cs_rank').
        arity: Number of children (1 = unary, 2 = binary).
        category: 'arithmetic', 'rolling', 'cross_sectional', 'time_series', 'unary'.
        description: One-line financial interpretation.
        detail: Longer explanation of what it computes and when it's useful.
        params: List of parameter names (e.g. ['window'] for rolling ops).
        param_defaults: Suggested ranges or values for parameters.
        node_type: Which Expr subclass to instantiate ('unary', 'binary', 'rolling', 'cs', 'ts').
        pandas_template: Reference implementation pattern (for LLM prompts).
    """
    name: str
    arity: int
    category: str
    description: str
    detail: str = ""
    params: list[str] = field(default_factory=list)
    param_defaults: dict[str, list] = field(default_factory=dict)
    node_type: str = "unary"
    pandas_template: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Operator definitions
# ═══════════════════════════════════════════════════════════════════════════════

UNARY_OPERATORS: list[OperatorMeta] = [
    OperatorMeta(
        name="neg",
        arity=1,
        category="arithmetic",
        description="Negation: flip the sign of a factor (e.g. turn momentum into reversal)",
        detail="Useful for inverting a factor's direction. If a factor has negative IC, "
               "negating it produces a positive-IC factor.",
        node_type="unary",
        pandas_template="-({x})",
    ),
    OperatorMeta(
        name="abs",
        arity=1,
        category="arithmetic",
        description="Absolute value: measure magnitude regardless of direction",
        detail="Captures extremeness. High absolute values indicate strong signals in "
               "either direction. Often used in volatility proxies.",
        node_type="unary",
        pandas_template="({x}).abs()",
    ),
    OperatorMeta(
        name="log",
        arity=1,
        category="arithmetic",
        description="Natural logarithm: compress extreme values, handle multiplicative processes",
        detail="Reduces skew from outliers. Returns are additive in log space. "
               "Common for volume, amount, and other positive right-skewed fields.",
        node_type="unary",
        pandas_template="np.log(({x}).clip(lower=1e-10))",
    ),
    OperatorMeta(
        name="sqrt",
        arity=1,
        category="arithmetic",
        description="Square root: mild compression of positive values",
        detail="Milder than log. Keeps zero at zero while compressing large values. "
               "Used when log would be too aggressive.",
        node_type="unary",
        pandas_template="np.sqrt(({x}).clip(lower=0))",
    ),
    OperatorMeta(
        name="sign",
        arity=1,
        category="arithmetic",
        description="Sign: extract direction only (-1, 0, +1)",
        detail="Reduces a factor to pure direction. Useful when magnitude varies too much "
               "across regimes but direction is stable.",
        node_type="unary",
        pandas_template="np.sign({x})",
    ),
]

BINARY_OPERATORS: list[OperatorMeta] = [
    OperatorMeta(
        name="add",
        arity=2,
        category="arithmetic",
        description="Addition: combine two factors with equal weight",
        detail="Simple factor blending. Equivalent to 1/N portfolio of two signals. "
               "Works best when both factors have similar scale after normalization.",
        node_type="binary",
        pandas_template="({x} + {y})",
    ),
    OperatorMeta(
        name="sub",
        arity=2,
        category="arithmetic",
        description="Subtraction: difference between two factors (e.g. short-term minus long-term momentum)",
        detail="Captures relative strength. Classic 12m-1m momentum is a subtraction. "
               "Also used for factor spreads and hedged signals.",
        node_type="binary",
        pandas_template="({x} - {y})",
    ),
    OperatorMeta(
        name="mul",
        arity=2,
        category="arithmetic",
        description="Multiplication: interaction between two factors",
        detail="Amplifies signal when both factors agree. If factor A says buy and "
               "factor B says buy, the product is strongly positive. Interaction terms "
               "capture non-linear relationships.",
        node_type="binary",
        pandas_template="({x} * {y})",
    ),
    OperatorMeta(
        name="div",
        arity=2,
        category="arithmetic",
        description="Division: ratio of two factors (e.g. price-to-volume as liquidity proxy)",
        detail="Creates normalized ratios. Value factors like EP, BP are divisions. "
               "Be careful about division by zero; the compiler clips denominators.",
        node_type="binary",
        pandas_template="({x} / {y}.clip(lower=1e-10))",
    ),
    OperatorMeta(
        name="max",
        arity=2,
        category="arithmetic",
        description="Maximum: take the stronger signal of two",
        detail="Optimistic combination. Use when either factor being positive is enough "
               "reason to go long. Related to 'long the best of both' strategies.",
        node_type="binary",
        pandas_template="np.maximum({x}, {y})",
    ),
    OperatorMeta(
        name="min",
        arity=2,
        category="arithmetic",
        description="Minimum: take the weaker (more conservative) signal of two",
        detail="Conservative combination. Use when both factors must agree. "
               "Reduces false positives at the cost of missing some opportunities.",
        node_type="binary",
        pandas_template="np.minimum({x}, {y})",
    ),
    OperatorMeta(
        name="gt",
        arity=2,
        category="arithmetic",
        description="Greater-than comparison: returns 1.0 where left > right, else 0.0",
        detail="Boolean operator for pattern detection. e.g. close > pre_close detects up days. "
               "Combined with rolling mean, gives up-day ratio — a core trend strength signal.",
        node_type="binary",
        pandas_template="(({x}) > ({y})).astype(float)",
    ),
    OperatorMeta(
        name="lt",
        arity=2,
        category="arithmetic",
        description="Less-than comparison: returns 1.0 where left < right, else 0.0",
        detail="Boolean operator for downside pattern detection. e.g. close < pre_close detects "
               "down days. Paired with gt, enables complex conditional logic in factor trees.",
        node_type="binary",
        pandas_template="(({x}) < ({y})).astype(float)",
    ),
]

ROLLING_OPERATORS: list[OperatorMeta] = [
    OperatorMeta(
        name="ts_sum",
        arity=1,
        category="rolling",
        description="Rolling sum over N days: cumulative effect over a window",
        detail="Captures total flow. For volume: total turnover over N days identifies "
               "accumulation/distribution. For returns: cumulative return over window.",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60, 120]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).sum().stack()",
    ),
    OperatorMeta(
        name="ts_mean",
        arity=1,
        category="rolling",
        description="Rolling mean over N days: smoothed central tendency",
        detail="Moving average. For price: trend indicator. For volume: average turnover "
               "baseline. Short windows capture recent shift; long windows capture regime.",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60, 120]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).mean().stack()",
    ),
    OperatorMeta(
        name="ts_std",
        arity=1,
        category="rolling",
        description="Rolling standard deviation over N days: time-series volatility",
        detail="Realized volatility. High std = increased uncertainty, wider stops. "
               "Low std = consolidation, potential breakout. Core building block for "
               "risk-adjusted factors (Sharpe-like signals).",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60, 120]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).std().stack()",
    ),
    OperatorMeta(
        name="ts_min",
        arity=1,
        category="rolling",
        description="Rolling minimum over N days: support level, worst case",
        detail="N-day low. Classic technical indicator. Break below signals weakness; "
               "holding above signals strength. Used in channel-based strategies.",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60, 120]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).min().stack()",
    ),
    OperatorMeta(
        name="ts_max",
        arity=1,
        category="rolling",
        description="Rolling maximum over N days: resistance level, best case",
        detail="N-day high. Break above signals momentum continuation; failure to break "
               "signals resistance. Used with ts_min for Donchian-like channels.",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60, 120]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).max().stack()",
    ),
    OperatorMeta(
        name="ts_delay",
        arity=1,
        category="rolling",
        description="Time delay (shift) by N days: lagged value for delta computations",
        detail="Shifts a series backward. Used to compute changes (current - delayed) "
               "or to avoid look-ahead when combining factors with different update frequencies.",
        params=["window"],
        param_defaults={"window": [1, 5, 10, 21, 63]},
        node_type="rolling",
        pandas_template="({x}).unstack().shift({w}).stack()",
    ),
    OperatorMeta(
        name="ts_rank",
        arity=1,
        category="rolling",
        description="Rolling rank (percentile) over N days: where is today vs recent history?",
        detail="Cross-time normalization. Rank = 1 means today is the highest value in "
               "the window. Robust to outliers and non-stationarity. Common in time-series "
               "momentum (Moskowitz et al. 2012).",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60, 120]},
        node_type="rolling",
        pandas_template=(
            "({x}).unstack().rolling({w}, min_periods=max(1,{w}//2))"
            ".apply(lambda x: (x.rank().iloc[-1]-1)/(len(x)-1) if len(x)>1 else 0.5, raw=False).stack()"
        ),
    ),
    OperatorMeta(
        name="ts_skew",
        arity=1,
        category="rolling",
        description="Rolling skewness: third moment — asymmetry of returns distribution",
        detail="Positive skew = fat right tail (crash-proof? moonshot potential). "
               "Negative skew = fat left tail (crash risk). High absolute skew signals "
               "regime change or pending reversal. Typical window: 20-60 days.",
        params=["window"],
        param_defaults={"window": [10, 20, 30, 60]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).skew().stack()",
    ),
    OperatorMeta(
        name="ts_kurt",
        arity=1,
        category="rolling",
        description="Rolling kurtosis: fourth moment — tail fatness vs normal distribution",
        detail="High kurtosis = fat tails, extreme events more likely. Low kurtosis = "
               "thin tails, returns clustered near mean. Spikes often precede volatility "
               "regime changes. Common in crash prediction models.",
        params=["window"],
        param_defaults={"window": [20, 30, 60, 120]},
        node_type="rolling",
        pandas_template="({x}).unstack().rolling({w}, min_periods=max(1,{w}//2)).kurt().stack()",
    ),
    OperatorMeta(
        name="ts_corr",
        arity=2,
        category="rolling",
        description="Rolling correlation: co-movement between two series over N days",
        detail="Measures linear relationship between two factors. Corr(price, volume) "
               "detects volume-price divergence. Corr(stock_ret, mkt_ret) is rolling beta. "
               "High positive corr = same direction; negative = divergence signal.",
        params=["window"],
        param_defaults={"window": [10, 20, 30, 60]},
        node_type="rolling",
        pandas_template=(
            "({x}).unstack().rolling({w}, min_periods=max(1,{w}//2))"
            ".corr(({y}).unstack()).stack()"
        ),
    ),
    OperatorMeta(
        name="ts_quantile",
        arity=1,
        category="rolling",
        description="Rolling quantile: value at a given percentile over N days",
        detail="Captures distribution extremes. q=0.1 gives the 'worst 10%' floor; "
               "q=0.9 gives the 'best 10%' ceiling. More robust than min/max since it "
               "ignores single outliers. Used in VaR-like risk signals.",
        params=["window", "quantile"],
        param_defaults={"window": [20, 30, 60], "quantile": [0.1, 0.25, 0.75, 0.9]},
        node_type="rolling",
        pandas_template=(
            "({x}).unstack().rolling({w}, min_periods=max(1,{w}//2))"
            ".quantile({q}).stack()"
        ),
    ),
    OperatorMeta(
        name="ts_ema",
        arity=1,
        category="rolling",
        description="Exponential moving average: decay-weighted mean with span S",
        detail="Gives more weight to recent observations. Faster to react than SMA. "
               "Span=5 tracks fast trends; span=60 captures slow regime. "
               "Standard in modern momentum (bid-ask bounce resistant).",
        params=["span"],
        param_defaults={"span": [5, 10, 20, 30, 60]},
        node_type="rolling",
        pandas_template="({x}).unstack().ewm(span={s}, adjust=False).mean().stack()",
    ),
    OperatorMeta(
        name="ts_prod",
        arity=1,
        category="rolling",
        description="Rolling product: cumulative (1+ret) product minus 1 over N days",
        detail="Equivalent to cumulative return. prod(close_pct_change, 20) = 20-day "
               "return. Different from sum because it compounds. Correct for return "
               "accumulation; sum approximates only for small returns.",
        params=["window"],
        param_defaults={"window": [5, 10, 20, 30, 60]},
        node_type="rolling",
        pandas_template=(
            "({x}).unstack().pipe(lambda x: (1+x).rolling({w}, min_periods=max(1,{w}//2))"
            ".apply(np.prod, raw=True)-1).stack()"
        ),
    ),
]

CROSS_SECTIONAL_OPERATORS: list[OperatorMeta] = [
    OperatorMeta(
        name="cs_rank",
        arity=1,
        category="cross_sectional",
        description="Cross-sectional rank (percentile): where is this stock vs peers today?",
        detail="Core normalization for long-short portfolios. Rank = 1 means highest in "
               "cross-section. Removes market-wide effects, isolates stock-specific signal. "
               "Standard preprocessing before combining multiple factors.",
        node_type="cs",
        pandas_template="({x}).unstack().rank(axis=1, pct=True).stack()",
    ),
    OperatorMeta(
        name="cs_zscore",
        arity=1,
        category="cross_sectional",
        description="Cross-sectional z-score: how many std devs from cross-sectional mean?",
        detail="Standard normalization. Assumes roughly normal cross-sectional distribution. "
               "More sensitive to outliers than rank but preserves magnitude information. "
               "Values > 2 or < -2 indicate extreme positions.",
        node_type="cs",
        pandas_template=(
            "({x}).unstack()"
            ".pipe(lambda df: df.sub(df.mean(axis=1), axis=0)"
            ".div(df.std(axis=1).clip(lower=1e-10), axis=0))"
            ".stack()"
        ),
    ),
    OperatorMeta(
        name="cs_scale",
        arity=1,
        category="cross_sectional",
        description="Cross-sectional min-max scaling to [0, 1]",
        detail="Robust to outliers compared to z-score. Maps to fixed range. "
               "Useful when combining factors that should have equal weight regardless "
               "of their raw scale. Loses relative distance information at extremes.",
        node_type="cs",
        pandas_template=(
            "({x}).unstack()"
            ".pipe(lambda df: (df - df.min(axis=1)) / "
            "(df.max(axis=1) - df.min(axis=1)).clip(lower=1e-10))"
            ".stack()"
        ),
    ),
    OperatorMeta(
        name="cs_group_mean",
        arity=1,
        category="cross_sectional",
        description="Within-group mean: average value for stocks in the same sector/board",
        detail="Isolates sector-level signal. Subtract from raw factor to get stock-specific "
               "alpha. Key building block for relative strength (leader vs followers).",
        params=["group_field"],
        param_defaults={"group_field": ["board"]},
        node_type="cs_group",
        pandas_template=(
            "({x}).unstack().T.groupby(g).transform('mean').T.stack()"
        ),
    ),
    OperatorMeta(
        name="cs_group_zscore",
        arity=1,
        category="cross_sectional",
        description="Within-group z-score: how many std devs from sector mean",
        detail="Identifies leaders within each sector. A stock 2σ above its sector is a "
               "clear leader, regardless of whether the whole sector is up or down.",
        params=["group_field"],
        param_defaults={"group_field": ["board"]},
        node_type="cs_group",
        pandas_template=(
            "({x}).unstack().T.groupby(g)"
            ".transform(lambda x: (x - x.mean()) / (x.std() + 1e-10)).T.stack()"
        ),
    ),
    OperatorMeta(
        name="cs_regression_residual",
        arity=1,
        category="cross_sectional",
        description="Cross-sectional residual: deviation from cross-sectional mean",
        detail="Removes market-level signal, isolating stock-specific alpha. "
               "Equivalent to x - cs_mean(x). Use to de-mean before ranking. "
               "Creates long-short neutral factors.",
        node_type="cs",
        pandas_template=(
            "({x}).unstack()"
            ".pipe(lambda df: df.sub(df.mean(axis=1), axis=0))"
            ".stack()"
        ),
    ),
]

TIME_SERIES_OPERATORS: list[OperatorMeta] = [
    OperatorMeta(
        name="pct_change",
        arity=1,
        category="time_series",
        description="Percentage change over N periods: return-like transformation",
        detail="Turn any series into a return-like signal. pct_change(close, 21) = 1-month "
               "momentum. pct_change(volume, 5) = volume surge indicator. Stationary by "
               "construction, good for factors that should be mean-reverting.",
        params=["periods"],
        param_defaults={"periods": [1, 5, 10, 21, 63]},
        node_type="ts",
        pandas_template="({x}).unstack().pct_change(periods={p}).stack()",
    ),
    OperatorMeta(
        name="delta",
        arity=1,
        category="time_series",
        description="Simple difference over N periods: absolute change",
        detail="Unlike pct_change, preserves magnitude. Useful when the level matters "
               "(e.g., change in dollar volume, change in turnover rate). Less sensitive "
               "to base effects than percentage change.",
        params=["periods"],
        param_defaults={"periods": [1, 5, 10, 21, 63]},
        node_type="ts",
        pandas_template="({x}).unstack().diff(periods={p}).stack()",
    ),
    OperatorMeta(
        name="ts_lag",
        arity=1,
        category="time_series",
        description="Lag by N periods: use past value for alignment",
        detail="Same as ts_delay but defined as a time-series pointwise op. "
               "Used to align signals temporally or build autoregressive features.",
        params=["periods"],
        param_defaults={"periods": [1, 5, 10, 21, 63]},
        node_type="ts",
        pandas_template="({x}).unstack().shift({p}).stack()",
    ),
]

TERNARY_OPERATORS: list[OperatorMeta] = [
    OperatorMeta(
        name="if_then",
        arity=3,
        category="conditional",
        description="Conditional: if cond>0 use then_branch, else use else_branch",
        detail="Enables non-linear switching logic. if_then(momentum>0, momentum, reversal) "
               "applies momentum in up-trending names and reversal in down-trending. "
               "Combined with gt/lt, creates powerful rule-based factors.",
        node_type="ternary",
        pandas_template="np.where(({c})>0, {t}, {e})",
    ),
]

ALL_OPERATORS: list[OperatorMeta] = (
    UNARY_OPERATORS + BINARY_OPERATORS + ROLLING_OPERATORS
    + CROSS_SECTIONAL_OPERATORS + TIME_SERIES_OPERATORS + TERNARY_OPERATORS
)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

class OperatorRegistry:
    """Lookup and query available operators for GP, compiler, and LLM prompting."""

    def __init__(self, operators: list[OperatorMeta] | None = None):
        self._operators: dict[str, OperatorMeta] = {}
        self._by_category: dict[str, list[OperatorMeta]] = {}
        self._by_arity: dict[int, list[OperatorMeta]] = {}
        for op in (operators or ALL_OPERATORS):
            self.register(op)

    def register(self, op: OperatorMeta) -> None:
        if op.name in self._operators:
            raise ValueError(f"Operator '{op.name}' already registered")
        self._operators[op.name] = op
        self._by_category.setdefault(op.category, []).append(op)
        self._by_arity.setdefault(op.arity, []).append(op)

    def get(self, name: str) -> OperatorMeta:
        return self._operators[name]

    def list_all(self) -> list[str]:
        return sorted(self._operators.keys())

    def list_by_category(self, category: str) -> list[str]:
        return sorted(op.name for op in self._by_category.get(category, []))

    def list_by_arity(self, arity: int) -> list[str]:
        return sorted(op.name for op in self._by_arity.get(arity, []))

    def categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    def names_by_category(self) -> dict[str, list[str]]:
        return {cat: self.list_by_category(cat) for cat in self.categories()}

    def random_operator(self, category: str | None = None, arity: int | None = None) -> OperatorMeta:
        """Pick a random operator, optionally filtered by category or arity."""
        import random
        if category and arity:
            candidates = [op for op in self._by_category.get(category, []) if op.arity == arity]
        elif category:
            candidates = self._by_category.get(category, [])
        elif arity is not None:
            candidates = self._by_arity.get(arity, [])
        else:
            candidates = list(self._operators.values())
        return random.choice(candidates) if candidates else random.choice(list(self._operators.values()))

    def to_llm_prompt(self) -> str:
        """Generate a structured description of all operators for LLM prompts."""
        lines = ["Available factor operators:\n"]
        for cat in self.categories():
            lines.append(f"## {cat}")
            for op in self._by_category[cat]:
                params = ", ".join(f"{p}: {self._describe_defaults(op, p)}" for p in op.params)
                lines.append(f"  - `{op.name}`{f'({params})' if params else ''}: {op.description}")
                if op.detail:
                    lines.append(f"    {op.detail}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _describe_defaults(op: OperatorMeta, param: str) -> str:
        vals = op.param_defaults.get(param, [])
        if not vals:
            return "int"
        return f"int, e.g. {vals}"


# Singleton
operator_registry = OperatorRegistry(ALL_OPERATORS)
