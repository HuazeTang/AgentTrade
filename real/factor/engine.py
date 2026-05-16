"""Factor computation engine with dependency resolution via topological sort."""

from __future__ import annotations

import logging
from collections import defaultdict, deque

import pandas as pd

from factor.base import Factor
from factor.registry import registry

logger = logging.getLogger(__name__)


class FactorEngine:
    """Computes factors in dependency order, caching intermediate results."""

    def __init__(self):
        self._computed: dict[str, pd.Series] = {}

    def compute(
        self,
        factor_names: list[str],
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute a set of factors.

        Args:
            factor_names: List of factor names to compute.
            data: Multi-indexed DataFrame (trade_date, symbol) with OHLCV etc.

        Returns:
            DataFrame with factor values, columns = factor names,
            multi-index (trade_date, symbol).
        """
        # Resolve dependencies
        ordered = self._resolve_order(factor_names)

        results: dict[str, pd.Series] = {}
        for name in ordered:
            cls = registry.get(name)
            factor = cls()
            # Check if this factor depends on other factors
            deps = factor.dependencies
            dep_data = data.copy()
            for dep in deps:
                if dep in results:
                    dep_data = dep_data.copy()
                    dep_data[dep] = results[dep]
                else:
                    logger.warning(
                        "Factor '%s' depends on '%s' which was not requested", name, dep
                    )

            logger.debug("Computing factor: %s", name)
            ser = factor.compute(dep_data)
            ser.name = name
            results[name] = ser

        self._computed.update(results)
        return pd.DataFrame(results)

    def _resolve_order(self, names: list[str]) -> list[str]:
        """Topological sort of factors by dependencies."""
        # Build full dependency graph (include transitive deps)
        all_names = set(names)
        queue = deque(names)
        while queue:
            name = queue.popleft()
            cls = registry.get(name)
            for dep in cls().dependencies:
                if dep not in all_names:
                    all_names.add(dep)
                    queue.append(dep)

        # Build adjacency and in-degree
        graph: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)
        for name in all_names:
            in_degree.setdefault(name, 0)
            factor = registry.get(name)()
            deps = factor.dependencies
            for dep in deps:
                if dep in all_names:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # Kahn's algorithm
        q = deque(n for n in all_names if in_degree[n] == 0)
        order: list[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in graph[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    q.append(m)

        if len(order) != len(all_names):
            raise RuntimeError(
                f"Circular dependency detected among factors: {all_names - set(order)}"
            )

        # Return only requested factors in computed order
        return [n for n in order if n in set(names)]

    def clear(self) -> None:
        self._computed.clear()
