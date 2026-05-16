"""Factor registry -- maps factor names to classes with decorator support."""

from __future__ import annotations

from factor.base import Factor


class FactorRegistry:
    """Central registry for factor lookup and instantiation."""

    _instance: FactorRegistry | None = None
    _factors: dict[str, type[Factor]] = {}

    def __new__(cls) -> FactorRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}
        return cls._instance

    def register(self, factor_cls: type[Factor]) -> type[Factor]:
        name = factor_cls.meta.name
        if name in self._factors:
            raise ValueError(f"Factor '{name}' already registered")
        self._factors[name] = factor_cls
        return factor_cls

    def get(self, name: str) -> type[Factor]:
        try:
            return self._factors[name]
        except KeyError:
            raise KeyError(
                f"Factor '{name}' not found. Available: {list(self._factors.keys())}"
            )

    def list_all(self) -> list[str]:
        return sorted(self._factors.keys())

    def list_by_category(self, category: str) -> list[str]:
        return sorted(
            k for k, v in self._factors.items() if v.meta.category == category
        )

    def categories(self) -> list[str]:
        cats = {v.meta.category for v in self._factors.values()}
        return sorted(cats)


# Singleton instance for use across the codebase
registry = FactorRegistry()


def register_factor(cls: type[Factor]) -> type[Factor]:
    """Decorator to register a factor class."""
    return registry.register(cls)
