"""Domain-specific exceptions."""


class RealTradeError(Exception):
    """Base exception for RealTrade."""


class DataError(RealTradeError):
    """Data-related errors (missing, corrupt, schema mismatch)."""


class OrderError(RealTradeError):
    """Order validation or execution errors."""


class UniverseError(RealTradeError):
    """Stock universe filtering errors."""


class BacktestError(RealTradeError):
    """Backtest engine errors."""


class FactorError(RealTradeError):
    """Factor computation errors."""
