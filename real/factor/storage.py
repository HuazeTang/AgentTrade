"""Factor storage: read/write factor values to parquet with versioning."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from config.settings import FACTOR_DIR


def write_factor(
    series: pd.Series,
    factor_name: str,
    version: str = "1.0.0",
) -> Path:
    """Write factor values to parquet.

    Args:
        series: Multi-indexed (trade_date, symbol) or (symbol,) Series.
        factor_name: Name of the factor.
        version: Factor version string.

    Returns:
        Path to the written file.
    """
    out_dir = FACTOR_DIR / factor_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"v{version}.parquet"
    df = series.to_frame(name=factor_name)
    df.to_parquet(path)
    return path


def read_factor(
    factor_name: str,
    start: date | None = None,
    end: date | None = None,
    version: str = "1.0.0",
) -> pd.DataFrame:
    """Read factor values from parquet.

    Returns DataFrame with factor_name as column and (trade_date, symbol)
    as multi-index.
    """
    path = FACTOR_DIR / factor_name / f"v{version}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")

    df = pd.read_parquet(path)

    if start is not None or end is not None:
        if "trade_date" in df.index.names:
            idx_val = df.index.get_level_values("trade_date")
        elif "trade_date" in df.columns:
            idx_val = pd.to_datetime(df["trade_date"])
        else:
            return df
        if start is not None:
            df = df[idx_val >= pd.Timestamp(start)]
        if end is not None:
            df = df[idx_val <= pd.Timestamp(end)]

    return df


def list_factors() -> list[str]:
    """List all stored factor names."""
    if not FACTOR_DIR.exists():
        return []
    return sorted(
        d.name for d in FACTOR_DIR.iterdir() if d.is_dir() and (d / "v1.0.0.parquet").exists()
    )
