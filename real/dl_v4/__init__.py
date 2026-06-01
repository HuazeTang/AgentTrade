"""V4: Raw OHLCV features + per-stock z-score normalization.

Let the Transformer learn patterns directly from price/volume data
rather than hand-crafted consolidation/divergence features.
"""

from dl import (
    DualTowerModel,
    FocalLoss,
    SmoothAPLoss,
    compute_average_precision,
    count_parameters,
)
