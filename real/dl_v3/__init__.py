"""V3 training module: pre-launch yaogu detection.

Re-exports architecture components from dl (same DualTowerModel, SmoothAP,
etc.) — only dataset and feature computation are V3-specific.
"""

# Re-export from dl (shared architecture)
from dl import (
    DualTowerModel,
    FocalLoss,
    SmoothAPLoss,
    compute_average_precision,
    count_parameters,
)

# V3-specific
from dl_v3.dataset import YaoguDatasetV3, build_dataloaders_v3
from dl_v3.derived_features import (
    ALL_V3_COLUMNS,
    build_v3_feature_cache,
)
