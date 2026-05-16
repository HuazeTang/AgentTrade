"""Model serialization utilities (joblib-based)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.settings import MODEL_DIR


def save_model(
    model,
    model_name: str,
    metadata: dict | None = None,
) -> Path:
    """Save a trained model and its metadata.

    Args:
        model: Trained AlphaModel instance.
        model_name: Unique name for the model.
        metadata: Optional dict of metadata (features, params, etc.).

    Returns:
        Path to the saved model directory.
    """
    out_dir = MODEL_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(out_dir / "model.pkl"))

    meta = {
        "model_name": model_name,
        "saved_at": datetime.now().isoformat(),
        **(metadata or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    return out_dir


def load_model(model_name: str):
    """Load a trained model by name.

    Args:
        model_name: Name used during save_model.

    Returns:
        (model_instance, metadata_dict)
    """
    import joblib

    path = MODEL_DIR / model_name / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    state = joblib.load(str(path))
    meta_path = MODEL_DIR / model_name / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    return state, metadata


def list_models() -> list[str]:
    """List all saved model names."""
    if not MODEL_DIR.exists():
        return []
    return sorted(
        d.name for d in MODEL_DIR.iterdir()
        if d.is_dir() and (d / "model.pkl").exists()
    )
