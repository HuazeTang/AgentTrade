"""
Shared matplotlib configuration for Chinese font rendering and consistent chart style.

Import this module once at startup and it configures matplotlib globally.
Works on macOS, Linux, and Windows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import fontManager

logger = logging.getLogger("chart_style")

# ── CJK Font Resolution ────────────────────────────────────────────────────────

_CJK_FONT_NAME: str | None = None

_CJK_CANDIDATES: list[tuple[str, list[str]]] = [
    # (font_name_matplotlib, [possible_file_paths])
    ("Heiti TC", [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/opt/X11/share/system_fonts/STHeiti Medium.ttc",
    ]),
    ("PingFang SC", [
        "/System/Library/Fonts/PingFang.ttc",
        "/opt/X11/share/system_fonts/PingFang.ttc",
    ]),
    ("Hiragino Sans GB", [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/opt/X11/share/system_fonts/Hiragino Sans GB.ttc",
    ]),
    ("STHeiti", [
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/*/AssetData/STHEITI.ttf",
    ]),
    ("SimHei", []),            # Windows
    ("Microsoft YaHei", []),   # Windows
    ("Noto Sans CJK SC", []),  # Linux
    ("WenQuanYi Micro Hei", []),  # Linux
]


def _register_font_file(path: Path) -> None:
    try:
        fontManager.addfont(str(path))
    except Exception:
        pass


def _resolve_cjk_font() -> str | None:
    """Find and register a CJK font. Returns the matplotlib font name, or None."""

    # Phase 1: try known candidates from hardcoded paths
    for name, paths in _CJK_CANDIDATES:
        for p in paths:
            if "*" in p:
                for pf in sorted(Path(p).parent.glob(Path(p).name)):
                    if pf.exists():
                        _register_font_file(pf)
            else:
                pf = Path(p)
                if pf.exists():
                    _register_font_file(pf)
        # Check if font became available by name
        if any(name in f.name for f in fontManager.ttflist):
            return name

    # Phase 2: brute-force scan font directories and register everything
    _font_dirs = [
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/opt/X11/share/system_fonts",
        "/opt/X11/share/system_fonts/Supplemental",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    for d in _font_dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for ext in ("*.ttc", "*.ttf", "*.otf"):
            for pf in sorted(dp.glob(ext)):
                _register_font_file(pf)

    # Phase 3: search for any CJK font by name keyword
    _cjk_keywords = ("Heiti", "PingFang", "Songti", "Hiragino", "SimHei",
                     "YaHei", "CJK", "WenQuanYi", "Noto Sans CJK")
    for f in fontManager.ttflist:
        if any(k in f.name for k in _cjk_keywords):
            return f.name

    return None


def _setup_matplotlib_style(cjk_font: str | None) -> None:
    """Apply global matplotlib rcParams for consistent chart style."""

    # Rebuild font cache so newly registered fonts are discovered
    try:
        fontManager.__dict__.pop("_lookup_cache", None)
    except Exception:
        pass

    sans_serif_fonts = [cjk_font] if cjk_font else []
    sans_serif_fonts += ["DejaVu Sans", "Arial", "Helvetica"]

    plt.rcParams.update({
        # ── Fonts ──
        "font.family": "sans-serif",
        "font.sans-serif": sans_serif_fonts,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.unicode_minus": False,

        # ── Figure ──
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",

        # ── Axes ──
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#dee2e6",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # ── Save ──
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ── Run at import time ──────────────────────────────────────────────────────────

_CJK_FONT_NAME = _resolve_cjk_font()

if _CJK_FONT_NAME:
    logger.info("CJK font resolved: %s", _CJK_FONT_NAME)
else:
    logger.warning("No CJK font found — Chinese characters may render as tofu (□)")

_setup_matplotlib_style(_CJK_FONT_NAME)


# ── Public API ──────────────────────────────────────────────────────────────────

def get_cjk_font_name() -> str | None:
    """Return the resolved CJK font name, or None if none found."""
    return _CJK_FONT_NAME


def get_mpl_rc() -> dict:
    """Return rc dict suitable for mplfinance make_mpf_style or matplotlib rcParams."""
    rc: dict = {}
    if _CJK_FONT_NAME:
        rc["font.sans-serif"] = [_CJK_FONT_NAME, "DejaVu Sans", "Arial", "Helvetica"]
        rc["font.family"] = "sans-serif"
    return rc


# Consistent color palette
COLORS = {
    "old": "#4a90d9",       # blue
    "new": "#e74c3c",       # red
    "benchmark": "#95a5a6",  # grey
    "buy": "#cc0000",        # red (A-share up convention)
    "sell": "#00aa00",       # green (A-share down convention)
    "up": "#cc0000",
    "down": "#00aa00",
    "equity": "#2c3e50",
    "positive": "#27ae60",
    "negative": "#e74c3c",
}
