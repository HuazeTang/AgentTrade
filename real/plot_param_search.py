#!/usr/bin/env python3
"""Pareto frontier chart — Sharpe vs Max DD across all parameter combos."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})

INPUT = Path(__file__).resolve().parent / "data" / "results" / "param_search.json"
OUTPUT = Path(__file__).resolve().parent / "data" / "results" / "param_search.png"


def pareto_frontier(xs, ys, maximize_x=True, maximize_y=True):
    """Return indices of points on the Pareto frontier."""
    points = np.column_stack([xs, ys])
    idx_sorted = np.lexsort((ys, xs))  # sort by x then y
    frontier = []
    best_y = -np.inf if maximize_y else np.inf
    for i in idx_sorted:
        y = ys[i]
        if maximize_y:
            if y > best_y:
                frontier.append(i)
                best_y = y
        else:
            if y < best_y:
                frontier.append(i)
                best_y = y
    return frontier


# ── Load ──
with open(INPUT) as f:
    data = json.load(f)

# ── Deduplicate (many params give identical metrics) ──
points: dict[tuple, list[dict]] = defaultdict(list)
for r in data:
    dd = float(str(r["max_drawdown"]).rstrip("%")) if isinstance(r["max_drawdown"], str) else float(r["max_drawdown"])
    sharpe = float(r["sharpe_ratio"])
    ret = float(str(r["cumulative_return"]).rstrip("%")) if isinstance(r["cumulative_return"], str) else float(r["cumulative_return"])
    key = (round(sharpe, 4), round(dd, 4))
    points[key].append(r)

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax1, ax2 = axes

# Pick representative for each unique outcome
unique = [(k[0], k[1], v[0]) for k, v in points.items()]
sharpe_vals = np.array([u[0] for u in unique])
dd_vals = np.array([u[1] for u in unique])

# Compute Pareto frontier (maximize Sharpe, minimize |DD| → maximize DD)
frontier_idx = pareto_frontier(-dd_vals, sharpe_vals, maximize_x=True, maximize_y=True)

# Color: frontier = green, dominated = gray
all_colors = []
for i in range(len(unique)):
    if i in frontier_idx:
        all_colors.append("#10b981")
    else:
        all_colors.append("#94a3b8")

# ── Left: full view ──
ax1.scatter(-dd_vals, sharpe_vals, c=all_colors, s=120, edgecolors="white",
            linewidth=1, zorder=5, alpha=0.9)
# Draw frontier line
fx = -dd_vals[frontier_idx]
fy = sharpe_vals[frontier_idx]
order = np.argsort(fx)
ax1.plot(fx[order], fy[order], color="#10b981", linewidth=2, alpha=0.6, zorder=3)

# Annotate
for i in range(len(unique)):
    r = unique[i][2]
    p = r["params"]
    label = f"TP={p['TAKE_PROFIT_PCT']} SV={p['STOP_LOSS_VOL_MULT']}"
    if p["TRAIL_STOP_FROM_PEAK"] != 0.15 or p["MIN_STOP_LOSS_PCT"] != 0.05 or p["CONSECUTIVE_DOWN_EXIT"] != 3:
        # Only label if deviates from baseline
        pass
    ax1.annotate(label, (-dd_vals[i], sharpe_vals[i]),
                 textcoords="offset points", xytext=(6, 6), fontsize=7,
                 color="#475569")

# Highlight baseline
baseline_dd = 20.84
baseline_sharpe = 1.558
ax1.scatter([baseline_dd], [baseline_sharpe], c="#f59e0b", s=180, edgecolors="white",
            linewidth=2, zorder=10, marker="D")
ax1.annotate("BASELINE\n(0.25, 2.0)", (baseline_dd, baseline_sharpe),
             textcoords="offset points", xytext=(10, -15), fontsize=8,
             color="#92400e", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2))

ax1.set_xlabel("|Max Drawdown| % (lower is better →)", fontsize=11)
ax1.set_ylabel("Sharpe Ratio (higher is better →)", fontsize=11)
ax1.set_title("Pareto Frontier: Sharpe vs Max Drawdown", fontsize=12, fontweight="bold")
ax1.grid(alpha=0.3)
ax1.set_xlim(min(-dd_vals) - 1, max(-dd_vals) + 1)
ax1.set_ylim(min(sharpe_vals) - 0.05, max(sharpe_vals) + 0.05)

# ── Right: zoom on the 3 unique outcomes ──
ax2.scatter(-dd_vals, sharpe_vals, c=all_colors, s=200, edgecolors="white",
            linewidth=1.5, zorder=5, alpha=0.9)
fx = -dd_vals[frontier_idx]
fy = sharpe_vals[frontier_idx]
order = np.argsort(fx)
ax2.plot(fx[order], fy[order], color="#10b981", linewidth=2.5, alpha=0.5, zorder=3)
ax2.scatter([baseline_dd], [baseline_sharpe], c="#f59e0b", s=250, edgecolors="white",
            linewidth=2, zorder=10, marker="D")

# Label each cluster with its param combo
from collections import Counter
# Group by param key differences
for i in range(len(unique)):
    r = unique[i][2]
    p = r["params"]
    # Which params differ from baseline?
    diffs = []
    if p["TAKE_PROFIT_PCT"] != 0.25:
        diffs.append(f"TP={p['TAKE_PROFIT_PCT']}")
    if p["STOP_LOSS_VOL_MULT"] != 2.0:
        diffs.append(f"SV={p['STOP_LOSS_VOL_MULT']}")
    if p["TRAIL_STOP_FROM_PEAK"] != 0.15:
        diffs.append(f"TS={p['TRAIL_STOP_FROM_PEAK']}")
    if p["MIN_STOP_LOSS_PCT"] != 0.05:
        diffs.append(f"MS={p['MIN_STOP_LOSS_PCT']}")
    if p["CONSECUTIVE_DOWN_EXIT"] != 3:
        diffs.append(f"CD={p['CONSECUTIVE_DOWN_EXIT']}")

    label = ", ".join(diffs) if diffs else "BASELINE"
    color = "#b91c1c" if sharpe_vals[i] < 1.5 else "#334155"
    ax2.annotate(label, (-dd_vals[i], sharpe_vals[i]),
                 textcoords="offset points", xytext=(10, 8), fontsize=8,
                 color=color, fontweight="bold")

ax2.set_xlabel("|Max Drawdown| % (lower is better →)", fontsize=11)
ax2.set_ylabel("Sharpe Ratio (higher is better →)", fontsize=11)
ax2.set_title("Zoom: 3 Unique Outcomes from 17 Configs", fontsize=12, fontweight="bold")
ax2.grid(alpha=0.3)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor="#10b981", label="Pareto Frontier"),
    Patch(facecolor="#94a3b8", label="Dominated"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#f59e0b", markersize=10, label="Baseline"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved to {OUTPUT}")
plt.close()
