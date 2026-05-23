"""Visualization for parameter search results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})


def _extract(results: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract Sharpe and max_dd arrays from results."""
    sharpe = np.array([r["metrics"].get("sharpe_ratio", -999) for r in results])
    dd = np.array([r["metrics"].get("max_drawdown", 0) for r in results])
    return sharpe, -dd, results  # -dd so larger = better


def _pareto_frontier(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return indices of Pareto frontier (maximize both x and y)."""
    idx = np.lexsort((ys, xs))
    frontier = []
    best_y = -np.inf
    for i in idx:
        if ys[i] > best_y:
            frontier.append(i)
            best_y = ys[i]
    return np.array(frontier)


def plot_pareto(results: list[dict], output_path: Path) -> None:
    """Pareto frontier: Sharpe vs Max Drawdown."""
    sharpe, dd_abs, _ = _extract(results)
    frontier_idx = _pareto_frontier(dd_abs, sharpe)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dominated points
    dominated = np.ones(len(results), dtype=bool)
    dominated[frontier_idx] = False
    ax.scatter(dd_abs[dominated], sharpe[dominated], c="#94a3b8", s=60,
               edgecolors="white", linewidth=0.5, alpha=0.6, zorder=3,
               label="Dominated")

    # Frontier points
    ax.scatter(dd_abs[frontier_idx], sharpe[frontier_idx], c="#10b981", s=100,
               edgecolors="white", linewidth=1, zorder=5, label="Pareto Frontier")

    # Frontier line
    fx, fy = dd_abs[frontier_idx], sharpe[frontier_idx]
    order = np.argsort(fx)
    ax.plot(fx[order], fy[order], color="#10b981", linewidth=2, alpha=0.5, zorder=3)

    # Baseline marker
    bl_dd = 20.84  # approximate baseline
    bl_sh = 1.558
    ax.scatter([bl_dd], [bl_sh], c="#f59e0b", s=180, edgecolors="white",
               linewidth=2, zorder=10, marker="D", label="Baseline")

    ax.set_xlabel("|Max Drawdown| % (lower → better)", fontsize=11)
    ax.set_ylabel("Sharpe Ratio (higher → better)", fontsize=11)
    ax.set_title(f"Pareto Frontier: {len(results)} Configurations", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sensitivity(results: list[dict], output_path: Path) -> None:
    """Sensitivity chart: one bar per parameter showing Sharpe range."""
    # Group by parameter name → collect Sharpe values for each distinct value
    from collections import defaultdict

    all_param_names = sorted(results[0]["params"].keys())
    n = len(all_param_names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    fig.suptitle("Parameter Sensitivity", fontsize=14, fontweight="bold", y=1.01)
    axes = np.atleast_1d(axes).flatten()

    for ax, pname in zip(axes, all_param_names):
        by_val = defaultdict(list)
        for r in results:
            v = r["params"][pname]
            s = r["metrics"].get("sharpe_ratio", -999)
            by_val[v].append(s)

        x = sorted(by_val.keys())
        y_mean = [np.mean(by_val[v]) for v in x]
        y_min = [np.min(by_val[v]) for v in x]
        y_max = [np.max(by_val[v]) for v in x]

        # Color: bars where all values identical = gray (dead param)
        colors = []
        for i in range(len(x)):
            if y_max[i] - y_min[i] < 0.001:
                colors.append("#94a3b8")
            elif y_mean[i] == max(y_mean):
                colors.append("#10b981")
            elif y_mean[i] < max(y_mean) - 0.05:
                colors.append("#ef4444")
            else:
                colors.append("#3b82f6")

        ax.bar(range(len(x)), y_mean, color=colors, edgecolor="white", linewidth=0.5)
        for i, m in enumerate(y_mean):
            ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=7,
                    fontweight="bold")

        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([str(v) for v in x], fontsize=8, rotation=30)
        ax.set_title(pname, fontsize=10)
        ax.set_ylabel("Sharpe", fontsize=9)
        ax.set_ylim(min(y_min) - 0.1, max(y_max) + 0.1)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for ax in axes[len(all_param_names):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_parallel(results: list[dict], output_path: Path) -> None:
    """Parallel coordinates plot: top N configs across all parameters."""
    top_n = min(20, len(results))
    from pandas import DataFrame

    # Build DataFrame
    rows = []
    for r in results[:top_n]:
        row = dict(r["params"])
        row["sharpe"] = r["metrics"].get("sharpe_ratio", -999)
        row["return"] = r["metrics"].get("cumulative_return", 0) * 100
        row["max_dd"] = r["metrics"].get("max_drawdown", 0) * 100
        rows.append(row)
    df = DataFrame(rows)

    param_cols = [c for c in df.columns if c not in ("sharpe", "return", "max_dd")]

    # Normalize each param to [0, 1]
    norm = df[param_cols].copy()
    for col in param_cols:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Color by Sharpe
    cmap = plt.cm.viridis
    sharpe_vals = df["sharpe"].values
    smin, smax = sharpe_vals.min(), sharpe_vals.max()
    norm_s = (sharpe_vals - smin) / (smax - smin + 1e-10)

    x = list(range(len(param_cols)))
    for i in range(len(norm)):
        ax.plot(x, norm.iloc[i].values, color=cmap(norm_s[i]), alpha=0.5,
                linewidth=1)

    # Highlight best
    best_idx = sharpe_vals.argmax()
    ax.plot(x, norm.iloc[best_idx].values, color="red", linewidth=2.5,
            label=f"Best (Sharpe={sharpe_vals[best_idx]:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(param_cols, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Normalized value", fontsize=10)
    ax.set_title(f"Top {top_n} Configurations — Parallel Coordinates", fontsize=13,
                 fontweight="bold")
    ax.legend()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(sharpe_vals)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")
