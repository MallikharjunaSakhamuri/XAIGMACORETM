"""Shared plotting helpers for the figure stages.

One house style across Figures 2, 3, 4, 6, 7 and 8, so the paper's panels are
visually consistent and every stage writes PNG and PDF at the same dpi.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLOURS = {"ManualAug-CL": "#3b6ea5", "GMACORE-TM": "#c0504d"}
POSITIVE = "#c0504d"
NEGATIVE = "#3b6ea5"
NEUTRAL = "#7f7f7f"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def save(figure, output_dir: str, stem: str, dpi: int = 300) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"    wrote {path}")
    plt.close(figure)


def beeswarm(ax, values_by_row: Dict[str, Sequence[float]], colour_by_row=None,
             jitter: float = 0.18, seed: int = 0, point_size: float = 11.0):
    """A SHAP-style beeswarm: one row per feature, one point per molecule.

    Points are jittered vertically by local density so that overlapping values
    stay visible, which is the property that makes a beeswarm more informative
    than a box plot for attribution.
    """
    rng = np.random.default_rng(seed)
    rows = list(values_by_row)
    for index, row in enumerate(rows):
        values = np.asarray(values_by_row[row], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        counts, edges = np.histogram(values, bins=min(24, max(4, values.size // 3)))
        slot = np.clip(np.digitize(values, edges[1:-1]), 0, len(counts) - 1)
        density = counts[slot] / max(counts.max(), 1)
        offset = rng.uniform(-1, 1, size=values.size) * jitter * density
        if colour_by_row is not None and row in colour_by_row:
            shade = np.asarray(colour_by_row[row], dtype=float)[: values.size]
            ax.scatter(values, index + offset, s=point_size, c=shade,
                       cmap="coolwarm", alpha=0.85, linewidths=0)
        else:
            colours = np.where(values >= 0, POSITIVE, NEGATIVE)
            ax.scatter(values, index + offset, s=point_size, c=colours,
                       alpha=0.75, linewidths=0)
    ax.axvline(0.0, color="0.4", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    return ax


def horizontal_bars(ax, labels: Sequence[str], values: Sequence[float],
                    colour=None, xlabel: str = ""):
    values = np.asarray(values, dtype=float)
    positions = np.arange(len(labels))
    if colour is None:
        colour = np.where(values >= 0, POSITIVE, NEGATIVE)
    ax.barh(positions, values, color=colour, height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.4", linewidth=0.9)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def waterfall(ax, labels: Sequence[str], contributions: Sequence[float],
              baseline: float, final: float):
    """How signed contributions accumulate from the baseline to the prediction."""
    contributions = np.asarray(contributions, dtype=float)
    running = baseline
    widest = float(np.max(np.abs(contributions))) if contributions.size else 1.0
    for index, (label, value) in enumerate(zip(labels, contributions)):
        colour = POSITIVE if value >= 0 else NEGATIVE
        ax.barh(index, value, left=running, color=colour, height=0.6)
        # A label wider than its bar is placed outside it, in the bar's colour,
        # rather than overflowing the bar in white.
        if abs(value) >= 0.28 * widest:
            ax.text(running + value / 2, index, f"{value:+.3f}", ha="center",
                    va="center", fontsize=7.5, color="white")
        else:
            side = 1 if value >= 0 else -1
            ax.text(running + value + side * 0.04 * widest, index, f"{value:+.3f}",
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=7.5, color=colour)
        running += value
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(baseline, color=NEUTRAL, linestyle="--", linewidth=1.0,
               label=f"baseline {baseline:.3f}")
    ax.axvline(final, color="black", linestyle="-", linewidth=1.0,
               label=f"prediction {final:.3f}")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def heatmap(ax, matrix: np.ndarray, row_labels: Sequence[str],
            col_labels: Sequence[str], cmap: str = "RdBu_r",
            centre: float = 0.0, fmt: str = "{:.2f}"):
    matrix = np.asarray(matrix, dtype=float)
    finite = matrix[np.isfinite(matrix)]
    span = float(np.nanmax(np.abs(finite - centre))) if finite.size else 1.0
    span = span if span > 0 else 1.0
    image = ax.imshow(matrix, cmap=cmap, vmin=centre - span, vmax=centre + span,
                      aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5,
                        color="0.35")
                continue
            shade = "white" if abs(value - centre) > 0.62 * span else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center",
                    fontsize=7.5, color=shade)
    return image


def radar(ax, metric_names: Sequence[str], series: Dict[str, Sequence[float]]):
    """Per-group performance profile. Axes are shared, so shapes are comparable."""
    count = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    angles += angles[:1]
    for name, values in series.items():
        closed = list(np.asarray(values, dtype=float)) + [float(values[0])]
        ax.plot(angles, closed, linewidth=1.6, label=name)
        ax.fill(angles, closed, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    return ax
