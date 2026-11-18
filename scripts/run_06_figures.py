r"""Stage 6: Figure 9, as separate figures, one file per plot.

    python scripts\run_06_figures.py ^
        --table-b results\table_b_10seed\table_b_aggregated_allsplits.csv ^
        --output-dir results\figures

Writes, as PNG and PDF at 300 dpi:

    figure9_slope_<dataset>.{png,pdf}   one per dataset: the label-level gap,
                                        then each model's prediction gap
    figure9_amplification.{png,pdf}     change relative to the labels, sorted

Pass --combined to additionally write the three-panel version.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REFERENCE = "ManualAug-CL"
TARGET = "GMACORE-TM"
STAGES = ["Benchmark\nlabels", "ManualAug-CL", "GMACORE-TM"]
# The two model stages are named, not described as "no GAN" versus "GAN".
# ManualAug-CL also lacks the momentum encoder and age-weighted negatives,
# so the contrast is between complete training pipelines (R1-1, R5-1).


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the Figure 9 set.")
    parser.add_argument("--table-b", required=True,
                        help="table_b_aggregated*.csv from stage 3.")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Default: every dataset present in the table.")
    parser.add_argument("--combined", action="store_true",
                        help="Also write the three-panel composite.")
    parser.add_argument("--drop-underpowered", action="store_true", default=True)
    parser.add_argument("--keep-underpowered", dest="drop_underpowered",
                        action="store_false")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _metric_columns(frame: pd.DataFrame, dataset: str):
    sub = frame[frame.dataset == dataset]
    if sub["data_statistical_parity_diff_mean"].notna().any():
        return ("data_statistical_parity_diff_mean",
                "model_statistical_parity_diff_mean",
                "Statistical parity difference")
    return ("data_mean_gap_mean", "model_mean_gap_mean",
            "Group mean gap (target units)")


def _save(figure, output_dir: str, stem: str, dpi: int) -> None:
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(figure)


def slope_panel(axis, frame: pd.DataFrame, dataset: str, drop_underpowered: bool,
                legend_outside: bool = True) -> int:
    data_col, model_col, label = _metric_columns(frame, dataset)
    sub = frame[frame.dataset == dataset]
    if drop_underpowered:
        sub = sub[~sub["underpowered"].astype(bool)]

    groups = sorted(sub.group.unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    drawn = 0

    for colour, group in zip(palette, groups):
        rows = sub[sub.group == group]
        reference = rows[rows.model == REFERENCE]
        target = rows[rows.model == TARGET]
        if reference.empty or target.empty:
            continue
        series = [
            float(reference.iloc[0][data_col]),
            float(reference.iloc[0][model_col]),
            float(target.iloc[0][model_col]),
        ]
        if not np.all(np.isfinite(series)):
            continue
        axis.plot([0, 1, 2], series, marker="o", color=colour, linewidth=2.0,
                  markersize=6, label=group.replace("RULE:", ""))
        drawn += 1

    axis.axhline(0.0, color="0.6", linewidth=0.9, linestyle="--", zorder=0)
    axis.set_xticks([0, 1, 2])
    axis.set_xticklabels(STAGES, fontsize=9)
    axis.set_xlim(-0.12, 2.12)
    axis.set_ylabel(label, fontsize=10)
    axis.set_title(dataset.upper(), fontsize=12, fontweight="bold")
    axis.tick_params(labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    if legend_outside and drawn:
        axis.legend(fontsize=8, frameon=False, ncol=min(4, max(1, drawn)),
                    loc="upper center", bbox_to_anchor=(0.5, -0.13))
    return drawn


def amplification_panel(axis, frame: pd.DataFrame, drop_underpowered: bool,
                        dataset: str | None = None) -> int:
    """Amplification bars. One dataset at a time by default.

    Mixing datasets on one axis would put a QM9 gap measured in Hartree
    (order 1e-4) beside an ESOL gap in log-solubility units (order 1), which
    makes every QM9 bar invisible and the shared axis meaningless.
    """
    sub = frame.copy()
    if dataset is not None:
        sub = sub[sub.dataset == dataset]
    if drop_underpowered:
        sub = sub[~sub["underpowered"].astype(bool)]
    sub = sub[np.isfinite(sub["amplification_mean"])]
    if sub.empty:
        return 0
    key = (sub.group.str.replace("RULE:", "", regex=False) if dataset is not None
           else sub.dataset + " / " + sub.group.str.replace("RULE:", "", regex=False))
    sub = sub.assign(key=key)
    labels = sorted(sub.key.unique(),
                    key=lambda k: sub[sub.key == k]["amplification_mean"].mean())
    positions = {k: i for i, k in enumerate(labels)}

    for model, offset, colour in ((REFERENCE, -0.2, "#4C72B0"),
                                  (TARGET, 0.2, "#C44E52")):
        rows = sub[sub.model == model]
        if rows.empty:
            continue
        axis.barh([positions[k] + offset for k in rows.key],
                  rows["amplification_mean"], height=0.38, color=colour,
                  label=model, xerr=rows.get("amplification_std"),
                  error_kw=dict(lw=0.7, ecolor="0.4"))

    axis.axvline(0.0, color="0.25", linewidth=1.1)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_ylim(-0.7, len(labels) - 0.3)
    axis.set_xlabel("Amplification:  |model gap| - |label gap|\n"
                    "negative = disparity reduced, positive = amplified", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=9, frameon=False, loc="lower right")
    title = "Change relative to the benchmark labels"
    axis.set_title(f"{dataset.upper()}: {title}" if dataset else title,
                   fontsize=12, fontweight="bold")
    return len(labels)


def main() -> None:
    args = parse_args()
    # Validate the input before creating anything. Creating the output
    # directory first leaves a stray results/ behind on a bad path, which
    # then makes a later `Copy-Item ... results` nest into results/results.
    if not os.path.exists(args.table_b):
        raise SystemExit(f"stage 3 table not found: {args.table_b}\n"
                         f"Run scripts/run_03_bias.py first.")
    os.makedirs(args.output_dir, exist_ok=True)
    frame = pd.read_csv(args.table_b)

    datasets = args.datasets or sorted(frame.dataset.unique())
    datasets = [d for d in datasets if d in set(frame.dataset)]

    for dataset in datasets:
        figure, axis = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
        drawn = slope_panel(axis, frame, dataset, args.drop_underpowered)
        if not drawn:
            print(f"skip {dataset}: no powered groups to draw")
            plt.close(figure)
            continue
        _save(figure, args.output_dir, f"figure9_slope_{dataset}", args.dpi)

    rows = frame if not args.drop_underpowered else frame[~frame["underpowered"].astype(bool)]
    rows = rows[np.isfinite(rows["amplification_mean"])]

    for dataset in datasets:
        n_rows = int(rows[rows.dataset == dataset].groupby("group").ngroups)
        if not n_rows:
            print(f"skip {dataset}: no powered groups to draw")
            continue
        figure, axis = plt.subplots(
            figsize=(7.0, max(3.2, 0.42 * n_rows + 1.6)), constrained_layout=True
        )
        if amplification_panel(axis, frame, args.drop_underpowered, dataset=dataset):
            _save(figure, args.output_dir, f"figure9_amplification_{dataset}", args.dpi)
        else:
            plt.close(figure)

    n_rows = int(rows.groupby(["dataset", "group"]).ngroups)

    if args.combined:
        n_slope = len(datasets)
        figure = plt.figure(figsize=(5.0 * n_slope + 7.0,
                                     min(10.0, max(5.5, 0.17 * n_rows + 2.6))),
                            constrained_layout=True)
        grid = figure.add_gridspec(1, n_slope + 1,
                                   width_ratios=[1] * n_slope + [1.5])
        for position, dataset in enumerate(datasets):
            slope_panel(figure.add_subplot(grid[0, position]), frame, dataset,
                        args.drop_underpowered)
        amplification_panel(figure.add_subplot(grid[0, n_slope]), frame,
                            args.drop_underpowered)
        figure.suptitle(
            "Disparities carried by the benchmark data, "
            "and what each model does with them",
            fontsize=13, fontweight="bold",
        )
        _save(figure, args.output_dir, "figure9_combined", args.dpi)


if __name__ == "__main__":
    main()
