r"""Stage 11: Figures 7 and 8, downstream group fairness under domain rules.

    python scripts\run_11_fairness_diagnostics.py ^
        --predictions results\predictions ^
        --output-dir results\figures_fairness

Reads the stage 2 prediction CSVs, so no checkpoint and no GPU are needed.
Molecules are partitioned by the domain rule R_d of their benchmark into
compliant (G = 0) and non-compliant (G = 1).

Figure 7, regression benchmarks: prediction distributions by group, then
independence, separation and sufficiency. Those three are DALEX constructs and
are computed with the `dalex` package; without it the distribution panels are
written and the diagnostic panel is skipped, rather than being replaced by a
different statistic under the same name. Install with `pip install dalex`.

Figure 8, classification benchmarks: a radar of accuracy, true positive rate,
true negative rate and precision for each group, and a bar chart of five parity
gaps computed as non-compliant minus compliant, where zero is parity.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.fairness_diagnostics import (CLASSIFICATION_GAPS, RADAR_LABELS,
                                            RADAR_METRICS,
                                            classification_fairness,
                                            dalex_regression_fairness,
                                            regression_group_summary)
from xai_audit.motifs import parse_smiles
from xai_audit.plots import (MODEL_COLOURS, apply_style, horizontal_bars, radar,
                             save)
from xai_audit.rules import CANONICAL_RULES

GAP_LABELS = {
    "statistical_parity": "Statistical parity",
    "equal_opportunity": "Equal opportunity (TPR)",
    "predictive_equality": "Predictive equality (FPR)",
    "predictive_parity": "Predictive parity (PPV)",
    "accuracy_equality": "Accuracy equality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figures 7 and 8.")
    parser.add_argument("--predictions", required=True,
                        help="Stage 2 output directory.")
    parser.add_argument("--output-dir", default="results/figures_fairness")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Default: every dataset present in the predictions.")
    parser.add_argument("--split", default="test", choices=["test", "all"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Which seed's predictions to draw. Metrics are "
                             "averaged across seeds in the CSV regardless.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_predictions(directory: str, split: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "*__*__seed*.csv"))):
        stem = os.path.basename(path)[:-4]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        frame = pd.read_csv(path)
        frame["dataset"] = parts[0]
        frame["model"] = parts[1]
        frame["seed"] = int(parts[2].replace("seed", ""))
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no prediction CSVs found in {directory}")
    out = pd.concat(frames, ignore_index=True)
    if split == "test" and "split" in out.columns:
        out = out[out["split"] == "test"].reset_index(drop=True)
    return out


def rule_groups(dataset: str, smiles: pd.Series) -> np.ndarray:
    rule = CANONICAL_RULES.get(dataset)
    out = np.full(len(smiles), np.nan)
    if rule is None:
        return out
    cache = {}
    for position, value in enumerate(smiles.astype(str)):
        if value not in cache:
            mol = parse_smiles(value)
            cache[value] = float(rule.group_label(mol)) if mol is not None else np.nan
        out[position] = cache[value]
    return out


def figure7(dataset, by_model, output_dir, dpi):
    """Prediction distributions by group, and the DALEX diagnostics beside them."""
    models = list(by_model)
    figure, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.8),
                                squeeze=False)
    for ax, model_label in zip(axes[0], models):
        frame = by_model[model_label]
        for key, mask, colour in (("compliant", frame.group == 0, "#3b6ea5"),
                                  ("non-compliant", frame.group == 1, "#c0504d")):
            values = frame.loc[mask, "y_pred"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=28, alpha=0.55, color=colour, density=True,
                        label=f"{key} (n = {values.size})")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset.upper()} | {model_label}")
        ax.legend(fontsize=8)
    figure.tight_layout()
    save(figure, output_dir, f"figure7a_prediction_distributions_{dataset}", dpi)

    rows, missing = [], False
    for model_label, frame in by_model.items():
        mask = np.isfinite(frame["group"].to_numpy(dtype=float))
        result = dalex_regression_fairness(
            frame.loc[mask, "y_true"].to_numpy(dtype=float),
            frame.loc[mask, "y_pred"].to_numpy(dtype=float),
            frame.loc[mask, "group"].to_numpy(dtype=int),
        )
        if result is None:
            missing = True
            continue
        for criterion, value in result.items():
            rows.append({"model": model_label, "criterion": criterion, "value": value})

    if missing:
        print("    dalex is not installed, skipping the Figure 7 diagnostic panel. "
              "pip install dalex")
    if not rows:
        return []

    diagnostics = pd.DataFrame(rows)
    criteria = ["independence", "separation", "sufficiency"]
    criteria = [c for c in criteria if c in set(diagnostics.criterion)]
    positions = np.arange(len(criteria))
    width = 0.8 / max(len(by_model), 1)

    figure, ax = plt.subplots(figsize=(7.0, 3.6))
    for offset, model_label in enumerate(by_model):
        sub = diagnostics[diagnostics.model == model_label].set_index("criterion")
        values = [float(sub.loc[c, "value"]) if c in sub.index else np.nan
                  for c in criteria]
        ax.bar(positions + offset * width, values, width=width,
               label=model_label, color=MODEL_COLOURS.get(model_label))
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions + width * (len(by_model) - 1) / 2)
    ax.set_xticklabels([c.capitalize() for c in criteria])
    ax.set_ylabel("Score (closer to one is better parity)")
    ax.set_title(f"{dataset.upper()}: group fairness diagnostics")
    ax.legend(fontsize=8)
    figure.tight_layout()
    save(figure, output_dir, f"figure7b_diagnostics_{dataset}", dpi)
    return rows


def figure8(dataset, by_model, threshold, output_dir, dpi):
    results = {}
    for model_label, frame in by_model.items():
        mask = np.isfinite(frame["group"].to_numpy(dtype=float))
        results[model_label] = classification_fairness(
            frame.loc[mask, "y_true"].to_numpy(dtype=float),
            frame.loc[mask, "y_pred"].to_numpy(dtype=float),
            frame.loc[mask, "group"].to_numpy(dtype=int),
            threshold=threshold,
        )

    models = list(results)
    figure, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.4),
                                subplot_kw={"projection": "polar"}, squeeze=False)
    for ax, model_label in zip(axes[0], models):
        series = {
            "compliant": [results[model_label]["compliant"][m] for m in RADAR_METRICS],
            "non-compliant": [results[model_label]["non_compliant"][m]
                              for m in RADAR_METRICS],
        }
        series = {k: [0.0 if not np.isfinite(v) else v for v in vals]
                  for k, vals in series.items()}
        radar(ax, RADAR_LABELS, series)
        ax.set_title(f"{dataset.upper()} | {model_label}", y=1.16)
    figure.tight_layout()
    save(figure, output_dir, f"figure8a_radar_{dataset}", dpi)

    names = list(CLASSIFICATION_GAPS)
    positions = np.arange(len(names))
    width = 0.8 / max(len(models), 1)
    figure, ax = plt.subplots(figsize=(8.2, 3.8))
    for offset, model_label in enumerate(models):
        values = [results[model_label]["gaps"][n] for n in names]
        ax.bar(positions + offset * width, values, width=width, label=model_label,
               color=MODEL_COLOURS.get(model_label))
    ax.axhline(0.0, color="0.35", linewidth=1.0)
    ax.set_xticks(positions + width * (len(models) - 1) / 2)
    ax.set_xticklabels([GAP_LABELS[n] for n in names], rotation=18, ha="right")
    ax.set_ylabel("Non-compliant minus compliant")
    ax.set_title(f"{dataset.upper()}: fairness gaps, zero is parity")
    ax.legend(fontsize=8)
    figure.tight_layout()
    save(figure, output_dir, f"figure8b_gaps_{dataset}", dpi)

    rows = []
    for model_label, result in results.items():
        for group_key in ("compliant", "non_compliant"):
            row = {"dataset": dataset, "model": model_label, "group": group_key}
            row.update(result[group_key])
            rows.append(row)
        gap_row = {"dataset": dataset, "model": model_label, "group": "gap"}
        gap_row.update(result["gaps"])
        rows.append(gap_row)
    return rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    predictions = load_predictions(args.predictions, args.split)
    datasets = args.datasets or sorted(predictions.dataset.unique())
    classification_rows, regression_rows, diagnostic_rows = [], [], []

    for dataset in datasets:
        if dataset not in CANONICAL_RULES:
            print(f"[skip] {dataset}: no domain rule defined")
            continue
        frame = predictions[(predictions.dataset == dataset)
                            & (predictions.seed == args.seed)]
        if frame.empty:
            print(f"[skip] {dataset}: no rows for seed {args.seed}")
            continue

        by_model = {}
        for model_label, sub in frame.groupby("model"):
            sub = sub.copy()
            sub["group"] = rule_groups(dataset, sub["smiles"])
            by_model[model_label] = sub

        is_classification = set(
            np.unique(frame["y_true"].dropna().to_numpy())
        ) <= {0.0, 1.0}
        kind = "classification" if is_classification else "regression"
        counts = {m: int((s.group == 1).sum()) for m, s in by_model.items()}
        print(f"[{dataset}] {kind}, non-compliant per model: {counts}")

        if is_classification:
            classification_rows.extend(
                figure8(dataset, by_model, args.threshold, args.output_dir, args.dpi))
        else:
            diagnostic_rows.extend(
                figure7(dataset, by_model, args.output_dir, args.dpi))
            for model_label, sub in by_model.items():
                mask = np.isfinite(sub["group"].to_numpy(dtype=float))
                summary = regression_group_summary(
                    sub.loc[mask, "y_true"].to_numpy(dtype=float),
                    sub.loc[mask, "y_pred"].to_numpy(dtype=float),
                    sub.loc[mask, "group"].to_numpy(dtype=int))
                for group_key, stats in summary.items():
                    row = {"dataset": dataset, "model": model_label,
                           "group": group_key}
                    row.update(stats)
                    regression_rows.append(row)

    if classification_rows:
        pd.DataFrame(classification_rows).to_csv(
            os.path.join(args.output_dir, "figure8_classification_fairness.csv"),
            index=False)
    if regression_rows:
        pd.DataFrame(regression_rows).to_csv(
            os.path.join(args.output_dir, "figure7_regression_groups.csv"), index=False)
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(
            os.path.join(args.output_dir, "figure7_dalex_diagnostics.csv"), index=False)
    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"split": args.split, "seed": args.seed,
                   "threshold": args.threshold}, h, indent=2)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
