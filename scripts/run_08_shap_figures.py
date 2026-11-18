r"""Stage 8: Figures 2 and 4, motif attribution and novelty-conditioned explanations.

    python scripts\run_08_shap_figures.py ^
        --data-root "D:\...\MolCLR_data\data" ^
        --runs-root "D:\...\runs\runs" ^
        --output-dir results\figures_shap --n-molecules 150

Figure 2, per dataset and model: a beeswarm of per-molecule motif attributions
with the mean absolute impact of the same motifs beside it.

Figure 4, per dataset and model: paired beeswarms over the node-feature
descriptors, novel test molecules on the left and training-similar molecules on
the right, split at the Appendix Table A.4 similarity quantiles.

Attribution uses l1_reg = 0.0. The exploratory notebooks behind the original
submission used num_features(10), which zeroes all but ten attributions and
would empty most rows of a beeswarm.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.attribution import NODE_FEATURE_NAMES, N_NODE_FEATURES
from xai_audit.explain_session import MODELS, build_session, sample_test
from xai_audit.motifs import MotifLibrary
from xai_audit.plots import apply_style, beeswarm, horizontal_bars, save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figures 2 and 4.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", default="results/figures_shap")
    parser.add_argument("--datasets", nargs="*",
                        default=["bace", "esol", "clintox", "qm9"])
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--n-molecules", type=int, default=150)
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument("--l1-reg", type=float, default=0.0)
    parser.add_argument("--q-low", type=float, default=0.25)
    parser.add_argument("--q-high", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--qm9-epochs", type=int, default=50)
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def figure2(dataset, model_label, motif_rows, output_dir, dpi):
    """Beeswarm of motif attributions, with mean absolute impact beside it."""
    motifs = [m for m in motif_rows if np.isfinite(motif_rows[m]).any()]
    if not motifs:
        print(f"    [{dataset}/{model_label}] no motif attributions, skipping Figure 2")
        return None
    order = sorted(motifs, key=lambda m: np.nanmean(np.abs(motif_rows[m])), reverse=True)
    values = {m: motif_rows[m] for m in order}
    means = [float(np.nanmean(np.abs(motif_rows[m]))) for m in order]

    height = max(2.8, 0.42 * len(order) + 1.4)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, height),
                                gridspec_kw={"width_ratios": [2.1, 1.0]})
    beeswarm(axes[0], values)
    axes[0].set_xlabel("SHAP contribution to the model output")
    axes[0].set_title(f"{dataset.upper()} | {model_label}: motif attribution")

    horizontal_bars(axes[1], order, means, colour="#6b6b6b",
                    xlabel="Mean |SHAP| impact")
    axes[1].set_yticklabels([])
    axes[1].set_title("Mean absolute impact")
    figure.tight_layout()
    save(figure, output_dir, f"figure2_motif_attribution_{dataset}_{model_label}", dpi)
    return order


def figure4(dataset, model_label, feature_matrix, novel_mask, similar_mask,
            output_dir, dpi):
    """Paired beeswarms over node-feature descriptors, novel against similar."""
    if novel_mask.sum() < 3 or similar_mask.sum() < 3:
        print(f"    [{dataset}/{model_label}] too few molecules in a novelty arm, "
              f"skipping Figure 4 (novel {int(novel_mask.sum())}, "
              f"similar {int(similar_mask.sum())})")
        return
    order = sorted(range(N_NODE_FEATURES),
                   key=lambda j: np.nanmean(np.abs(feature_matrix[:, j])),
                   reverse=True)
    names = [NODE_FEATURE_NAMES[j] for j in order]

    span = np.nanmax(np.abs(feature_matrix)) if np.isfinite(feature_matrix).any() else 1.0
    span = span * 1.08 if span > 0 else 1.0

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 0.42 * len(names) + 1.9),
                                sharey=True)
    for ax, mask, title in (
        (axes[0], novel_mask, "Novel, low similarity to training"),
        (axes[1], similar_mask, "Training-similar, high similarity"),
    ):
        rows = {NODE_FEATURE_NAMES[j]: feature_matrix[mask, j] for j in order}
        beeswarm(ax, rows)
        ax.set_xlim(-span, span)
        ax.set_xlabel("SHAP contribution")
        ax.set_title(f"{title}  (n = {int(mask.sum())})")
    figure.suptitle(f"{dataset.upper()} | {model_label}: explanations under novelty",
                    y=1.01)
    figure.tight_layout()
    save(figure, output_dir, f"figure4_novelty_{dataset}_{model_label}", dpi)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()
    library = MotifLibrary.build()
    long_rows = []

    for dataset in args.datasets:
        for model_label in args.models:
            session = build_session(
                dataset, model_label, args.data_root, args.runs_root,
                qm9_target=args.qm9_target, seed=args.seed, epochs=args.epochs,
                qm9_epochs=args.qm9_epochs, q_low=args.q_low, q_high=args.q_high,
            )
            chosen = sample_test(session, args.n_molecules, seed=args.seed)
            print(f"    KernelSHAP on {len(chosen)} molecules")

            motif_rows = {m: [] for m in session.groups}
            feature_matrix = np.full((len(chosen), N_NODE_FEATURES), np.nan)

            for position, index in enumerate(chosen):
                graph = session.graphs[session.test_idx[index]]
                smiles = session.test_smiles[index]
                values, nodes, wrapper = session.shap_for(
                    graph, nsamples=args.nsamples, l1_reg=args.l1_reg, seed=args.seed)

                per_feature = values.reshape(wrapper.n_nodes, N_NODE_FEATURES).sum(axis=0)
                feature_matrix[position] = per_feature

                attribution = session.motif_attributions(smiles, nodes, library)
                for motif in session.groups:
                    motif_rows[motif].append(attribution.get(motif, np.nan))

                if (position + 1) % 25 == 0:
                    print(f"      {position + 1}/{len(chosen)}")

            motif_rows = {m: np.asarray(v, dtype=float) for m, v in motif_rows.items()}
            figure2(dataset, model_label, motif_rows, args.output_dir, args.dpi)
            figure4(dataset, model_label, feature_matrix,
                    session.novel[chosen], session.similar[chosen],
                    args.output_dir, args.dpi)

            for motif, values in motif_rows.items():
                for smiles_index, value in zip(chosen, values):
                    long_rows.append({
                        "dataset": dataset, "model": model_label, "kind": "motif",
                        "name": motif, "smiles": session.test_smiles[smiles_index],
                        "attribution": float(value),
                        "similarity": float(session.similarity[smiles_index]),
                        "novel": bool(session.novel[smiles_index]),
                        "similar": bool(session.similar[smiles_index]),
                    })
            for j, name in enumerate(NODE_FEATURE_NAMES):
                for position, smiles_index in enumerate(chosen):
                    long_rows.append({
                        "dataset": dataset, "model": model_label, "kind": "feature",
                        "name": name, "smiles": session.test_smiles[smiles_index],
                        "attribution": float(feature_matrix[position, j]),
                        "similarity": float(session.similarity[smiles_index]),
                        "novel": bool(session.novel[smiles_index]),
                        "similar": bool(session.similar[smiles_index]),
                    })

    pd.DataFrame(long_rows).to_csv(
        os.path.join(args.output_dir, "attribution_long.csv"), index=False)
    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"l1_reg": args.l1_reg, "nsamples": args.nsamples,
                   "n_molecules": args.n_molecules, "seed": args.seed,
                   "q_low": args.q_low, "q_high": args.q_high}, h, indent=2)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
