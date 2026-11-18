r"""Stage 9: Figure 3, counterfactual motif ablations.

    python scripts\run_09_counterfactual.py ^
        --data-root "D:\...\MolCLR_data\data" ^
        --runs-root "D:\...\runs\runs" ^
        --output-dir results\figures_counterfactual --datasets bace esol

For each sampled test molecule, the k most strongly attributed motifs are
removed one at a time and the molecule is re-scored. The change in prediction,
delta, is the counterfactual effect of that motif. k defaults to 3, matching
Appendix Table A.4.

Removal is structural: the motif's atoms are deleted from the molecular graph
and the result is re-sanitised, so the ablated molecule is a real molecule and
not a feature mask. Edits that fail sanitisation are recorded as invalid.

Writes a triptych per example, matching the figure in the paper: the original
molecule with its score, the removed motif, and the edited molecule with its
score and the annotated delta. Also writes a summary of whether attributed
importance predicts the measured effect.
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

from xai_audit.counterfactual import ablate_top_motifs, concordance
from xai_audit.explain_session import MODELS, build_session, sample_test
from xai_audit.motifs import MotifLibrary, parse_smiles
from xai_audit.plots import NEGATIVE, POSITIVE, apply_style, horizontal_bars, save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 3, counterfactual ablations.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", default="results/figures_counterfactual")
    parser.add_argument("--datasets", nargs="*", default=["bace", "esol"],
                        help="The paper shows BACE and ESOL.")
    parser.add_argument("--models", nargs="*", default=["GMACORE-TM"])
    parser.add_argument("--n-molecules", type=int, default=60,
                        help="Molecules to attribute and ablate per dataset.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Motifs ablated per molecule. Appendix Table A.4 uses 3.")
    parser.add_argument("--n-examples", type=int, default=4,
                        help="Triptych panels drawn per dataset, largest |delta| first.")
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument("--l1-reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--qm9-epochs", type=int, default=50)
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _draw(mol, size=(320, 260), highlight=None):
    from rdkit.Chem.Draw import rdMolDraw2D
    from io import BytesIO
    import matplotlib.image as mpimg

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=list(highlight) if highlight else [])
    drawer.FinishDrawing()
    return mpimg.imread(BytesIO(drawer.GetDrawingText()), format="png")


def triptych(ablation, library, output_dir, dataset, model_label, index, dpi):
    """Original molecule, the motif removed, and the edited molecule with delta."""
    original = parse_smiles(ablation.smiles)
    modified = parse_smiles(ablation.modified_smiles) if ablation.modified_smiles else None
    if original is None or modified is None:
        return
    atoms = library.atom_set(original, ablation.motif)

    try:
        left = _draw(original, highlight=atoms)
        right = _draw(modified)
    except Exception as exc:
        print(f"    could not render {ablation.smiles}: {exc}")
        return

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                                gridspec_kw={"width_ratios": [1, 0.55, 1]})
    axes[0].imshow(left)
    axes[0].set_title(f"Original\nprediction {ablation.prediction_original:.3f}",
                      fontsize=10)
    axes[1].text(0.5, 0.62, ablation.motif.replace("_", " "), ha="center",
                 va="center", fontsize=13, fontweight="bold")
    axes[1].text(0.5, 0.42, f"{ablation.n_atoms_removed} atoms removed",
                 ha="center", va="center", fontsize=9, color="0.35")
    axes[1].text(0.5, 0.22,
                 f"$\\Delta$ = {ablation.delta:+.3f}", ha="center", va="center",
                 fontsize=12,
                 color=POSITIVE if ablation.delta >= 0 else NEGATIVE)
    axes[2].imshow(right)
    axes[2].set_title(f"Ablated\nprediction {ablation.prediction_modified:.3f}",
                      fontsize=10)
    for ax in axes:
        ax.axis("off")
    figure.suptitle(f"{dataset.upper()} | {model_label}", y=1.02, fontsize=11)
    figure.tight_layout()
    save(figure, output_dir,
         f"figure3_ablation_{dataset}_{model_label}_{index:02d}_{ablation.motif}", dpi)


def summary_panel(rows, dataset, model_label, output_dir, dpi):
    """Mean delta per motif, which is the aggregate behind the triptychs."""
    frame = pd.DataFrame([r.as_row() for r in rows])
    frame = frame[frame.valid & np.isfinite(frame.delta)]
    if frame.empty:
        return
    grouped = frame.groupby("motif")["delta"].agg(["mean", "std", "count"])
    grouped = grouped.sort_values("mean")
    figure, ax = plt.subplots(figsize=(7.4, max(2.6, 0.42 * len(grouped) + 1.2)))
    horizontal_bars(ax, list(grouped.index), grouped["mean"].to_numpy(),
                    xlabel="Mean change in prediction on removal ($\\Delta$)")
    for position, (_, row) in enumerate(grouped.iterrows()):
        ax.text(row["mean"], position, f"  n={int(row['count'])}", va="center",
                fontsize=7.5, color="0.3")
    ax.set_title(f"{dataset.upper()} | {model_label}: effect of removing each motif")
    figure.tight_layout()
    save(figure, output_dir, f"figure3_summary_{dataset}_{model_label}", dpi)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()
    library = MotifLibrary.build()

    all_rows, concordance_rows = [], []

    for dataset in args.datasets:
        for model_label in args.models:
            session = build_session(
                dataset, model_label, args.data_root, args.runs_root,
                qm9_target=args.qm9_target, seed=args.seed, epochs=args.epochs,
                qm9_epochs=args.qm9_epochs,
            )
            chosen = sample_test(session, args.n_molecules, seed=args.seed)
            print(f"    attributing {len(chosen)} molecules")

            smiles_list, attributions = [], []
            for position, index in enumerate(chosen):
                graph = session.graphs[session.test_idx[index]]
                smiles = session.test_smiles[index]
                _, nodes, _ = session.shap_for(
                    graph, nsamples=args.nsamples, l1_reg=args.l1_reg, seed=args.seed)
                smiles_list.append(smiles)
                attributions.append(session.motif_attributions(smiles, nodes, library))
                if (position + 1) % 20 == 0:
                    print(f"      {position + 1}/{len(chosen)}")

            print(f"    ablating top-{args.top_k} motifs per molecule")
            ablations = ablate_top_motifs(
                session.model, smiles_list, attributions, library,
                k=args.top_k, device=session.device,
            )

            valid = [a for a in ablations if a.valid]
            print(f"    {len(valid)} of {len(ablations)} ablations produced a valid molecule")

            stats = concordance(ablations)
            stats.update({"dataset": dataset, "model": model_label,
                          "n_ablations": len(ablations), "n_valid": len(valid)})
            concordance_rows.append(stats)
            print(f"    |attribution| vs |delta|: rho = {stats['spearman_rho']:.3f}, "
                  f"sign agreement = {stats['sign_agreement']:.3f}")

            for a in ablations:
                row = a.as_row()
                row.update({"dataset": dataset, "model": model_label})
                all_rows.append(row)

            summary_panel(ablations, dataset, model_label, args.output_dir, args.dpi)
            examples = sorted(valid, key=lambda a: abs(a.delta), reverse=True)
            for position, ablation in enumerate(examples[: args.n_examples]):
                triptych(ablation, library, args.output_dir, dataset, model_label,
                         position, args.dpi)

    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.output_dir, "counterfactual_ablations.csv"), index=False)
    pd.DataFrame(concordance_rows).to_csv(
        os.path.join(args.output_dir, "counterfactual_concordance.csv"), index=False)
    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"top_k": args.top_k, "n_molecules": args.n_molecules,
                   "l1_reg": args.l1_reg, "nsamples": args.nsamples,
                   "seed": args.seed}, h, indent=2)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
