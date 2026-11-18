r"""Stage 10: Figure 6, node- and feature-level explanations for one molecule.

    python scripts\run_10_local_explanations.py ^
        --data-root "D:\...\MolCLR_data\data" ^
        --runs-root "D:\...\runs\runs" ^
        --output-dir results\figures_local --dataset bace

Four panels, matching the figure in the paper:

    (a) the molecule with atoms shaded by their SHAP contribution
    (b) atoms ranked by contribution magnitude
    (c) a waterfall from the background baseline to the final prediction
    (d) LIME at one selected high-impact atom, ranking the local descriptors

Panel (d) needs the `lime` package. Without it the first three panels are still
written and (d) is skipped with a message.

By default the molecule with the largest absolute prediction in the test sample
is used. Pass --smiles to pick one explicitly, which is what you want when
regenerating the exact molecule shown in the paper.
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
from xai_audit.gmacore_model import featurize
from xai_audit.motifs import parse_smiles
from xai_audit.plots import (NEGATIVE, POSITIVE, apply_style, horizontal_bars,
                             save, waterfall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 6, local explanations.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", default="results/figures_local")
    parser.add_argument("--dataset", default="bace")
    parser.add_argument("--model", default="GMACORE-TM", choices=list(MODELS))
    parser.add_argument("--smiles", default=None,
                        help="Explain this molecule instead of an automatic pick.")
    parser.add_argument("--atom", type=int, default=None,
                        help="Atom index for panel (d). Default: largest |contribution|.")
    parser.add_argument("--n-candidates", type=int, default=40,
                        help="Test molecules scored when picking automatically.")
    parser.add_argument("--top-atoms", type=int, default=10,
                        help="Atoms shown in panels (b) and (c).")
    parser.add_argument("--lime-samples", type=int, default=2000)
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument("--l1-reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--qm9-epochs", type=int, default=50)
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def panel_a(mol, node_scores, dataset, model_label, output_dir, dpi):
    """The molecule, atoms shaded by signed SHAP contribution."""
    from io import BytesIO
    import matplotlib.image as mpimg
    from rdkit.Chem.Draw import rdMolDraw2D

    span = float(np.max(np.abs(node_scores))) or 1.0
    colours, highlight = {}, []
    for index in range(min(mol.GetNumAtoms(), node_scores.size)):
        value = float(node_scores[index]) / span
        highlight.append(index)
        if value >= 0:
            colours[index] = (1.0, 1.0 - 0.75 * value, 1.0 - 0.75 * value)
        else:
            colours[index] = (1.0 + 0.75 * value, 1.0 + 0.75 * value, 1.0)

    drawer = rdMolDraw2D.MolDraw2DCairo(620, 480)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=highlight, highlightAtomColors=colours)
    drawer.FinishDrawing()
    image = mpimg.imread(BytesIO(drawer.GetDrawingText()), format="png")

    figure, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{dataset.upper()} | {model_label}\n"
                 "Atom contributions: red raises, blue lowers the prediction")
    figure.tight_layout()
    save(figure, output_dir, f"figure6a_atom_map_{dataset}_{model_label}", dpi)


def panel_b(mol, node_scores, top, dataset, model_label, output_dir, dpi):
    order = np.argsort(-np.abs(node_scores))[:top]
    labels = [f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}" for i in order]
    figure, ax = plt.subplots(figsize=(6.6, max(2.6, 0.38 * len(order) + 1.2)))
    horizontal_bars(ax, labels, node_scores[order],
                    xlabel="SHAP contribution")
    ax.set_title("Atoms ranked by contribution magnitude")
    figure.tight_layout()
    save(figure, output_dir, f"figure6b_atom_ranking_{dataset}_{model_label}", dpi)
    return order


def panel_c(mol, node_scores, order, baseline, prediction, dataset, model_label,
            output_dir, dpi):
    labels = [f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}" for i in order]
    figure, ax = plt.subplots(figsize=(7.4, max(2.8, 0.40 * len(order) + 1.4)))
    waterfall(ax, labels, node_scores[order], baseline, prediction)
    ax.set_xlabel("Model output")
    ax.set_title("Accumulation from the background baseline to the prediction")
    figure.tight_layout()
    save(figure, output_dir, f"figure6c_waterfall_{dataset}_{model_label}", dpi)


def panel_d(wrapper, atom_index, background, dataset, model_label, output_dir,
            dpi, n_samples, seed):
    """LIME over the selected atom's descriptors, holding every other atom fixed."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("    lime is not installed, skipping panel (d). pip install lime")
        return None

    base = wrapper.flatten()
    start = atom_index * N_NODE_FEATURES
    stop = start + N_NODE_FEATURES

    def local_fn(rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        full = np.repeat(base.reshape(1, -1), len(rows), axis=0)
        full[:, start:stop] = rows
        return wrapper(full)

    training = background.reshape(background.shape[0], wrapper.n_nodes,
                                  N_NODE_FEATURES)[:, atom_index, :]
    rng = np.random.default_rng(seed)
    spread = training.std(axis=0)
    spread[spread == 0] = 1e-3
    extra = base[start:stop].reshape(1, -1) + rng.normal(
        0.0, spread, size=(256, N_NODE_FEATURES))
    training = np.vstack([training, extra])

    explainer = LimeTabularExplainer(
        training, feature_names=NODE_FEATURE_NAMES, mode="regression",
        discretize_continuous=False, random_state=seed,
    )
    explanation = explainer.explain_instance(
        base[start:stop], local_fn, num_features=N_NODE_FEATURES,
        num_samples=n_samples,
    )
    pairs = sorted(explanation.as_list(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = [k for k, _ in pairs]
    weights = [v for _, v in pairs]

    figure, ax = plt.subplots(figsize=(6.8, max(2.6, 0.38 * len(labels) + 1.2)))
    horizontal_bars(ax, labels, weights, xlabel="LIME weight")
    ax.set_title(f"Local descriptors driving atom {atom_index}")
    figure.tight_layout()
    save(figure, output_dir, f"figure6d_lime_atom{atom_index}_{dataset}_{model_label}",
         dpi)
    return pairs


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    session = build_session(
        args.dataset, args.model, args.data_root, args.runs_root,
        qm9_target=args.qm9_target, seed=args.seed, epochs=args.epochs,
        qm9_epochs=args.qm9_epochs,
    )

    if args.smiles:
        graph = featurize(args.smiles, [0.0])
        if graph is None:
            raise SystemExit(f"could not featurize: {args.smiles}")
        smiles = args.smiles
    else:
        chosen = sample_test(session, args.n_candidates, seed=args.seed)
        scores = []
        for index in chosen:
            wrapper = session.wrapper(session.graphs[session.test_idx[index]])
            scores.append(abs(float(wrapper(wrapper.flatten().reshape(1, -1))[0])))
        pick = int(chosen[int(np.argmax(scores))])
        graph = session.graphs[session.test_idx[pick]]
        smiles = session.test_smiles[pick]
        print(f"    selected {smiles}")

    mol = parse_smiles(smiles)
    if mol is None:
        raise SystemExit(f"could not parse: {smiles}")

    values, node_scores, wrapper = session.shap_for(
        graph, nsamples=args.nsamples, l1_reg=args.l1_reg, seed=args.seed)
    background = session.background(wrapper, seed=args.seed)
    prediction = float(wrapper(wrapper.flatten().reshape(1, -1))[0])
    baseline = float(np.mean(wrapper(background)))
    print(f"    prediction {prediction:.4f}, background baseline {baseline:.4f}")

    panel_a(mol, node_scores, args.dataset, args.model, args.output_dir, args.dpi)
    order = panel_b(mol, node_scores, args.top_atoms, args.dataset, args.model,
                    args.output_dir, args.dpi)
    panel_c(mol, node_scores, order, baseline, prediction, args.dataset, args.model,
            args.output_dir, args.dpi)

    atom_index = args.atom if args.atom is not None else int(order[0])
    pairs = panel_d(wrapper, atom_index, background, args.dataset, args.model,
                    args.output_dir, args.dpi, args.lime_samples, args.seed)

    pd.DataFrame({
        "atom": np.arange(node_scores.size),
        "symbol": [mol.GetAtomWithIdx(int(i)).GetSymbol()
                   if i < mol.GetNumAtoms() else "" for i in range(node_scores.size)],
        "shap_contribution": node_scores,
    }).to_csv(os.path.join(args.output_dir, "figure6_atom_contributions.csv"),
              index=False)
    if pairs:
        pd.DataFrame(pairs, columns=["descriptor", "lime_weight"]).to_csv(
            os.path.join(args.output_dir, "figure6_lime_weights.csv"), index=False)
    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"dataset": args.dataset, "model": args.model, "smiles": smiles,
                   "atom": atom_index, "prediction": prediction,
                   "baseline": baseline, "l1_reg": args.l1_reg,
                   "nsamples": args.nsamples, "seed": args.seed}, h, indent=2)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
