r"""Stage 4: measures 1 and 2, motif concentration and explanation stability.

    python scripts\run_04_attribution.py ^
        --data-root "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data" ^
        --runs-root "D:\GMACORE_RIE_Submission\Code Run\runs\runs" ^
        --output-dir results\table_b_attribution

Fine-tunes once per dataset and checkpoint, then runs KernelSHAP on a sample
of test molecules. This is the only expensive stage; use --n-molecules to
trade precision for time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from xai_audit.datasets import load_benchmark
from xai_audit.finetune import FineTuneConfig, train_downstream_model
from xai_audit.gmacore_model import featurize
from xai_audit.motifs import DATASET_GROUPS
from xai_audit.attribution import novelty_partition
from xai_audit.shap_runner import (
    BatchedGraphWrapper,
    attribute_molecules,
    masking_background,
    max_similarity_to_reference,
    summarise,
)

MODELS = {
    "ManualAug-CL": "manualaug_cl_fixed",
    "GMACORE-TM": "gmacore_tm_fixedgan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP-based bias measures.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--datasets", nargs="*",
                        default=["bace", "esol", "clintox", "qm9"])
    parser.add_argument("--output-dir", default="results/table_b_attribution")
    parser.add_argument("--n-molecules", type=int, default=150,
                        help="Test molecules to explain per dataset and model.")
    parser.add_argument("--nsamples", type=int, default=None,
                        help="KernelSHAP draws. Default follows 2M + 2048.")
    parser.add_argument("--l1-reg", type=float, default=0.0)
    parser.add_argument("--q-low", type=float, default=0.25)
    parser.add_argument("--q-high", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--qm9-max-molecules", type=int, default=25000,
                        help="Cap for QM9 only, matching stage 2 so the two are "
                             "comparable. Pass 0 for no cap.")
    parser.add_argument("--qm9-epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = FineTuneConfig(epochs=args.epochs)
    print(f"device: {config.device}")

    summary_rows, importance_rows = [], []

    for name in args.datasets:
        target = args.qm9_target if name == "qm9" else None
        frame = load_benchmark(name, args.data_root, target_column=target)
        cap = args.max_molecules
        if name == "qm9" and args.qm9_max_molecules:
            cap = min(cap, args.qm9_max_molecules) if cap else args.qm9_max_molecules
        if cap and len(frame) > cap:
            frame = frame.sample(cap, random_state=0).reset_index(drop=True)
            print(f"  subsampled to {len(frame)} molecules")
        task = frame.attrs["task_type"]

        dataset_config = config
        if name == "qm9" and args.qm9_epochs != config.epochs:
            dataset_config = replace(config, epochs=args.qm9_epochs)

        graphs = [g for g in (featurize(s, [float(v)])
                              for s, v in zip(frame["smiles"], frame["target"])) if g]
        print(f"  featurized {len(graphs)}")
        groups = DATASET_GROUPS[name]

        for label, folder in MODELS.items():
            checkpoint = os.path.join(args.runs_root, folder, "best_model.pt")
            stats_path = os.path.join(args.runs_root, folder, "feature_stats.pt")
            for path in (checkpoint, stats_path):
                if not os.path.exists(path):
                    raise SystemExit(f"{label}: missing {path}")

            print(f"  [{name}/{label}] fine-tuning")
            model, stats, splits, valid_score = train_downstream_model(
                graphs, task, checkpoint, stats_path, seed=args.seed,
                config=dataset_config, num_tasks=1,
            )
            train_idx, _, test_idx = splits
            print(f"    validation score {valid_score:.4f}")

            train_smiles = [graphs[i].smiles for i in train_idx]
            test_smiles = [graphs[i].smiles for i in test_idx]
            similarity = max_similarity_to_reference(test_smiles, train_smiles)
            novel, similar = novelty_partition(similarity, args.q_low, args.q_high)

            rng = np.random.default_rng(args.seed)
            take = min(args.n_molecules, len(test_idx))
            chosen = np.sort(rng.choice(len(test_idx), size=take, replace=False))
            sample_graphs = [graphs[test_idx[i]] for i in chosen]
            sample_smiles = [test_smiles[i] for i in chosen]

            def factory(graph, _stats=stats, _model=model):
                wrapper = BatchedGraphWrapper(_model, graph, _stats,
                                              device=dataset_config.device)
                background = masking_background(wrapper, _stats[0], n_samples=24,
                                                seed=args.seed)
                return wrapper, background

            print(f"    KernelSHAP on {take} molecules")
            per_molecule, features = attribute_molecules(
                factory, sample_smiles, sample_graphs, groups,
                nsamples=args.nsamples, l1_reg=args.l1_reg, seed=args.seed,
            )

            result = summarise(
                name, label, per_molecule, features,
                novel[chosen], similar[chosen],
                nsamples=args.nsamples or -1, l1_reg=args.l1_reg,
            )
            row = {"dataset": name, "model": label, "n_molecules": result.n_molecules,
                   "top3_share": result.top3_share, "top5_share": result.top5_share,
                   "n_novel": int(novel[chosen].sum()),
                   "n_similar": int(similar[chosen].sum()),
                   "median_similarity": float(np.nanmedian(similarity))}
            row.update(result.stability)
            summary_rows.append(row)
            for motif, value in result.motif_importance.items():
                importance_rows.append({"dataset": name, "model": label,
                                        "motif": motif, "importance": value})
            print(f"    top-3 motif share {result.top3_share:.4f} | "
                  f"sign consistency novel {result.stability['sign_consistency_novel']:.4f} "
                  f"vs similar {result.stability['sign_consistency_similar']:.4f}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "attribution_summary.csv"), index=False)
    pd.DataFrame(importance_rows).to_csv(
        os.path.join(args.output_dir, "motif_importance.csv"), index=False)
    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"l1_reg": args.l1_reg, "nsamples": args.nsamples,
                   "n_molecules": args.n_molecules, "seed": args.seed,
                   "q_low": args.q_low, "q_high": args.q_high}, h, indent=2)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
