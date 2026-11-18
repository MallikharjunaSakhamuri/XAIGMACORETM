r"""Stage 2: fine-tune both pretrained encoders and save test-set predictions.

    python scripts/run_02_predict.py ^
        --data-root "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data" ^
        --runs-root "D:\GMACORE_RIE_Submission\Code Run\runs\runs" ^
        --output-dir results/predictions

Writes one CSV per dataset, model and seed, holding the test-set SMILES, the
true target and the model prediction. Those files are the only input the bias
analysis in stage 3 needs.
"""

from __future__ import annotations

import os as _os

# Set before any torch import; see xai_audit/__init__.py for why.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.datasets import BENCHMARKS, load_benchmark, target_columns_for
from xai_audit.finetune import FineTuneConfig, finetune_and_predict
from xai_audit.gmacore_model import featurize

MODELS = {
    "ManualAug-CL": "manualaug_cl_fixed",
    "GMACORE-TM": "gmacore_tm_fixedgan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and predict for the audit.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True,
                        help="Folder holding manualaug_cl_fixed and gmacore_tm_fixedgan.")
    parser.add_argument("--datasets", nargs="*",
                        default=["bace", "esol", "clintox", "qm9"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default="results/predictions")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-molecules", type=int, default=None,
                        help="Cap applied to EVERY dataset. Recorded in the manifest, "
                             "since it changes what was audited.")
    parser.add_argument("--qm9-max-molecules", type=int, default=25000,
                        help="Cap for QM9 only; it has 133,885 molecules and "
                             "fine-tuning it whole is hours per run. Pass 0 for no cap.")
    parser.add_argument("--qm9-epochs", type=int, default=50,
                        help="Epoch budget for QM9, which needs fewer passes than "
                             "the small benchmarks.")
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = FineTuneConfig(
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
    )
    if args.device:
        config.device = args.device
    print(f"device: {config.device}")

    manifest = {"datasets": {}, "models": {}, "max_molecules": args.max_molecules}
    for label, folder in MODELS.items():
        checkpoint = os.path.join(args.runs_root, folder, "best_model.pt")
        stats = os.path.join(args.runs_root, folder, "feature_stats.pt")
        for path in (checkpoint, stats):
            if not os.path.exists(path):
                raise SystemExit(f"{label}: missing {path}")
        manifest["models"][label] = {"checkpoint": checkpoint, "feature_stats": stats}

    gate_rows = []
    for name in args.datasets:
        target = args.qm9_target if name == "qm9" else None
        frame = load_benchmark(name, args.data_root, target_column=target)
        task = frame.attrs["task_type"]

        cap = args.max_molecules
        if name == "qm9" and args.qm9_max_molecules:
            cap = min(cap, args.qm9_max_molecules) if cap else args.qm9_max_molecules
        if cap and len(frame) > cap:
            frame = frame.sample(cap, random_state=0).reset_index(drop=True)
            print(f"  subsampled to {len(frame)} molecules (recorded in the manifest)")

        graphs = []
        for smiles, value in zip(frame["smiles"], frame["target"]):
            graph = featurize(smiles, [float(value)])
            if graph is not None:
                graphs.append(graph)
        print(f"  featurized {len(graphs)} of {len(frame)}")
        manifest["datasets"][name] = {
            "task": task, "target_column": frame.attrs["target_column"],
            "source_file": frame.attrs["source_file"], "n_graphs": len(graphs),
            "cap_applied": cap,
        }

        dataset_config = config
        if name == "qm9" and args.qm9_epochs != config.epochs:
            dataset_config = replace(config, epochs=args.qm9_epochs)
            print(f"  QM9 epoch budget: {args.qm9_epochs}")

        for label, folder in MODELS.items():
            for seed in args.seeds:
                result = finetune_and_predict(
                    graphs=graphs, task=task,
                    checkpoint_path=manifest["models"][label]["checkpoint"],
                    stats_path=manifest["models"][label]["feature_stats"],
                    model_name=label, dataset_name=name, seed=seed,
                    config=dataset_config,
                    num_tasks=1,
                )
                out = pd.DataFrame({
                    "smiles": result.smiles,
                    "split": result.split,
                    "y_true": result.y_true[:, 0],
                    "y_pred": result.y_pred[:, 0],
                })
                path = os.path.join(args.output_dir, f"{name}__{label}__seed{seed}.csv")
                out.to_csv(path, index=False)

                gate = dict(result.gate)
                gate.update({"dataset": name, "model": label, "seed": seed,
                             "epochs_run": result.epochs_run,
                             "n_test": int((out["split"] == "test").sum()),
                             "n_all": len(out)})
                gate_rows.append(gate)
                flag = "" if gate.get("passes", 0.0) == 1.0 else "   <-- FAILS GATE"
                summary = (f"roc_auc={gate.get('roc_auc', float('nan')):.4f}"
                           if task == "classification"
                           else f"r2={gate.get('r2', float('nan')):+.4f} "
                                f"rmse={gate.get('rmse', float('nan')):.4f} "
                                f"vs mean-predictor {gate.get('mean_predictor_rmse', float('nan')):.4f}")
                print(f"  {name:9} {label:14} seed {seed}  {summary}{flag}")

    gates = pd.DataFrame(gate_rows)
    gates.to_csv(os.path.join(args.output_dir, "performance_gate.csv"), index=False)
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    failed = gates[gates["passes"] != 1.0]
    print(f"\nwrote predictions to {args.output_dir}")
    if len(failed):
        print("\nThese runs FAILED the performance gate. Their predictions cannot")
        print("support a disparity claim; resolve before running stage 3:")
        print(failed[["dataset", "model", "seed"]].to_string(index=False))
    else:
        print("All runs passed the performance gate.")


if __name__ == "__main__":
    main()
