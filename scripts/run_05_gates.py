r"""Stage 5: what the GAN generator keeps and drops, by motif.

    python scripts\run_05_gates.py ^
        --data-root "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data" ^
        --runs-root "D:\GMACORE_RIE_Submission\Code Run\runs\runs" ^
        --output-dir results\generator_gates

Uses the generator stored in the GMACORE-TM checkpoint. No fine-tuning and no
training; this reads the augmentation mechanism directly.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from xai_audit.datasets import load_benchmark
from xai_audit.generator import load_generator, motif_gate_profile
from xai_audit.gmacore_model import featurize, load_feature_stats
from xai_audit.motifs import DATASET_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generator gate profile by motif.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--variant", default="gmacore_tm_fixedgan")
    parser.add_argument("--datasets", nargs="*",
                        default=["bace", "esol", "clintox", "qm9"])
    parser.add_argument("--output-dir", default="results/generator_gates")
    parser.add_argument("--n-molecules", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=8,
                        help="Noise draws per molecule; the generator is stochastic.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qm9-target", default="gap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    folder = os.path.join(args.runs_root, args.variant)
    checkpoint = os.path.join(folder, "best_model.pt")
    stats_path = os.path.join(folder, "feature_stats.pt")
    for path in (checkpoint, stats_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    generator, payload = load_generator(checkpoint, device=device)
    stats = load_feature_stats(stats_path, device=device)
    metrics = payload.get("metrics", {})
    print(f"generator from {args.variant}, epoch {payload.get('epoch')}, "
          f"keep_ratio {generator.keep_ratio}")
    print(f"recorded during training: keep_node "
          f"{metrics.get('keep_node', float('nan')):.4f}, dropped_frac "
          f"{metrics.get('dropped_frac', float('nan')):.4f}")

    frames = []
    for name in args.datasets:
        target = args.qm9_target if name == "qm9" else None
        frame = load_benchmark(name, args.data_root, target_column=target)
        if len(frame) > args.n_molecules:
            frame = frame.sample(args.n_molecules, random_state=args.seed).reset_index(drop=True)

        graphs, smiles = [], []
        for value in frame["smiles"]:
            graph = featurize(value, [0.0])
            if graph is not None:
                graphs.append(graph)
                smiles.append(value)

        profile = motif_gate_profile(
            generator, graphs, smiles, DATASET_GROUPS[name], stats,
            device=device, repeats=args.repeats, seed=args.seed,
        )
        profile.insert(0, "dataset", name)
        profile["background_gate_score"] = profile.attrs["background_gate_score"]
        profile["background_drop_rate"] = profile.attrs["background_drop_rate"]
        frames.append(profile)
        print(f"  [{name}] {len(graphs)} molecules, background gate score "
              f"{profile.attrs['background_gate_score']:.4f}, drop rate "
              f"{profile.attrs['background_drop_rate']:.4f}")

    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(args.output_dir, "generator_gate_profile.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")

    powered = out[out["n_occurrences"] >= 10].copy()
    if len(powered):
        powered = powered.sort_values("gate_score_vs_background", ascending=False)
        print("\nMotifs the generator most and least preferentially retains:")
        cols = ["dataset", "motif", "n_occurrences", "gate_score",
                "gate_score_vs_background", "drop_rate"]
        print(powered[cols].head(6).round(4).to_string(index=False))
        print("  ...")
        print(powered[cols].tail(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
