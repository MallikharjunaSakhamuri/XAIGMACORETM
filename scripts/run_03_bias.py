r"""Stage 3: Table B, measures 3 and 4, from the stage 2 predictions.

    python scripts\run_03_bias.py --predictions results\predictions --output-dir results\table_b

No model is loaded and no GPU is used. Reads only the prediction CSVs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.table_b import (
    aggregate_seeds,
    apply_multiple_comparisons,
    contrast_models,
    disparity_for_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Table B from predictions.")
    parser.add_argument("--predictions", default="results/predictions")
    parser.add_argument("--output-dir", default="results/table_b")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for classification outputs.")
    parser.add_argument("--min-arm-size", type=int, default=20)
    parser.add_argument("--split", default="test", choices=["test", "all"],
                        help="'test' is the clean primary analysis. 'all' pools every "
                             "molecule for power and MUST be reported as such.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--reference-model", default="ManualAug-CL")
    parser.add_argument("--target-model", default="GMACORE-TM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.predictions, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"manifest.json not found in {args.predictions}; rerun stage 2.")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    files = sorted(glob.glob(os.path.join(args.predictions, "*__*__seed*.csv")))
    if not files:
        raise SystemExit(f"no prediction CSVs in {args.predictions}")

    per_seed = []
    for path in files:
        stem = os.path.basename(path)[: -len(".csv")]
        dataset, model, seed_part = stem.split("__")
        seed = int(seed_part.replace("seed", ""))
        task = manifest["datasets"][dataset]["task"]

        frame = pd.read_csv(path)
        if "split" not in frame.columns:
            # Files written before stage 2 gained the split column hold test
            # molecules only. Treating them as the pooled set would mislabel
            # the output, so this is refused rather than silently accepted.
            raise SystemExit(
                f"{os.path.basename(path)} has no 'split' column, so it predates the "
                "current stage 2 and contains test molecules only. Rerun stage 2 to "
                "regenerate the predictions with split labels."
            )
        if args.split == "test":
            frame = frame[frame["split"] == "test"].reset_index(drop=True)
        per_seed.append(
            disparity_for_run(
                frame, dataset=dataset, task=task, model=model, seed=seed,
                threshold=args.threshold, min_arm_size=args.min_arm_size,
            )
        )
        print(f"  {dataset:9} {model:14} seed {seed}  n={len(frame)} ({args.split})")

    per_seed_table = pd.concat(per_seed, ignore_index=True)
    aggregated = aggregate_seeds(per_seed_table)
    aggregated = apply_multiple_comparisons(aggregated, alpha=args.alpha)
    contrast = contrast_models(
        aggregated, reference=args.reference_model, target=args.target_model
    )

    suffix = "" if args.split == "test" else "_allsplits"
    paths = {
        f"table_b_per_seed{suffix}.csv": per_seed_table,
        f"table_b_aggregated{suffix}.csv": aggregated,
        f"table_b_model_contrast{suffix}.csv": contrast,
    }
    for filename, table in paths.items():
        table.to_csv(os.path.join(args.output_dir, filename), index=False)
        print(f"wrote {os.path.join(args.output_dir, filename)}")

    usable = aggregated[~aggregated["underpowered"].astype(bool)]
    significant = usable[usable["significant_bh"].astype(bool)]
    print(f"\ngroup tests            : {len(aggregated)}")
    print(f"underpowered, excluded : {int(aggregated['underpowered'].astype(bool).sum())}")
    print(f"significant after BH   : {len(significant)} of {len(usable)} powered tests")

    if len(contrast):
        counts = contrast["direction"].value_counts().to_dict()
        print("\nDirection of change, generative against handcrafted augmentation:")
        for key, value in counts.items():
            print(f"  {key:28} {value}")


if __name__ == "__main__":
    main()
