"""E1 / Table A: data-level characteristics of the benchmark data.

No model and no checkpoint are required. Run this first.

    python scripts/run_01_data_profile.py \
        --data-root "D:/GMACORE_RIE_Submission/Code/MolCLR_data/data" \
        --output-dir results/table_a
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.data_profile import profile_dataset, profile_rule_partition
from xai_audit.datasets import load_benchmark, target_columns_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Table A, the data-level profile.")
    parser.add_argument("--data-root", required=True, help="MolCLR_data/data directory.")
    parser.add_argument(
        "--datasets", nargs="*", default=["bace", "esol", "qm9", "clintox"]
    )
    parser.add_argument("--output-dir", default="results/table_a")
    parser.add_argument(
        "--min-arm-size", type=int, default=20,
        help="Flag groups whose smaller arm falls below this count.",
    )
    parser.add_argument(
        "--qm9-target", default=None,
        help="Restrict QM9 to a single target column. By default all six "
             "targets used by gmacore.data.moleculenet are profiled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    group_frames, rule_frames = [], []
    for name in args.datasets:
        if name == "qm9":
            targets = [args.qm9_target] if args.qm9_target else target_columns_for("qm9")
        else:
            targets = [None]

        for target in targets:
            try:
                frame = load_benchmark(name, args.data_root, target_column=target)
            except (FileNotFoundError, KeyError) as exc:
                print(f"[skip] {name}" + (f" target={target}" if target else "") + f": {exc}")
                continue

            groups = profile_dataset(frame, min_arm_size=args.min_arm_size)
            rules = profile_rule_partition(frame)
            groups.insert(1, "target", frame.attrs["target_column"])
            rules.insert(1, "target", frame.attrs["target_column"])
            group_frames.append(groups)
            rule_frames.append(rules)

    if not group_frames:
        raise SystemExit("No datasets loaded. Check --data-root.")

    groups = pd.concat(group_frames, ignore_index=True)
    rules = pd.concat(rule_frames, ignore_index=True)

    group_path = os.path.join(args.output_dir, "table_a_groups.csv")
    rule_path = os.path.join(args.output_dir, "table_a_rule_partitions.csv")
    groups.to_csv(group_path, index=False)
    rules.to_csv(rule_path, index=False)

    print(f"\nwrote {group_path}\nwrote {rule_path}")
    flagged = groups[groups.get("underpowered", False) == True]  # noqa: E712
    if len(flagged):
        print("\nUnderpowered groups, do not report these as findings without care:")
        print(flagged[["dataset", "group", "n_present", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
