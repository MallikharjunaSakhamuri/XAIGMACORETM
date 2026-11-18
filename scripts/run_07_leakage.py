r"""Stage 7: leakage between the pretraining corpus and the benchmarks (R1-5).

    python scripts\run_07_leakage.py ^
        --data-root "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data" ^
        --corpus "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data\pubchem-10m-clean.txt" ^
        --output-dir results\leakage

Reviewer 1 point 5 asks whether molecules seen during generative pretraining
resemble benchmark test molecules closely enough to inflate the results. This
compares the two sets directly: exact overlap on canonical SMILES, and
nearest-neighbour Tanimoto similarity.

By default only the first 100,000 unique molecules are scanned, because that
is what the pretraining runs actually consumed: step 1 read the corpus with
LIMIT = 100_000 after de-duplication, and both run_config.json files record
num_graphs = 100000. Scanning the whole 10M file would answer a question the
models were never asked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.datasets import load_benchmark
from xai_audit.gmacore_model import scaffold_split
from xai_audit.leakage import check_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretraining corpus leakage check.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--corpus", required=True,
                        help="The pubchem SMILES file used for pretraining.")
    parser.add_argument("--datasets", nargs="*",
                        default=["bace", "esol", "clintox", "qm9"])
    parser.add_argument("--output-dir", default="results/leakage")
    parser.add_argument("--limit", type=int, default=100_000,
                        help="Corpus molecules to scan. 0 scans the whole file.")
    parser.add_argument("--test-only", action="store_true", default=True,
                        help="Compare only held-out molecules, which is what the "
                             "reviewer comment concerns.")
    parser.add_argument("--all-molecules", dest="test_only", action="store_false")
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--max-benchmark", type=int, default=3000,
                        help="Cap benchmark molecules compared, for runtime.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.corpus):
        raise SystemExit(f"corpus not found: {args.corpus}")

    limit = args.limit if args.limit > 0 else None
    print(f"corpus: {args.corpus}")
    print(f"scanning {'all' if limit is None else f'{limit:,}'} molecules\n")

    rows, distributions = [], []
    for name in args.datasets:
        target = args.qm9_target if name == "qm9" else None
        frame = load_benchmark(name, args.data_root, target_column=target)
        smiles = frame["smiles"].astype(str).tolist()

        if args.test_only:
            _, _, test_idx = scaffold_split(smiles)
            smiles = [smiles[i] for i in test_idx]
        if len(smiles) > args.max_benchmark:
            rng = np.random.default_rng(0)
            smiles = [smiles[i] for i in
                      sorted(rng.choice(len(smiles), args.max_benchmark, replace=False))]

        print(f"[{name}] comparing {len(smiles)} molecules against the corpus")
        report = check_leakage(smiles, args.corpus, max_corpus=limit, progress_every=40)

        summary = report.summary()
        summary["dataset"] = name
        summary["split"] = "test" if args.test_only else "all"
        rows.append(summary)
        for value in report.max_similarity:
            distributions.append({"dataset": name, "max_similarity": float(value)})

        print(f"  exact overlap   : {report.n_exact_overlap} "
              f"({100 * report.fraction_exact_overlap:.2f}%)")
        print(f"  median max sim  : {report.median_max_similarity:.3f}")
        print(f"  fraction >= 0.9 : {100 * report.fraction_above_0_9:.2f}%")
        print(f"  fraction >= 0.99: {100 * report.fraction_above_0_99:.2f}%\n")

    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "leakage_summary.csv"), index=False)
    pd.DataFrame(distributions).to_csv(
        os.path.join(args.output_dir, "leakage_similarity_distribution.csv"), index=False)
    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"corpus": args.corpus, "limit": args.limit,
                   "test_only": args.test_only,
                   "max_benchmark": args.max_benchmark}, h, indent=2)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
