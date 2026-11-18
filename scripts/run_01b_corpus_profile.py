r"""Stage 1b: motif and property profile of the PubChem pretraining corpus (E1).

    python scripts\run_01b_corpus_profile.py ^
        --data-root "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data" ^
        --corpus "D:\GMACORE_RIE_Submission\Code\MolCLR_data\data\pubchem-10m-clean.txt" ^
        --output-dir results\corpus_profile

Table A characterises the benchmark data. The encoders, however, were
pretrained on PubChem, so the composition of that corpus is what the
representation was shaped by. This stage profiles it on the same motif
library and the same rule sets, and then contrasts it against each benchmark.

Only the first `--limit` unique molecules are profiled, because that is what
the pretraining consumed: step 1 read the corpus with LIMIT = 100_000 after
de-duplication, and both run_config.json files record num_graphs = 100000.

Nothing here is an XAI measurement. These are data-level characteristics, and
they are reported as such: prevalence, property distributions and rule
compliance. Attribution concentration and explanation stability are
model-dependent and belong to stages 3 and 4.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xai_audit.datasets import load_benchmark
from xai_audit.leakage import canonical, stream_corpus
from xai_audit.motifs import DATASET_GROUPS, MOTIF_SMARTS, MotifLibrary, parse_smiles
from xai_audit.rules import CANONICAL_RULES, molecule_properties

PROPERTIES = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "HeavyAtoms"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motif and property profile of the pretraining corpus.")
    parser.add_argument("--corpus", required=True,
                        help="The pubchem SMILES file used for pretraining.")
    parser.add_argument("--data-root", required=True,
                        help="Benchmark root, for the corpus-versus-benchmark contrast.")
    parser.add_argument("--datasets", nargs="*",
                        default=["bace", "esol", "clintox", "qm9"])
    parser.add_argument("--output-dir", default="results/corpus_profile")
    parser.add_argument("--limit", type=int, default=100_000,
                        help="Unique corpus molecules to profile. 0 profiles the whole file.")
    parser.add_argument("--qm9-target", default="gap")
    parser.add_argument("--no-properties", dest="properties", action="store_false",
                        default=True,
                        help="Skip the descriptor block and rule compliance, which "
                             "is the slow part, and report motif prevalence only.")
    parser.add_argument("--progress-every", type=int, default=20_000)
    return parser.parse_args()


def profile_corpus(args: argparse.Namespace, library: MotifLibrary):
    """Stream the corpus once, accumulating motif, property and rule counts."""
    limit = args.limit if args.limit > 0 else None
    rules = {name: CANONICAL_RULES[name] for name in args.datasets
             if name in CANONICAL_RULES}

    motif_counts = {name: 0 for name in library.names}
    rule_non_compliant = {name: 0 for name in rules}
    property_values = {key: [] for key in PROPERTIES}

    seen = set()
    kept = 0
    unparsed = 0

    for chunk in stream_corpus(args.corpus):
        for smiles in chunk:
            key = canonical(smiles)
            if key is None:
                unparsed += 1
                continue
            if key in seen:
                continue
            seen.add(key)

            mol = parse_smiles(key)
            if mol is None:
                unparsed += 1
                continue

            for name, present in library.membership(mol).items():
                motif_counts[name] += present

            if args.properties:
                props = molecule_properties(mol)
                for prop in PROPERTIES:
                    property_values[prop].append(props[prop])
                for name, rule in rules.items():
                    if not rule.is_compliant_from_properties(props, mol):
                        rule_non_compliant[name] += 1

            kept += 1
            if args.progress_every and kept % args.progress_every == 0:
                print(f"  profiled {kept:,} unique molecules", flush=True)
            if limit is not None and kept >= limit:
                break
        if limit is not None and kept >= limit:
            break

    return motif_counts, rule_non_compliant, property_values, kept, unparsed


def benchmark_motif_prevalence(args: argparse.Namespace, library: MotifLibrary):
    """Motif prevalence within each benchmark, and its rule non-compliance rate."""
    prevalence, compliance, sizes = {}, {}, {}
    for name in args.datasets:
        target = args.qm9_target if name == "qm9" else None
        try:
            frame = load_benchmark(name, args.data_root, target_column=target)
        except Exception as exc:
            print(f"[skip] {name}: {exc}")
            continue

        counts = {motif: 0 for motif in library.names}
        non_compliant = 0
        n = 0
        rule = CANONICAL_RULES.get(name)
        for smiles in frame["smiles"].astype(str):
            mol = parse_smiles(smiles)
            if mol is None:
                continue
            for motif, present in library.membership(mol).items():
                counts[motif] += present
            if args.properties and rule is not None:
                if not rule.is_compliant(mol):
                    non_compliant += 1
            n += 1

        if n == 0:
            continue
        prevalence[name] = {motif: counts[motif] / n for motif in counts}
        compliance[name] = non_compliant / n if (args.properties and rule) else float("nan")
        sizes[name] = n
        print(f"[{name}] profiled {n} benchmark molecules")
    return prevalence, compliance, sizes


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.corpus):
        raise SystemExit(f"corpus not found: {args.corpus}")

    library = MotifLibrary.build(list(MOTIF_SMARTS))

    limit = args.limit if args.limit > 0 else None
    print(f"corpus: {args.corpus}")
    print(f"profiling {'all' if limit is None else f'{limit:,}'} unique molecules "
          f"on {len(library.names)} motifs\n")

    motif_counts, rule_non_compliant, property_values, n_corpus, unparsed = \
        profile_corpus(args, library)
    print(f"\ncorpus: {n_corpus:,} unique molecules profiled, {unparsed} unparsed\n")

    corpus_fraction = {m: motif_counts[m] / n_corpus for m in motif_counts}
    pd.DataFrame([
        {"motif": m, "n_present": motif_counts[m],
         "n_corpus": n_corpus, "fraction_present": corpus_fraction[m]}
        for m in library.names
    ]).sort_values("fraction_present", ascending=False).to_csv(
        os.path.join(args.output_dir, "corpus_motif_profile.csv"), index=False)

    if args.properties:
        pd.DataFrame([
            {"property": prop,
             "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
             "p10": float(np.quantile(values, 0.10)),
             "p50": float(np.quantile(values, 0.50)),
             "p90": float(np.quantile(values, 0.90))}
            for prop, values in property_values.items() if values
        ]).to_csv(os.path.join(args.output_dir, "corpus_property_summary.csv"),
                  index=False)

    prevalence, compliance, sizes = benchmark_motif_prevalence(args, library)

    contrast_rows = []
    for dataset, fractions in prevalence.items():
        in_group = set(DATASET_GROUPS.get(dataset, []))
        for motif in library.names:
            bench = fractions[motif]
            corp = corpus_fraction[motif]
            contrast_rows.append({
                "dataset": dataset,
                "motif": motif,
                "reported_group": motif in in_group,
                "n_benchmark": sizes[dataset],
                "corpus_fraction": corp,
                "benchmark_fraction": bench,
                "difference": bench - corp,
                "ratio": (bench / corp) if corp > 0 else float("nan"),
            })
    if contrast_rows:
        pd.DataFrame(contrast_rows).to_csv(
            os.path.join(args.output_dir, "corpus_vs_benchmark_motifs.csv"), index=False)

    if args.properties:
        pd.DataFrame([
            {"dataset": dataset,
             "rule_name": CANONICAL_RULES[dataset].name,
             "corpus_fraction_non_compliant": rule_non_compliant[dataset] / n_corpus,
             "benchmark_fraction_non_compliant": compliance.get(dataset, float("nan"))}
            for dataset in rule_non_compliant
        ]).to_csv(os.path.join(args.output_dir, "corpus_rule_compliance.csv"),
                  index=False)

    with open(os.path.join(args.output_dir, "settings.json"), "w", encoding="utf-8") as h:
        json.dump({"corpus": args.corpus, "limit": args.limit,
                   "n_corpus_profiled": n_corpus, "unparsed": unparsed,
                   "datasets": args.datasets, "properties": args.properties,
                   "n_motifs": len(library.names)}, h, indent=2)

    top = sorted(library.names, key=lambda m: corpus_fraction[m], reverse=True)[:8]
    print("\nMost prevalent motifs in the pretraining corpus:")
    for motif in top:
        print(f"  {motif:<16} {100 * corpus_fraction[motif]:6.2f}%")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
