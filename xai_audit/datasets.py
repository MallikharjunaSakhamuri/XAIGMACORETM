"""Benchmark loading, tolerant of local MoleculeNet layout differences.

The MolCLR distribution of these benchmarks renames some columns and files
relative to the DeepChem originals (BACE ships with a `smiles` column rather
than `mol`, for instance). File, SMILES column and target column are therefore
resolved by search over known aliases, and the resolution actually used is
reported so it can be recorded in the manuscript.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .motifs import parse_smiles

SMILES_ALIASES = ["smiles", "mol", "SMILES", "smi", "canonical_smiles", "Smiles"]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    filename: str
    smiles_column: str
    task_type: str
    target_columns: List[str]
    filename_aliases: List[str] = field(default_factory=list)
    target_keywords: List[str] = field(default_factory=list)
    subdir: Optional[str] = None

    @property
    def directory(self) -> str:
        return self.subdir or self.name


# QM9 is multi-target in gmacore.data.moleculenet; all six are profiled.
QM9_TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "cv"]

BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "bace": BenchmarkSpec(
        "bace", "bace.csv", "mol", "classification", ["Class"],
        target_keywords=["class"],
    ),
    "bbbp": BenchmarkSpec(
        "bbbp", "BBBP.csv", "smiles", "classification", ["p_np"],
        filename_aliases=["bbbp.csv"], target_keywords=["p_np"],
    ),
    "clintox": BenchmarkSpec(
        "clintox", "clintox.csv", "smiles", "classification", ["CT_TOX"],
        filename_aliases=["clintox.csv.gz"], target_keywords=["ct_tox"],
    ),
    "esol": BenchmarkSpec(
        "esol", "delaney-processed.csv", "smiles", "regression",
        ["measured log solubility in mols per litre"],
        filename_aliases=["esol.csv", "delaney.csv", "ESOL.csv"],
        target_keywords=["measured log solubility", "logsolubility", "measured_log"],
    ),
    "qm9": BenchmarkSpec(
        "qm9", "qm9.csv", "smiles", "regression", QM9_TARGETS,
        target_keywords=QM9_TARGETS,
    ),
}


def resolve_file(spec: BenchmarkSpec, data_root: str) -> str:
    """Find the benchmark file, falling back to a search of its directory."""
    directory = os.path.join(data_root, spec.directory)
    for candidate in [spec.filename, *spec.filename_aliases]:
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"{spec.name}: no directory at {directory}")

    found = sorted(
        glob.glob(os.path.join(directory, "*.csv"))
        + glob.glob(os.path.join(directory, "*.csv.gz"))
    )
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        preferred = [p for p in found if spec.name in os.path.basename(p).lower()]
        if len(preferred) == 1:
            return preferred[0]
        raise FileNotFoundError(
            f"{spec.name}: several candidate files in {directory}: "
            f"{[os.path.basename(p) for p in found]}. Pass path_override."
        )
    raise FileNotFoundError(f"{spec.name}: no CSV found in {directory}")


def resolve_column(frame: pd.DataFrame, preferred: str, aliases, keywords) -> str:
    """Resolve a column by exact name, then alias, then keyword substring."""
    if preferred in frame.columns:
        return preferred
    for alias in aliases:
        if alias in frame.columns:
            return alias
    lowered = {c.lower(): c for c in frame.columns}
    for keyword in keywords:
        key = keyword.lower()
        if key in lowered:
            return lowered[key]
        for low, original in lowered.items():
            if key in low:
                return original
    raise KeyError(
        f"could not resolve a column for '{preferred}'. "
        f"Available: {list(frame.columns)[:15]}"
    )


def load_benchmark(
    name: str,
    data_root: str,
    target_column: Optional[str] = None,
    path_override: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load one benchmark as a frame with columns: smiles, target, mol."""
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark '{name}'; expected one of {list(BENCHMARKS)}")
    spec = BENCHMARKS[name]

    path = path_override or resolve_file(spec, data_root)
    frame = pd.read_csv(path)

    smiles_col = resolve_column(frame, spec.smiles_column, SMILES_ALIASES, [])
    target_col = resolve_column(
        frame,
        target_column or spec.target_columns[0],
        spec.target_columns,
        spec.target_keywords,
    )

    out = frame[[smiles_col, target_col]].rename(
        columns={smiles_col: "smiles", target_col: "target"}
    )
    out = out.dropna(subset=["smiles", "target"]).reset_index(drop=True)

    out["mol"] = [parse_smiles(s) for s in out["smiles"]]
    n_before = len(out)
    out = out[out["mol"].notna()].reset_index(drop=True)

    out.attrs.update({
        "name": name,
        "task_type": spec.task_type,
        "target_column": target_col,
        "smiles_column": smiles_col,
        "source_file": path,
        "n_unparsed": n_before - len(out),
    })
    if verbose:
        print(
            f"[{name}] file={os.path.basename(path)} smiles='{smiles_col}' "
            f"target='{target_col}' n={len(out)} unparsed={out.attrs['n_unparsed']}"
        )
    return out


def target_columns_for(name: str) -> List[str]:
    """Targets to profile for a benchmark. QM9 is multi-target."""
    return list(BENCHMARKS[name].target_columns)
