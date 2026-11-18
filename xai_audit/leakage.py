"""Leakage checks between the pretraining corpus and held-out benchmark sets.

Reviewer 1 point 5 asks whether molecules seen during generative pretraining
resemble benchmark test molecules closely enough to inflate the stability
results. Fingerprint similarity computed within a benchmark cannot answer
that, because it never looks at the pretraining corpus. These functions
compare the two sets directly.

Two quantities are reported:
  * exact overlap, on canonical SMILES, which is cheap and settles the
    strongest form of the concern;
  * nearest-neighbour Tanimoto similarity of each benchmark molecule against
    the whole corpus, which quantifies near-duplication.

The corpus is streamed in chunks, so a multi-million line SMILES file does not
need to be held in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ECFP4 as specified in Appendix Table A.4: Morgan, radius 2, 2048 bits.
_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(mol: Chem.Mol):
    return _GENERATOR.GetFingerprint(mol)


def canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def stream_corpus(path: str, chunk_size: int = 50_000) -> Iterator[List[str]]:
    """Yield lists of SMILES from a one-per-line corpus file."""
    chunk: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            smiles = line.strip().split()[0] if line.strip() else ""
            if smiles:
                chunk.append(smiles)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


@dataclass
class LeakageReport:
    n_benchmark: int
    n_corpus_scanned: int
    n_exact_overlap: int
    fraction_exact_overlap: float
    max_similarity: np.ndarray
    fraction_above_0_9: float
    fraction_above_0_99: float
    median_max_similarity: float

    def summary(self) -> Dict[str, float]:
        return {
            "n_benchmark": self.n_benchmark,
            "n_corpus_scanned": self.n_corpus_scanned,
            "n_exact_overlap": self.n_exact_overlap,
            "fraction_exact_overlap": self.fraction_exact_overlap,
            "median_max_similarity": self.median_max_similarity,
            "fraction_above_0.9": self.fraction_above_0_9,
            "fraction_above_0.99": self.fraction_above_0_99,
        }


def check_leakage(
    benchmark_smiles: Sequence[str],
    corpus_path: str,
    chunk_size: int = 50_000,
    max_corpus: Optional[int] = None,
    progress_every: int = 20,
) -> LeakageReport:
    """Compare benchmark molecules against the pretraining corpus.

    `max_corpus` caps the number of corpus molecules scanned, which is useful
    for a quick pass before committing to the full file. Report the cap in the
    manuscript whenever it is used, since a partial scan gives a lower bound on
    overlap rather than a complete answer.
    """
    bench_mols, bench_canon, kept = [], [], []
    for index, smiles in enumerate(benchmark_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bench_mols.append(mol)
        bench_canon.append(Chem.MolToSmiles(mol))
        kept.append(index)

    bench_fps = [fingerprint(mol) for mol in bench_mols]
    canon_set = set(bench_canon)
    best = np.zeros(len(bench_fps), dtype=float)
    exact_hits: set = set()
    scanned = 0

    for chunk_index, chunk in enumerate(stream_corpus(corpus_path, chunk_size)):
        for smiles in chunk:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            scanned += 1
            canon_smiles = Chem.MolToSmiles(mol)
            if canon_smiles in canon_set:
                exact_hits.add(canon_smiles)
            corpus_fp = fingerprint(mol)
            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(corpus_fp, bench_fps), dtype=float
            )
            np.maximum(best, sims, out=best)
            if max_corpus is not None and scanned >= max_corpus:
                break
        if progress_every and (chunk_index + 1) % progress_every == 0:
            print(f"  scanned {scanned:,} corpus molecules", flush=True)
        if max_corpus is not None and scanned >= max_corpus:
            break

    return LeakageReport(
        n_benchmark=len(bench_fps),
        n_corpus_scanned=scanned,
        n_exact_overlap=len(exact_hits),
        fraction_exact_overlap=len(exact_hits) / max(1, len(bench_fps)),
        max_similarity=best,
        fraction_above_0_9=float((best >= 0.9).mean()) if best.size else float("nan"),
        fraction_above_0_99=float((best >= 0.99).mean()) if best.size else float("nan"),
        median_max_similarity=float(np.median(best)) if best.size else float("nan"),
    )
