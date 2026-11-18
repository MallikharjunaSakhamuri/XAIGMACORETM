"""Shared setup for the figure stages.

Stages 8 to 11 all need the same preamble: load a benchmark, featurize it,
fine-tune a probe on one checkpoint, then recover the split, the test-set
similarity to training, and a SHAP wrapper factory. Keeping that here means
the figure scripts differ only in what they draw, and that every figure is
computed on the same molecules under the same protocol as Table B.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .attribution import (aggregate_to_motifs, aggregate_to_nodes,
                          kernel_shap_values, novelty_partition)
from .datasets import load_benchmark
from .finetune import FineTuneConfig, train_downstream_model
from .gmacore_model import featurize
from .motifs import DATASET_GROUPS, MotifLibrary, parse_smiles
from .rules import CANONICAL_RULES
from .shap_runner import (BatchedGraphWrapper, masking_background,
                          max_similarity_to_reference)

MODELS = {"ManualAug-CL": "manualaug_cl_fixed", "GMACORE-TM": "gmacore_tm_fixedgan"}


@dataclass
class Session:
    dataset: str
    model_label: str
    task: str
    model: object
    stats: object
    graphs: List
    train_idx: Sequence[int]
    test_idx: Sequence[int]
    test_smiles: List[str]
    similarity: np.ndarray
    novel: np.ndarray
    similar: np.ndarray
    valid_score: float
    device: str
    groups: List[str]

    def wrapper(self, graph):
        return BatchedGraphWrapper(self.model, graph, self.stats, device=self.device)

    def background(self, wrapper, n_samples: int = 24, seed: int = 0):
        return masking_background(wrapper, self.stats[0], n_samples=n_samples, seed=seed)

    def shap_for(self, graph, nsamples=None, l1_reg: float = 0.0, seed: int = 0):
        """Flat feature attributions and their per-node sums for one molecule."""
        wrapper = self.wrapper(graph)
        background = self.background(wrapper, seed=seed)
        values = kernel_shap_values(
            wrapper, background, wrapper.flatten(),
            nsamples=nsamples or "auto", l1_reg=l1_reg, seed=seed,
        )
        nodes = aggregate_to_nodes(values, wrapper.n_nodes)
        return values, nodes, wrapper

    def motif_attributions(self, smiles: str, node_scores: np.ndarray,
                           library: MotifLibrary) -> Dict[str, float]:
        mol = parse_smiles(smiles)
        if mol is None:
            return {}
        atom_sets = {m: library.atom_set(mol, m) for m in self.groups}
        return aggregate_to_motifs(node_scores, atom_sets)

    def rule_group(self, smiles_list: Sequence[str]) -> np.ndarray:
        """G(x): 1 for non-compliant, 0 for compliant. NaN where unparseable."""
        rule = CANONICAL_RULES.get(self.dataset)
        out = np.full(len(smiles_list), np.nan)
        if rule is None:
            return out
        for index, value in enumerate(smiles_list):
            mol = parse_smiles(value)
            if mol is not None:
                out[index] = float(rule.group_label(mol))
        return out


def build_session(
    dataset: str,
    model_label: str,
    data_root: str,
    runs_root: str,
    qm9_target: str = "gap",
    seed: int = 0,
    epochs: int = 100,
    qm9_epochs: int = 50,
    max_molecules: Optional[int] = None,
    qm9_max_molecules: Optional[int] = 25000,
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> Session:
    import os

    folder = MODELS[model_label]
    checkpoint = os.path.join(runs_root, folder, "best_model.pt")
    stats_path = os.path.join(runs_root, folder, "feature_stats.pt")
    for path in (checkpoint, stats_path):
        if not os.path.exists(path):
            raise SystemExit(f"{model_label}: missing {path}")

    target = qm9_target if dataset == "qm9" else None
    frame = load_benchmark(dataset, data_root, target_column=target)

    cap = max_molecules
    if dataset == "qm9" and qm9_max_molecules:
        cap = min(cap, qm9_max_molecules) if cap else qm9_max_molecules
    if cap and len(frame) > cap:
        frame = frame.sample(cap, random_state=0).reset_index(drop=True)

    task = frame.attrs["task_type"]
    config = FineTuneConfig(epochs=qm9_epochs if dataset == "qm9" else epochs)

    graphs = [g for g in (featurize(s, [float(v)])
                          for s, v in zip(frame["smiles"], frame["target"])) if g]
    print(f"  [{dataset}/{model_label}] featurized {len(graphs)}, fine-tuning")

    model, stats, splits, valid_score = train_downstream_model(
        graphs, task, checkpoint, stats_path, seed=seed, config=config, num_tasks=1,
    )
    train_idx, _, test_idx = splits
    print(f"    validation score {valid_score:.4f}")

    train_smiles = [graphs[i].smiles for i in train_idx]
    test_smiles = [graphs[i].smiles for i in test_idx]
    similarity = max_similarity_to_reference(test_smiles, train_smiles)
    novel, similar = novelty_partition(similarity, q_low, q_high)

    return Session(
        dataset=dataset, model_label=model_label, task=task, model=model, stats=stats,
        graphs=graphs, train_idx=train_idx, test_idx=test_idx, test_smiles=test_smiles,
        similarity=similarity, novel=novel, similar=similar, valid_score=valid_score,
        device=config.device, groups=list(DATASET_GROUPS[dataset]),
    )


def sample_test(session: Session, n: int, seed: int = 0):
    """A reproducible subsample of the test set, as indices into test_idx."""
    rng = np.random.default_rng(seed)
    take = min(n, len(session.test_idx))
    return np.sort(rng.choice(len(session.test_idx), size=take, replace=False))
