"""Stage 4: KernelSHAP attributions for the fine-tuned downstream models.

Measures 1 and 2 of the audit:

  1. motif concentration   share of total absolute attribution held by the
                           top-k motifs
  2. explanation stability attribution coherence on structurally novel
                           molecules against training-similar ones

Two settings differ deliberately from the original notebooks and must be
reported in the manuscript:

  l1_reg   The notebooks used "num_features(10)", which keeps ten non-zero
           attributions and forces every other one to exactly zero by AIC
           feature selection. On a 30-atom molecule that is roughly 97 percent
           of the attributions zeroed by the explainer rather than the model,
           which directly distorts a concentration measure. Here it is 0.0.
  nsamples The notebooks used 200 for a feature space of several hundred.
           Here it follows the usual 2M + 2048 convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Batch, Data

from .attribution import (
    N_NODE_FEATURES,
    aggregate_to_motifs,
    aggregate_to_nodes,
    explanation_stability,
    motif_importance,
    top_k_share,
)
from .motifs import MotifLibrary, parse_smiles

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class BatchedGraphWrapper:
    """Expose a fine-tuned model as f(X) -> scores over node features.

    The molecule's topology is held fixed while its node features vary, which
    is what makes a message-passing model explainable by a tabular method.
    Perturbed rows are evaluated as one PyG batch rather than one at a time,
    which is the difference between minutes and hours per molecule.
    """

    def __init__(self, model, graph: Data, stats, device: str = "cpu",
                 batch_size: int = 256):
        self.model = model.eval()
        self.graph = graph
        self.device = device
        self.batch_size = batch_size
        self.n_nodes = int(graph.x_cat.shape[0])
        self.n_features = N_NODE_FEATURES
        self._mean, self._std, self._clamp = stats

    def _make(self, row: np.ndarray) -> Data:
        values = torch.as_tensor(
            np.asarray(row, dtype=np.float32).reshape(self.n_nodes, self.n_features)
        )
        data = Data(
            x_cat=values[:, :2].round().long(),
            x_phys=values[:, 2:].float(),
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,
        )
        data.num_nodes = self.n_nodes
        return data

    @torch.no_grad()
    def __call__(self, features: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(features, dtype=np.float32))
        out: List[float] = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            batch = Batch.from_data_list([self._make(r) for r in chunk]).to(self.device)
            logits = self.model(batch).reshape(len(chunk), -1)[:, 0]
            out.extend(logits.detach().cpu().numpy().tolist())
        return np.asarray(out, dtype=float)

    def flatten(self) -> np.ndarray:
        stacked = torch.cat(
            [self.graph.x_cat.float(), self.graph.x_phys.float()], dim=-1
        )
        return stacked.detach().cpu().numpy().reshape(-1)


def masking_background(wrapper: BatchedGraphWrapper, feature_mean: torch.Tensor,
                       n_samples: int = 24, seed: int = 0) -> np.ndarray:
    """Background rows that replace node features with the corpus mean.

    This matches how both pretraining notebooks mask a node: the neutral value
    is the corpus mean, which standardises to zero, rather than raw zero. A
    background built from raw zeros would place the reference far outside the
    data and inflate every attribution.
    """
    rng = np.random.default_rng(seed)
    base = wrapper.flatten().reshape(wrapper.n_nodes, wrapper.n_features)
    neutral = feature_mean.detach().cpu().numpy().reshape(1, -1)
    rows = []
    for _ in range(n_samples):
        keep = (rng.random((wrapper.n_nodes, 1)) >= 0.5).astype(float)
        rows.append((keep * base + (1 - keep) * neutral).reshape(-1))
    rows.append(np.repeat(neutral, wrapper.n_nodes, axis=0).reshape(-1))
    return np.asarray(rows, dtype=float)


def max_similarity_to_reference(smiles: Sequence[str],
                                reference: Sequence[str]) -> np.ndarray:
    """s(x): maximum Tanimoto similarity of each molecule to a reference set."""
    ref_fps = []
    for value in reference:
        mol = parse_smiles(value)
        if mol is not None:
            ref_fps.append(_FPGEN.GetFingerprint(mol))
    out = np.zeros(len(smiles), dtype=float)
    for index, value in enumerate(smiles):
        mol = parse_smiles(value)
        if mol is None or not ref_fps:
            out[index] = np.nan
            continue
        sims = DataStructs.BulkTanimotoSimilarity(_FPGEN.GetFingerprint(mol), ref_fps)
        out[index] = float(max(sims)) if sims else np.nan
    return out


@dataclass
class AttributionResult:
    dataset: str
    model: str
    n_molecules: int
    top3_share: float
    top5_share: float
    motif_importance: Dict[str, float]
    stability: Dict[str, float]
    nsamples: int
    l1_reg: float


def attribute_molecules(
    wrapper_factory,
    smiles: Sequence[str],
    graphs: Sequence[Data],
    groups: List[str],
    nsamples: Optional[int] = None,
    l1_reg: float = 0.0,
    seed: int = 0,
    progress_every: int = 25,
):
    """KernelSHAP over a sample, aggregated to nodes and then motifs."""
    import shap

    library = MotifLibrary.build(groups)
    per_molecule: List[Dict[str, float]] = []
    feature_vectors: List[np.ndarray] = []

    for index, (value, graph) in enumerate(zip(smiles, graphs)):
        mol = parse_smiles(value)
        if mol is None or graph is None:
            continue
        wrapper, background = wrapper_factory(graph)
        instance = wrapper.flatten()

        n_features = instance.size
        draws = nsamples if nsamples is not None else 2 * n_features + 2048

        np.random.seed(seed + index)
        explainer = shap.KernelExplainer(wrapper, background, link="identity")
        values = explainer.shap_values(
            instance, nsamples=draws, l1_reg=l1_reg, silent=True
        )
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values).reshape(-1)

        node_scores = aggregate_to_nodes(values, wrapper.n_nodes)
        atom_sets = {g: library.atom_set(mol, g) for g in groups}
        per_molecule.append(aggregate_to_motifs(node_scores, atom_sets))

        # Per-feature-channel mean, used for the stability comparison.
        feature_vectors.append(
            values.reshape(wrapper.n_nodes, N_NODE_FEATURES).mean(axis=0)
        )

        if progress_every and (index + 1) % progress_every == 0:
            print(f"    {index + 1}/{len(smiles)} molecules", flush=True)

    return per_molecule, np.asarray(feature_vectors)


def summarise(
    dataset: str,
    model: str,
    per_molecule: List[Dict[str, float]],
    features: np.ndarray,
    novel_mask: np.ndarray,
    similar_mask: np.ndarray,
    nsamples: int,
    l1_reg: float,
) -> AttributionResult:
    importance = motif_importance(per_molecule)
    stability = explanation_stability(features[novel_mask], features[similar_mask])
    return AttributionResult(
        dataset=dataset, model=model, n_molecules=len(per_molecule),
        top3_share=top_k_share(importance, 3), top5_share=top_k_share(importance, 5),
        motif_importance=importance, stability=stability.as_dict(),
        nsamples=nsamples, l1_reg=l1_reg,
    )
