"""Torch-side adapter: load a checkpoint and expose it as a plain callable.

`attribution.py` deliberately knows nothing about torch. This module is the
only place that does. It rebuilds the encoder described by `run_config.json`,
loads the saved weights, and wraps the resulting predictor as a function from
a flattened node-feature matrix to a scalar output, which is the interface
KernelSHAP requires.

The node feature layout is fixed by gmacore.data.featurizer: two categorical
channels then eight physicochemical channels per atom, giving node_dim 10,
matching `run_config.json`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .attribution import N_NODE_FEATURES


@dataclass
class RunConfig:
    """The subset of run_config.json needed to rebuild an encoder."""

    variant: str
    node_dim: int
    edge_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    dropout: float
    raw: Dict

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            variant=raw["variant"],
            node_dim=raw["node_dim"],
            edge_dim=raw["edge_dim"],
            hidden_dim=raw["hidden_dim"],
            output_dim=raw["output_dim"],
            num_layers=raw["num_layers"],
            dropout=raw.get("dropout", 0.0),
            raw=raw,
        )

    def check_feature_layout(self) -> None:
        if self.node_dim != N_NODE_FEATURES:
            raise ValueError(
                f"run_config node_dim is {self.node_dim} but the attribution "
                f"code assumes {N_NODE_FEATURES} node features. The motif "
                "aggregation would be misaligned; reconcile before running."
            )


def load_state_dict(checkpoint_path: str, map_location: str = "cpu") -> Dict:
    """Return the model state dict from a gmacore checkpoint.

    `gmacore.utils.checkpoint.save_checkpoint` stores it under
    'model_state_dict'. A bare state dict is accepted as well.
    """
    import torch

    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def strip_prefix(state: Dict, prefix: str) -> Dict:
    """Select the submodule weights saved under `prefix` and drop the prefix."""
    selected = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise KeyError(
            f"no weights found under prefix '{prefix}'. "
            f"Available top-level prefixes: "
            f"{sorted({k.split('.')[0] for k in state})}"
        )
    return selected


class GraphPredictorWrapper:
    """Expose a graph predictor as f(X) -> scores for KernelSHAP.

    The graph topology is held fixed while node features vary, which is what
    makes a graph model explainable by a tabular attribution method: each
    perturbed row is the same molecule with altered node features.

    `head` maps a graph embedding to a scalar. For a classification benchmark
    pass the fine-tuned prediction head; for an embedding-only analysis pass a
    reduction such as a fixed projection, and say so in the manuscript.
    """

    def __init__(self, encoder, graph, device: str = "cpu", head: Optional[Callable] = None):
        import torch

        self.torch = torch
        self.encoder = encoder.eval().to(device)
        self.graph = graph
        self.device = device
        self.head = head
        self.n_nodes = int(graph.x_cat.shape[0])
        self.n_features = N_NODE_FEATURES

    def _rebuild(self, flat_row: np.ndarray):
        from torch_geometric.data import Data

        torch = self.torch
        values = torch.as_tensor(
            np.asarray(flat_row, dtype=np.float32).reshape(self.n_nodes, self.n_features)
        )
        x_cat = values[:, :2].round().long()
        x_phys = values[:, 2:].float()
        data = Data(
            x_cat=x_cat,
            x_phys=x_phys,
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,
        )
        data.num_nodes = self.n_nodes
        data.batch = torch.zeros(self.n_nodes, dtype=torch.long)
        return data.to(self.device)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        torch = self.torch
        rows = np.atleast_2d(np.asarray(features, dtype=np.float32))
        outputs = []
        with torch.no_grad():
            for row in rows:
                data = self._rebuild(row)
                embedding = self.encoder(data)
                if isinstance(embedding, (tuple, list)):
                    embedding = embedding[0]
                value = self.head(embedding) if self.head is not None else embedding
                outputs.append(float(torch.as_tensor(value).reshape(-1)[0].cpu()))
        return np.asarray(outputs, dtype=float)

    def flatten(self, graph=None) -> np.ndarray:
        """Flatten a graph's node features into the vector KernelSHAP sees."""
        torch = self.torch
        graph = graph if graph is not None else self.graph
        stacked = torch.cat(
            [graph.x_cat.float(), graph.x_phys.float()], dim=-1
        )
        return stacked.detach().cpu().numpy().reshape(-1)


def background_matrix(wrapper: GraphPredictorWrapper, n_samples: int, seed: int = 0) -> np.ndarray:
    """Background distribution for KernelSHAP, built from the molecule itself.

    Each background row perturbs the physicochemical channels toward their
    column mean, which gives a reference where a feature carries no
    distinguishing information, while leaving the categorical channels intact.
    Report whatever background is used, since SHAP values are defined relative
    to it.
    """
    rng = np.random.default_rng(seed)
    base = wrapper.flatten().reshape(wrapper.n_nodes, wrapper.n_features)
    mean = base.mean(axis=0, keepdims=True)
    rows = []
    for _ in range(n_samples):
        weight = rng.random((wrapper.n_nodes, 1))
        row = base * (1.0 - weight) + mean * weight
        row[:, :2] = base[:, :2]
        rows.append(row.reshape(-1))
    return np.asarray(rows, dtype=float)
