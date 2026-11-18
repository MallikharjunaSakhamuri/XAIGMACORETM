"""Stage 5: the GAN generator's gating behaviour, by motif.

Reconstructed from 06_pretrain_gmacore_tm_fixedgan_runpod. The generator does
not synthesise new molecules: it scores each node and edge, drops the
lowest-ranked fraction, and attenuates the rest. So the question worth asking
of it is which substructures it preferentially keeps or drops, which is a
property of the scores rather than of any generated structure.

Two quantities are reported per motif:
  gate score  mean sigmoid score the generator assigns to the motif's atoms
  drop rate   fraction of those atoms removed under the rank mask at
              keep_ratio, the same rule used during pretraining
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .gmacore_model import NODE_DIM, NUM_EDGE_FEATURES, _mlp
from .motifs import MotifLibrary, parse_smiles


class GraphGenerator(nn.Module):
    """Node and edge scorer, matching the pretraining notebook exactly."""

    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = NUM_EDGE_FEATURES,
                 hidden_dim: int = 256, noise_dim: int = 32, num_layers: int = 3,
                 keep_ratio: float = 0.75) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.keep_ratio = keep_ratio
        self.node_encoder = _mlp(node_dim, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(edge_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + noise_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _noise(self, data, device):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x_cat.size(0), dtype=torch.long, device=device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        return torch.randn(num_graphs, self.noise_dim, device=device)[batch]

    def forward(self, data, feature_mean, feature_std, clamp):
        device = data.x_cat.device
        x = torch.cat([data.x_cat.float(), data.x_phys.float()], dim=-1)
        x = torch.clamp((x - feature_mean) / feature_std, -clamp, clamp)
        x = self.node_encoder(x)
        _ = self.edge_encoder(data.edge_attr.float())

        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)

        noise = self._noise(data, device)
        node_scores = torch.sigmoid(self.node_scorer(torch.cat([x, noise], dim=-1)))
        source, target = data.edge_index[0], data.edge_index[1]
        edge_scores = torch.sigmoid(
            self.edge_scorer(torch.cat([x[source], x[target], noise[source]], dim=-1))
        )
        return node_scores, edge_scores

    @staticmethod
    def rank_mask(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Drop the lowest-ranked fraction, as the pretraining run did."""
        flat = scores.reshape(-1)
        num_drop = int(round((1.0 - keep_ratio) * flat.numel()))
        hard = torch.ones_like(flat)
        if num_drop > 0:
            lowest = torch.topk(flat.detach(), num_drop, largest=False).indices
            hard[lowest] = 0.0
        return hard.view_as(scores)


def load_generator(checkpoint_path: str, device: str = "cpu"):
    """Rebuild the generator from a GMACORE-TM checkpoint, loading strictly."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "generator_state_dict" not in payload:
        raise KeyError(
            "this checkpoint has no generator_state_dict; it is not a "
            "generatively augmented variant"
        )
    config = payload.get("config", {})
    generator = GraphGenerator(
        node_dim=config.get("node_dim", NODE_DIM),
        edge_dim=config.get("edge_dim", NUM_EDGE_FEATURES),
        hidden_dim=config.get("generator_hidden_dim", 256),
        noise_dim=config.get("noise_dim", 32),
        keep_ratio=config.get("keep_ratio", 0.75),
    )
    generator.load_state_dict(payload["generator_state_dict"], strict=True)
    return generator.eval().to(device), payload


@torch.no_grad()
def motif_gate_profile(
    generator: GraphGenerator,
    graphs: Sequence,
    smiles: Sequence[str],
    groups: List[str],
    stats,
    device: str = "cpu",
    repeats: int = 8,
    seed: int = 0,
) -> "object":
    """Mean gate score and drop rate per motif.

    The generator is stochastic: its scores depend on a noise vector sampled
    per graph. Each molecule is therefore scored `repeats` times and the
    results averaged, so a motif's profile reflects the generator rather than
    one noise draw.
    """
    import pandas as pd

    mean, std, clamp = stats
    library = MotifLibrary.build(groups)
    torch.manual_seed(seed)

    records: Dict[str, Dict[str, List[float]]] = {
        g: {"score": [], "drop": []} for g in groups
    }
    background = {"score": [], "drop": []}

    for graph, value in zip(graphs, smiles):
        mol = parse_smiles(value)
        if mol is None or graph is None:
            continue
        data = graph.clone().to(device)
        data.batch = torch.zeros(data.x_cat.size(0), dtype=torch.long, device=device)

        atom_sets = {g: library.atom_set(mol, g) for g in groups}
        for _ in range(repeats):
            node_scores, _ = generator(data, mean, std, clamp)
            keep = generator.rank_mask(node_scores, generator.keep_ratio)
            scores = node_scores.reshape(-1).cpu().numpy()
            dropped = (1.0 - keep.reshape(-1).cpu().numpy())

            background["score"].append(float(scores.mean()))
            background["drop"].append(float(dropped.mean()))
            for group, atoms in atom_sets.items():
                valid = [a for a in atoms if a < scores.size]
                if not valid:
                    continue
                records[group]["score"].append(float(scores[valid].mean()))
                records[group]["drop"].append(float(dropped[valid].mean()))

    rows = []
    base_score = float(np.mean(background["score"])) if background["score"] else float("nan")
    base_drop = float(np.mean(background["drop"])) if background["drop"] else float("nan")
    for group, payload in records.items():
        if not payload["score"]:
            rows.append({"motif": group, "n_occurrences": 0})
            continue
        rows.append({
            "motif": group,
            "n_occurrences": len(payload["score"]) // max(repeats, 1),
            "gate_score": float(np.mean(payload["score"])),
            "gate_score_sd": float(np.std(payload["score"], ddof=1))
            if len(payload["score"]) > 1 else float("nan"),
            "drop_rate": float(np.mean(payload["drop"])),
            "gate_score_vs_background": float(np.mean(payload["score"])) - base_score,
            "drop_rate_vs_background": float(np.mean(payload["drop"])) - base_drop,
        })
    frame = pd.DataFrame(rows)
    frame.attrs["background_gate_score"] = base_score
    frame.attrs["background_drop_rate"] = base_drop
    return frame
