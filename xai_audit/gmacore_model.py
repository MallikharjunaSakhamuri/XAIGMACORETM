"""Faithful reconstruction of the GMACORE-TM encoder and featurizer.

Copied from the pretraining notebooks (01_prepare_graphs_runpod,
07_pretrain_manualaug_cl_fixed_runpod, 06_pretrain_gmacore_tm_fixedgan_runpod)
and the downstream evaluation notebook, so that a checkpoint loads with
strict=True and the forward pass matches the one that produced it.

The two details that matter and are not in the public GMACORETM repository:
  * `pool_norm` is a LayerNorm applied AFTER global mean pooling;
  * `encode()` returns the pooled, normalized representation, and the
    projection head is discarded for downstream transfer.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

NUM_NODE_CATEGORICAL = 2
NUM_NODE_PHYSICOCHEMICAL = 8
NUM_EDGE_FEATURES = 5
NODE_DIM = NUM_NODE_CATEGORICAL + NUM_NODE_PHYSICOCHEMICAL

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_DIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

_ATOM_INDEX = {z: i for i, z in enumerate(ATOM_LIST)}
_CHIRAL_INDEX = {c: i for i, c in enumerate(CHIRALITY_LIST)}
_BOND_INDEX = {b: i for i, b in enumerate(BOND_LIST)}
_BOND_DIR_INDEX = {d: i for i, d in enumerate(BOND_DIR_LIST)}
_ELEMENT_CACHE: dict = {}


def element_contributions(symbol: str) -> Tuple[float, float]:
    if symbol in _ELEMENT_CACHE:
        return _ELEMENT_CACHE[symbol]
    weight, logp = 0.0, 0.0
    probe = Chem.MolFromSmiles(f"[{symbol}]")
    if probe is not None:
        try:
            weight = float(Descriptors.ExactMolWt(probe))
            logp = float(Descriptors.MolLogP(probe))
        except Exception:
            pass
    _ELEMENT_CACHE[symbol] = (weight, logp)
    return weight, logp


def featurize(smiles: str, label=None, keep_largest_fragment: bool = True) -> Optional[Data]:
    """Identical to the downstream notebook's featurizer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if keep_largest_fragment and "." in smiles:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(fragments) > 1:
            mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumHeavyAtoms() < 2:
        return None

    categorical, physicochemical = [], []
    for atom in mol.GetAtoms():
        categorical.append([
            _ATOM_INDEX.get(atom.GetAtomicNum(), 0),
            _CHIRAL_INDEX.get(atom.GetChiralTag(), 0),
        ])
        weight, logp = element_contributions(atom.GetSymbol())
        physicochemical.append([
            weight, logp, float(atom.GetFormalCharge()),
            float(int(atom.GetHybridization())), float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()), float(atom.GetTotalValence()),
            float(atom.GetDegree()),
        ])

    rows, cols, features = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rotatable = (
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE
            and not bond.IsInRing()
            and bond.GetBeginAtom().GetDegree() > 1
            and bond.GetEndAtom().GetDegree() > 1
        )
        feature = [
            float(_BOND_INDEX.get(bond.GetBondType(), 0)),
            float(_BOND_DIR_INDEX.get(bond.GetBondDir(), 0)),
            float(bond.GetIsConjugated()), float(rotatable), 0.0,
        ]
        rows += [start, end]
        cols += [end, start]
        features += [feature, feature]

    if not rows:
        return None

    data = Data(
        x_cat=torch.tensor(categorical, dtype=torch.long),
        x_phys=torch.tensor(physicochemical, dtype=torch.float),
        edge_index=torch.tensor([rows, cols], dtype=torch.long),
        edge_attr=torch.tensor(features, dtype=torch.float),
        num_nodes=len(categorical),
    )
    data.smiles = smiles
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


def _mlp(a: int, b: int, c: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))


class GraphEncoder(nn.Module):
    """Matches the pretraining architecture exactly, LayerNorm included."""

    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = NUM_EDGE_FEATURES,
                 hidden_dim: int = 128, output_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.node_encoder = _mlp(node_dim, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(edge_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, output_dim if i == num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = dropout
        self.pool_norm = nn.LayerNorm(output_dim)
        self.projection = _mlp(output_dim, hidden_dim, output_dim)

    def encode(self, data, feature_mean, feature_std, clamp):
        x = torch.cat([data.x_cat.float(), data.x_phys.float()], dim=-1)
        x = torch.clamp((x - feature_mean) / feature_std, -clamp, clamp)
        x = self.node_encoder(x)
        _ = self.edge_encoder(data.edge_attr.float())

        edge_weight = getattr(data, "edge_weight", None)
        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index, edge_weight)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.pool_norm(global_mean_pool(x, batch))

    def forward(self, data, feature_mean, feature_std, clamp):
        return self.projection(self.encode(data, feature_mean, feature_std, clamp))


class DownstreamModel(nn.Module):
    """Pretrained encoder plus the linear head used in the downstream notebook."""

    def __init__(self, encoder: GraphEncoder, num_tasks: int, stats, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.mean, self.std, self.clamp = stats
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.convs[-1].out_channels, num_tasks),
        )

    def forward(self, data):
        features = self.encoder.encode(data, self.mean, self.std, self.clamp)
        return self.head(features)


def scaffold_split(smiles: Sequence[str], fractions=(0.8, 0.1, 0.1)):
    """Deterministic Bemis-Murcko scaffold split, largest groups into train."""
    from collections import defaultdict

    groups = defaultdict(list)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""
        groups[key].append(index)

    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    total = len(smiles)
    train_cut = fractions[0] * total
    valid_cut = (fractions[0] + fractions[1]) * total

    train, valid, test = [], [], []
    for group in ordered:
        if len(train) + len(group) <= train_cut:
            train += group
        elif len(train) + len(valid) + len(group) <= valid_cut:
            valid += group
        else:
            test += group
    return train, valid, test


def load_encoder(checkpoint_path: str, device: str = "cpu"):
    """Rebuild the pretrained encoder from a checkpoint, loading strictly.

    strict=True is deliberate. The downstream notebook used strict=False, which
    silently tolerates a mismatched architecture; here any discrepancy must
    surface immediately, since a silently dropped layer would change every
    embedding and therefore every attribution.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload.get("config", {})
    encoder = GraphEncoder(
        node_dim=config.get("node_dim", NODE_DIM),
        edge_dim=config.get("edge_dim", NUM_EDGE_FEATURES),
        hidden_dim=config.get("hidden_dim", 128),
        output_dim=config.get("output_dim", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.1),
    )
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    return encoder.to(device), payload


def load_feature_stats(stats_path: str, device: str = "cpu"):
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    return stats["mean"].to(device), stats["std"].to(device), stats["clamp"]
