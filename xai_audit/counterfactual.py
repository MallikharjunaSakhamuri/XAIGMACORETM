"""Counterfactual motif ablation (Figure 3).

The ablation T_m(x) removes the atoms matched by a motif from a molecule,
re-sanitises the result, and re-predicts. The change in prediction is

    delta = f(T_m(x)) - f(x)

Removal is structural, done on the molecular graph with RDKit, not a masking
of node features: deleting an atom leaves RDKit to fill the open valence with
implicit hydrogens, which is the standard substituent-to-hydrogen ablation.
Edits that fail sanitisation are reported as invalid rather than silently
dropped, because a motif that cannot be removed without breaking the molecule
is a finding about the motif, not a missing row.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger

from .gmacore_model import featurize
from .motifs import MotifLibrary, parse_smiles

RDLogger.DisableLog("rdApp.*")


@dataclass
class Ablation:
    smiles: str
    motif: str
    n_atoms_removed: int
    valid: bool
    modified_smiles: Optional[str]
    prediction_original: float
    prediction_modified: float
    delta: float
    attribution: float

    def as_row(self) -> Dict[str, object]:
        return asdict(self)


def remove_motif(smiles: str, atom_indices: Sequence[int]):
    """Delete the given atoms and return (canonical SMILES, mol) or (None, None)."""
    mol = parse_smiles(smiles)
    if mol is None or not len(atom_indices):
        return None, None
    editable = Chem.RWMol(mol)
    for index in sorted({int(a) for a in atom_indices}, reverse=True):
        if index < editable.GetNumAtoms():
            editable.RemoveAtom(index)
    try:
        edited = editable.GetMol()
        Chem.SanitizeMol(edited)
    except Exception:
        return None, None
    if edited.GetNumAtoms() == 0:
        return None, None
    # Keep the largest fragment: removing a linker can disconnect the graph,
    # and a disconnected result is not the counterfactual we intend to score.
    fragments = Chem.GetMolFrags(edited, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        edited = max(fragments, key=lambda m: m.GetNumAtoms())
        try:
            Chem.SanitizeMol(edited)
        except Exception:
            return None, None
    try:
        return Chem.MolToSmiles(edited), edited
    except Exception:
        return None, None


@torch.no_grad()
def predict_smiles(model, smiles_list: Sequence[str], device: str = "cpu"):
    """Score a list of SMILES with a fine-tuned downstream model."""
    from torch_geometric.data import Batch

    graphs, keep = [], []
    for index, value in enumerate(smiles_list):
        graph = featurize(value, [0.0])
        if graph is not None:
            graphs.append(graph)
            keep.append(index)
    out = np.full(len(smiles_list), np.nan, dtype=float)
    if not graphs:
        return out
    batch = Batch.from_data_list(graphs).to(device)
    logits = model(batch).reshape(len(graphs), -1)[:, 0].detach().cpu().numpy()
    out[keep] = logits
    return out


def ablate_top_motifs(
    model,
    smiles_list: Sequence[str],
    motif_attributions: Sequence[Dict[str, float]],
    library: MotifLibrary,
    k: int = 3,
    device: str = "cpu",
) -> List[Ablation]:
    """Remove each molecule's k most strongly attributed motifs, one at a time.

    `k` defaults to 3, matching Appendix Table A.4. `motif_attributions` is one
    dict per molecule, as produced by `attribution.aggregate_to_motifs`.
    """
    originals = predict_smiles(model, list(smiles_list), device=device)

    pending, records = [], []
    for position, (smiles, attribution) in enumerate(zip(smiles_list, motif_attributions)):
        mol = parse_smiles(smiles)
        if mol is None:
            continue
        ranked = sorted(
            ((m, v) for m, v in attribution.items() if np.isfinite(v)),
            key=lambda item: abs(item[1]), reverse=True,
        )[:k]
        for motif, value in ranked:
            atoms = library.atom_set(mol, motif)
            if not atoms:
                continue
            modified, _ = remove_motif(smiles, atoms)
            records.append({
                "smiles": smiles, "motif": motif, "n_atoms_removed": len(atoms),
                "attribution": float(value),
                "prediction_original": float(originals[position]),
                "modified_smiles": modified,
            })
            pending.append(modified)

    valid_positions = [i for i, s in enumerate(pending) if s]
    scores = np.full(len(pending), np.nan, dtype=float)
    if valid_positions:
        scored = predict_smiles(model, [pending[i] for i in valid_positions],
                                device=device)
        for slot, value in zip(valid_positions, scored):
            scores[slot] = value

    out: List[Ablation] = []
    for index, record in enumerate(records):
        modified_score = float(scores[index])
        valid = bool(record["modified_smiles"]) and np.isfinite(modified_score)
        out.append(Ablation(
            smiles=record["smiles"], motif=record["motif"],
            n_atoms_removed=int(record["n_atoms_removed"]), valid=valid,
            modified_smiles=record["modified_smiles"],
            prediction_original=record["prediction_original"],
            prediction_modified=modified_score if valid else float("nan"),
            delta=(modified_score - record["prediction_original"]) if valid else float("nan"),
            attribution=record["attribution"],
        ))
    return out


def concordance(ablations: Sequence[Ablation]) -> Dict[str, float]:
    """Does attributed importance predict the measured effect of removal?

    Reports the Spearman correlation between absolute attribution and absolute
    delta, and the share of ablations whose delta carries the sign the
    attribution implies. Both are the quantitative version of the claim that
    Figure 3 makes visually.
    """
    from scipy import stats

    rows = [a for a in ablations if a.valid and np.isfinite(a.delta)]
    if len(rows) < 3:
        return {"n": len(rows), "spearman_rho": float("nan"),
                "spearman_p": float("nan"), "sign_agreement": float("nan")}
    attribution = np.array([a.attribution for a in rows])
    delta = np.array([a.delta for a in rows])
    rho, pvalue = stats.spearmanr(np.abs(attribution), np.abs(delta))
    # Removing a positively contributing motif should lower the prediction.
    agreement = float(np.mean(np.sign(delta) == -np.sign(attribution)))
    return {"n": len(rows), "spearman_rho": float(rho), "spearman_p": float(pvalue),
            "sign_agreement": agreement}
