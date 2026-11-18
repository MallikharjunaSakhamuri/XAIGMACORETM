"""KernelSHAP attribution and its aggregation to nodes and motifs.

This module is deliberately free of any torch dependency. A model is supplied
as a plain callable mapping a batch of flattened node-feature vectors to model
outputs, exactly as the original `GraphModelWrapper` in the project notebooks
did. The torch-specific adapter lives in `model_adapter.py`, so the
aggregation logic here can be unit tested without loading a checkpoint.

Implements Equations 6 to 8 of the manuscript:
    node score      s_i(x)      = sum over the feature channels of node i
    motif score     s_m(x)      = sum over the nodes matched by motif m
    dataset summary I_m         = mean absolute motif score over molecules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Node feature layout from gmacore.data.featurizer: two categorical channels
# followed by eight physicochemical channels, per atom.
NODE_FEATURE_NAMES: List[str] = [
    "AtomIndex", "Chirality",
    "ElemWeight", "ElemLogP", "FormalCharge", "Hybridization",
    "IsAromatic", "NumHs", "Valence", "Degree",
]
N_NODE_FEATURES = len(NODE_FEATURE_NAMES)

ModelFunction = Callable[[np.ndarray], np.ndarray]


def kernel_shap_values(
    model_fn: ModelFunction,
    background: np.ndarray,
    instance: np.ndarray,
    nsamples: int | str = "auto",
    link: str = "identity",
    l1_reg: object = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """KernelSHAP values for one flattened instance.

    KernelSHAP is model agnostic, which is what makes it applicable to a
    message-passing predictor. Note that its additivity property holds for the
    estimated values under the supplied background, approximately rather than
    exactly; the manuscript wording should reflect that.

    `l1_reg` defaults to 0.0 rather than the shap library default of "auto".
    The library default performs AIC-based feature selection and drives a
    subset of attributions to exactly zero, which would silently remove atoms
    from the motif aggregation and distort the concentration measure. Any
    departure from 0.0 must be reported in the manuscript, since it changes
    the attributions rather than only their precision.
    """
    import shap

    np.random.seed(seed)
    explainer = shap.KernelExplainer(model_fn, background, link=link)
    values = explainer.shap_values(
        instance, nsamples=nsamples, l1_reg=l1_reg, silent=True
    )
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values).reshape(-1)


def aggregate_to_nodes(
    feature_values: np.ndarray, n_nodes: int, n_features: int = N_NODE_FEATURES
) -> np.ndarray:
    """Equation 6: sum feature attributions within each node."""
    expected = n_nodes * n_features
    values = np.asarray(feature_values, dtype=float).reshape(-1)
    if values.size != expected:
        raise ValueError(
            f"expected {expected} attributions for {n_nodes} nodes "
            f"x {n_features} features, received {values.size}"
        )
    return values.reshape(n_nodes, n_features).sum(axis=1)


def aggregate_to_motifs(
    node_scores: np.ndarray,
    atom_sets: Dict[str, Sequence[int]],
    normalize: bool = False,
) -> Dict[str, float]:
    """Equation 7: sum node scores over the atoms matched by each motif.

    With `normalize` the size-normalized variant is returned instead, which
    prevents large motifs from ranking highly purely by atom count.
    """
    node_scores = np.asarray(node_scores, dtype=float)
    out: Dict[str, float] = {}
    for motif, atoms in atom_sets.items():
        atoms = [int(a) for a in atoms if 0 <= int(a) < node_scores.size]
        if not atoms:
            out[motif] = float("nan")
            continue
        total = float(node_scores[atoms].sum())
        out[motif] = total / len(atoms) if normalize else total
    return out


def motif_importance(per_molecule: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Equation 8: mean absolute motif attribution across molecules.

    Molecules in which a motif does not occur contribute nothing rather than a
    zero, so the summary is not diluted by absence.
    """
    totals: Dict[str, List[float]] = {}
    for record in per_molecule:
        for motif, value in record.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            totals.setdefault(motif, []).append(abs(float(value)))
    return {motif: float(np.mean(vals)) for motif, vals in totals.items() if vals}


def top_k_share(importances: Dict[str, float], k: int = 3) -> float:
    """Bias measure 1: fraction of total attribution held by the top k motifs.

    A value near 1 means predictive influence is concentrated in a few
    recurrent substructures; a value near k/number of motifs means influence is
    spread evenly.
    """
    values = np.array([v for v in importances.values() if np.isfinite(v)], dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return float("nan")
    ordered = np.sort(values)[::-1]
    return float(ordered[: min(k, ordered.size)].sum() / values.sum())


@dataclass
class StabilityResult:
    """Bias measure 2: attribution stability between two novelty partitions."""

    n_novel: int
    n_similar: int
    sign_consistency_novel: float
    sign_consistency_similar: float
    sign_consistency_gap: float
    dispersion_novel: float
    dispersion_similar: float
    dispersion_ratio: float

    def as_dict(self) -> Dict[str, float]:
        from dataclasses import asdict
        return asdict(self)


def _sign_consistency(matrix: np.ndarray) -> float:
    """Mean over features of the dominant-sign fraction across molecules.

    One means every molecule attributes a feature in the same direction; 0.5
    means the direction is effectively random across molecules.
    """
    if matrix.size == 0:
        return float("nan")
    signs = np.sign(matrix)
    positive = (signs > 0).sum(axis=0)
    negative = (signs < 0).sum(axis=0)
    total = positive + negative
    valid = total > 0
    if not valid.any():
        return float("nan")
    dominant = np.maximum(positive[valid], negative[valid]) / total[valid]
    return float(dominant.mean())


def explanation_stability(
    novel_values: np.ndarray, similar_values: np.ndarray
) -> StabilityResult:
    """Compare attribution coherence between novel and training-similar sets.

    Both arrays are molecules by features. A higher sign consistency and a
    lower dispersion indicate more stable explanations. Report the direction as
    it comes out: an augmentation that broadens coverage may well improve
    stability on novel inputs rather than degrade it.
    """
    novel = np.atleast_2d(np.asarray(novel_values, dtype=float))
    similar = np.atleast_2d(np.asarray(similar_values, dtype=float))

    disp_novel = float(np.nanmean(np.nanstd(novel, axis=0))) if novel.size else float("nan")
    disp_similar = (
        float(np.nanmean(np.nanstd(similar, axis=0))) if similar.size else float("nan")
    )
    sc_novel, sc_similar = _sign_consistency(novel), _sign_consistency(similar)

    return StabilityResult(
        n_novel=int(novel.shape[0]) if novel.size else 0,
        n_similar=int(similar.shape[0]) if similar.size else 0,
        sign_consistency_novel=sc_novel,
        sign_consistency_similar=sc_similar,
        sign_consistency_gap=sc_similar - sc_novel,
        dispersion_novel=disp_novel,
        dispersion_similar=disp_similar,
        dispersion_ratio=disp_novel / disp_similar if disp_similar else float("nan"),
    )


def novelty_partition(
    similarities: Sequence[float], q_low: float = 0.25, q_high: float = 0.75
):
    """Equations 3 and 4: split a test set by similarity to the training set.

    Defaults follow Appendix Table A.4. Note that Section 3.3 illustrates 0.10
    and 0.90 instead; the manuscript must state one pair consistently.
    """
    values = np.asarray(similarities, dtype=float)
    low, high = np.quantile(values, q_low), np.quantile(values, q_high)
    return values <= low, values >= high
