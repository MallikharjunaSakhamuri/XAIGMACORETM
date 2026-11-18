"""Table B: model-level disparity, and its change relative to the data.

Measures 3 and 4 of the audit, computed from the per-molecule predictions
written by stage 2. Neither needs attribution, only forward passes, which is
why they are separated from the SHAP-based measures 1 and 2.

Two reference points are reported for every group:

  data   the disparity carried by the ground-truth labels of the SAME test
         molecules. Computing it on the test split rather than the whole
         benchmark keeps the comparison like for like; Table A gives the
         whole-dataset picture separately.
  model  the disparity in the model's predictions on those molecules.

`amplification` is |model gap| - |data gap|. Positive means the model carries
a larger disparity than the labels do, negative means a smaller one. Signed
gaps are kept alongside, because a sign flip is a different finding from a
magnitude change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .disparity import (
    amplification,
    benjamini_hochberg,
    classification_disparity,
    regression_disparity,
)
from .motifs import DATASET_GROUPS, MotifLibrary, parse_smiles
from .rules import CANONICAL_RULES

DEFAULT_THRESHOLD = 0.5


def _membership(smiles: Sequence[str], groups: List[str]) -> pd.DataFrame:
    library = MotifLibrary.build(groups)
    records = []
    for value in smiles:
        mol = parse_smiles(value)
        records.append(library.membership(mol))
    return pd.DataFrame(records)


def _rule_labels(smiles: Sequence[str], dataset: str) -> np.ndarray:
    rule = CANONICAL_RULES[dataset]
    labels = []
    for value in smiles:
        mol = parse_smiles(value)
        labels.append(1 if mol is None else rule.group_label(mol))
    return np.asarray(labels)


def _classification_row(outcome, mask, y_true=None) -> Dict[str, float]:
    stat = classification_disparity(outcome, mask, y_true=y_true)
    return {
        "positive_rate_group": stat.pr_group,
        "positive_rate_comp": stat.pr_comp,
        "disparate_impact": stat.disparate_impact,
        "statistical_parity_diff": stat.statistical_parity_diff,
        "equalized_odds_diff": stat.equalized_odds_diff,
        "p_value": stat.p_statistical_parity,
    }


def _regression_row(values, mask, y_true=None) -> Dict[str, float]:
    stat = regression_disparity(values, mask, y_true=y_true)
    return {
        "mean_group": stat.mean_group,
        "mean_comp": stat.mean_comp,
        "mean_gap": stat.mean_gap,
        "cohens_d": stat.cohens_d,
        "mae_gap": stat.mae_gap,
        "p_value": stat.p_mean_gap,
    }


def disparity_for_run(
    frame: pd.DataFrame,
    dataset: str,
    task: str,
    model: str,
    seed: int,
    threshold: float = DEFAULT_THRESHOLD,
    min_arm_size: int = 20,
    groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-group disparity for one dataset, model and seed."""
    groups = groups or DATASET_GROUPS[dataset]
    smiles = frame["smiles"].tolist()
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)

    membership = _membership(smiles, groups)
    rule_labels = _rule_labels(smiles, dataset)

    partitions: List[tuple] = [(g, membership[g].to_numpy().astype(bool)) for g in groups]
    partitions.append(("RULE:non_compliant", rule_labels == 1))

    if task == "classification":
        decision = (y_pred >= threshold).astype(int)
        label = y_true.astype(int)

    rows = []
    for name, mask in partitions:
        n_group, n_comp = int(mask.sum()), int((~mask).sum())
        row: Dict[str, object] = {
            "dataset": dataset, "model": model, "seed": seed, "group": name,
            "n_group": n_group, "n_comp": n_comp, "n_test": len(frame),
            "underpowered": bool(min(n_group, n_comp) < min_arm_size),
        }
        if n_group == 0 or n_comp == 0:
            row["note"] = "degenerate on this test split"
            rows.append(row)
            continue

        if task == "classification":
            data_side = _classification_row(label, mask)
            model_side = _classification_row(decision, mask, y_true=label)
            key = "statistical_parity_diff"
        else:
            data_side = _regression_row(y_true, mask)
            model_side = _regression_row(y_pred, mask, y_true=y_true)
            key = "mean_gap"

        for prefix, payload in (("data", data_side), ("model", model_side)):
            for metric, value in payload.items():
                row[f"{prefix}_{metric}"] = value

        row["amplification"] = amplification(data_side[key], model_side[key])
        row["sign_flip"] = bool(
            np.isfinite(data_side[key]) and np.isfinite(model_side[key])
            and np.sign(data_side[key]) != np.sign(model_side[key])
            and data_side[key] != 0 and model_side[key] != 0
        )
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across seeds, per dataset, model and group."""
    numeric = [
        c for c in per_seed.columns
        if per_seed[c].dtype.kind in "fc" and c not in ("seed",)
    ]
    grouped = per_seed.groupby(["dataset", "model", "group"], dropna=False)

    out = grouped[numeric].agg(["mean", "std"])
    out.columns = [f"{metric}_{statistic}" for metric, statistic in out.columns]
    out = out.reset_index()

    context = grouped.agg(
        n_group=("n_group", "first"),
        n_comp=("n_comp", "first"),
        n_test=("n_test", "first"),
        underpowered=("underpowered", "first"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return context.merge(out, on=["dataset", "model", "group"], how="left")


def apply_multiple_comparisons(
    table: pd.DataFrame, alpha: float = 0.05, column: str = "model_p_value_mean"
) -> pd.DataFrame:
    """Benjamini-Hochberg across the family of group tests.

    The family is every group test in the table, which is what Reviewer 1
    point 3 asks about: many motifs, several datasets and two models tested
    together. Underpowered groups are included in the correction rather than
    quietly dropped, since excluding them would understate the multiplicity.
    """
    table = table.copy()
    if column not in table.columns:
        raise KeyError(f"{column} not in the table; columns: {list(table.columns)[:12]}")
    rejected, adjusted = benjamini_hochberg(table[column].to_numpy(float), alpha=alpha)
    table["p_adjusted_bh"] = adjusted
    table["significant_bh"] = rejected
    table["n_tests_in_family"] = int(np.isfinite(table[column].to_numpy(float)).sum())
    return table


def contrast_models(
    table: pd.DataFrame,
    reference: str = "ManualAug-CL",
    target: str = "GMACORE-TM",
    metric: str = "amplification_mean",
) -> pd.DataFrame:
    """Side-by-side of the two models, with the difference between them.

    This is the column the reviewers asked for: whether a disparity is larger,
    smaller or unchanged under GMACORE-TM relative to ManualAug-CL.

    The labels name the model, not the mechanism. ManualAug-CL differs from
    GMACORE-TM in the momentum encoder and age-weighted negatives as well as
    the generative augmentation, so the contrast supports differences
    associated with the complete training pipeline and is not a causal
    estimate for the generator alone (R1-1, R5-1).
    """
    wide = table.pivot_table(
        index=["dataset", "group"], columns="model", values=metric, aggfunc="first"
    ).reset_index()
    for model in (reference, target):
        if model not in wide.columns:
            raise KeyError(f"model '{model}' absent; present: {list(wide.columns)}")
    wide["difference"] = wide[target] - wide[reference]
    wide["direction"] = np.where(
        ~np.isfinite(wide["difference"]), "undetermined",
        np.where(wide["difference"] > 0, "amplified_under_gmacore_tm",
                 np.where(wide["difference"] < 0, "reduced_under_gmacore_tm", "unchanged")),
    )
    return wide
