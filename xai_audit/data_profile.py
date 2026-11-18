"""Table A: data-level characteristics of the benchmark data.

These are properties of the data, not of any model. Raw molecules have no
attribution values and no explanation stability, so this table reports only
what the data itself can support: how common each group is, what its observed
label rate or target mean is, how rule-partition membership is distributed,
and how structurally diverse the set is.

Table A therefore establishes which imbalances pre-date any augmentation. It
is the reference against which the model-level results in Table B are read.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .disparity import classification_disparity, regression_disparity
from .motifs import DATASET_GROUPS, MotifLibrary
from .rules import CANONICAL_RULES


def group_membership_frame(frame: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    """Binary membership matrix M_x^m over the requested groups."""
    library = MotifLibrary.build(groups)
    records = [library.membership(mol) for mol in frame["mol"]]
    return pd.DataFrame(records, index=frame.index)


def profile_dataset(
    frame: pd.DataFrame,
    groups: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    min_arm_size: int = 20,
) -> pd.DataFrame:
    """Per-group prevalence and observed outcome gap for one benchmark.

    `min_arm_size` flags groups whose present or absent arm is too small to
    support a stable disparity estimate. Such groups are still reported, but
    marked, so that an unstable ratio computed over a handful of molecules is
    not read as a finding.
    """
    name = dataset_name or frame.attrs.get("name")
    task = frame.attrs["task_type"]
    groups = groups or DATASET_GROUPS[name]

    membership = group_membership_frame(frame, groups)
    target = frame["target"].to_numpy()
    n_total = len(frame)

    rows = []
    for group in groups:
        mask = membership[group].to_numpy().astype(bool)
        row: Dict[str, object] = {
            "dataset": name,
            "group": group,
            "n_total": n_total,
            "n_present": int(mask.sum()),
            "prevalence": float(mask.mean()),
        }
        n_present, n_absent = int(mask.sum()), int((~mask).sum())
        if n_present == 0 or n_absent == 0:
            row["note"] = "degenerate group, no contrast available"
            rows.append(row)
            continue
        if min(n_present, n_absent) < min_arm_size:
            row["underpowered"] = True
            row["note"] = (
                f"smallest arm has {min(n_present, n_absent)} molecules, "
                f"below the min_arm_size of {min_arm_size}"
            )
        else:
            row["underpowered"] = False

        if task == "classification":
            stat = classification_disparity(target.astype(int), mask)
            row.update({
                "base_rate_present": stat.pr_group,
                "base_rate_absent": stat.pr_comp,
                "label_rate_ratio": stat.disparate_impact,
                "label_rate_gap": stat.statistical_parity_diff,
                "p_value": stat.p_statistical_parity,
                "ci_present_low": stat.pr_group_ci[0],
                "ci_present_high": stat.pr_group_ci[1],
            })
        else:
            stat = regression_disparity(target.astype(float), mask)
            row.update({
                "target_mean_present": stat.mean_group,
                "target_mean_absent": stat.mean_comp,
                "target_mean_gap": stat.mean_gap,
                "cohens_d": stat.cohens_d,
                "p_value": stat.p_mean_gap,
            })
        rows.append(row)

    return pd.DataFrame(rows)


def profile_rule_partition(
    frame: pd.DataFrame, dataset_name: Optional[str] = None
) -> pd.DataFrame:
    """Rule-partition composition and observed outcome gap."""
    name = dataset_name or frame.attrs.get("name")
    task = frame.attrs["task_type"]
    rule = CANONICAL_RULES[name]

    labels = np.array([rule.group_label(mol) for mol in frame["mol"]])
    mask = labels == 1  # non-compliant is the focal group, matching the manuscript
    target = frame["target"].to_numpy()

    row: Dict[str, object] = {
        "dataset": name,
        "rule": rule.name,
        "n_total": len(frame),
        "n_non_compliant": int(mask.sum()),
        "n_compliant": int((~mask).sum()),
        "fraction_non_compliant": float(mask.mean()),
    }
    if mask.sum() in (0, len(frame)):
        row["note"] = "degenerate partition, no contrast available"
        return pd.DataFrame([row])

    if task == "classification":
        stat = classification_disparity(target.astype(int), mask)
        row.update({
            "base_rate_non_compliant": stat.pr_group,
            "base_rate_compliant": stat.pr_comp,
            "label_rate_gap": stat.statistical_parity_diff,
            "p_value": stat.p_statistical_parity,
        })
    else:
        stat = regression_disparity(target.astype(float), mask)
        row.update({
            "target_mean_non_compliant": stat.mean_group,
            "target_mean_compliant": stat.mean_comp,
            "target_mean_gap": stat.mean_gap,
            "cohens_d": stat.cohens_d,
            "p_value": stat.p_mean_gap,
        })
    return pd.DataFrame([row])
