"""Disparity measures for the functional-group and rule-partition analyses.

Every function takes raw per-molecule observations rather than pre-aggregated
summaries, so that valid hypothesis tests can be computed alongside the point
estimates and corrected for multiplicity downstream.

The same functions serve two uses:
  * functional-group disparity, where the mask is motif present vs absent;
  * rule-partition parity, where the mask is non-compliant (G=1) vs compliant.

They also serve two score sources:
  * ground-truth labels, giving the data-level imbalance for Table A;
  * model predictions, giving the model-level disparity for Table B.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

EPS = 1e-10


def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    """Two-sided Wilson score interval for a proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    p = successes / total
    denom = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(numer: int, denom: int) -> float:
    return float(numer) / denom if denom > 0 else float("nan")


@dataclass
class ClassificationDisparity:
    """Binary-outcome disparity between a group and its complement."""

    n_group: int
    n_comp: int
    pos_group: int
    pos_comp: int
    pr_group: float
    pr_comp: float
    pr_group_ci: tuple
    pr_comp_ci: tuple
    disparate_impact: float
    statistical_parity_diff: float
    equalized_odds_diff: float
    tpr_group: float
    tpr_comp: float
    fpr_group: float
    fpr_comp: float
    ppv_group: float
    ppv_comp: float
    acc_group: float
    acc_comp: float
    p_statistical_parity: float
    p_tpr: float
    p_fpr: float

    def as_dict(self) -> Dict:
        return asdict(self)


def classification_disparity(
    outcome: Sequence[int],
    mask: Sequence[bool],
    y_true: Optional[Sequence[int]] = None,
) -> ClassificationDisparity:
    """Disparity of a binary `outcome` between mask=True and mask=False.

    `outcome` is the quantity whose rate is compared. Pass ground-truth labels
    to obtain the data-level base-rate gap, or model predictions to obtain the
    prediction-rate gap. The two are then directly comparable, being in the
    same units.

    `y_true` enables the error-conditioned quantities (TPR, FPR, PPV, accuracy)
    and is required for the equalized-odds difference. When `outcome` is itself
    the ground truth, leave `y_true` as None.
    """
    outcome = np.asarray(outcome).astype(int)
    mask = np.asarray(mask).astype(bool)

    grp, comp = outcome[mask], outcome[~mask]
    n_g, n_c = grp.size, comp.size
    pos_g, pos_c = int(grp.sum()), int(comp.sum())
    pr_g, pr_c = _rate(pos_g, n_g), _rate(pos_c, n_c)

    # Fisher exact on the 2x2 table of group membership against the outcome.
    table = [[pos_g, n_g - pos_g], [pos_c, n_c - pos_c]]
    p_sp = stats.fisher_exact(table).pvalue if min(n_g, n_c) > 0 else float("nan")

    tpr_g = tpr_c = fpr_g = fpr_c = ppv_g = ppv_c = acc_g = acc_c = float("nan")
    eod = p_tpr = p_fpr = float("nan")

    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)
        yg, yc = y_true[mask], y_true[~mask]

        tp_g, pos_lab_g = int(((grp == 1) & (yg == 1)).sum()), int((yg == 1).sum())
        tp_c, pos_lab_c = int(((comp == 1) & (yc == 1)).sum()), int((yc == 1).sum())
        fp_g, neg_lab_g = int(((grp == 1) & (yg == 0)).sum()), int((yg == 0).sum())
        fp_c, neg_lab_c = int(((comp == 1) & (yc == 0)).sum()), int((yc == 0).sum())

        tpr_g, tpr_c = _rate(tp_g, pos_lab_g), _rate(tp_c, pos_lab_c)
        fpr_g, fpr_c = _rate(fp_g, neg_lab_g), _rate(fp_c, neg_lab_c)
        ppv_g, ppv_c = _rate(tp_g, pos_g), _rate(tp_c, pos_c)
        acc_g, acc_c = _rate(int((grp == yg).sum()), n_g), _rate(int((comp == yc).sum()), n_c)

        if not (np.isnan(tpr_g) or np.isnan(tpr_c) or np.isnan(fpr_g) or np.isnan(fpr_c)):
            eod = abs(tpr_g - tpr_c) + abs(fpr_g - fpr_c)

        if pos_lab_g > 0 and pos_lab_c > 0:
            p_tpr = stats.fisher_exact(
                [[tp_g, pos_lab_g - tp_g], [tp_c, pos_lab_c - tp_c]]
            ).pvalue
        if neg_lab_g > 0 and neg_lab_c > 0:
            p_fpr = stats.fisher_exact(
                [[fp_g, neg_lab_g - fp_g], [fp_c, neg_lab_c - fp_c]]
            ).pvalue

    return ClassificationDisparity(
        n_group=n_g, n_comp=n_c, pos_group=pos_g, pos_comp=pos_c,
        pr_group=pr_g, pr_comp=pr_c,
        pr_group_ci=wilson_interval(pos_g, n_g),
        pr_comp_ci=wilson_interval(pos_c, n_c),
        disparate_impact=pr_g / max(EPS, pr_c) if not np.isnan(pr_g) else float("nan"),
        statistical_parity_diff=pr_g - pr_c,
        equalized_odds_diff=eod,
        tpr_group=tpr_g, tpr_comp=tpr_c, fpr_group=fpr_g, fpr_comp=fpr_c,
        ppv_group=ppv_g, ppv_comp=ppv_c, acc_group=acc_g, acc_comp=acc_c,
        p_statistical_parity=p_sp, p_tpr=p_tpr, p_fpr=p_fpr,
    )


@dataclass
class RegressionDisparity:
    """Continuous-target disparity between a group and its complement."""

    n_group: int
    n_comp: int
    mean_group: float
    mean_comp: float
    mean_gap: float
    cohens_d: float
    p_mean_gap: float
    mae_group: float
    mae_comp: float
    mae_gap: float
    p_mae_gap: float

    def as_dict(self) -> Dict:
        return asdict(self)


def regression_disparity(
    values: Sequence[float],
    mask: Sequence[bool],
    y_true: Optional[Sequence[float]] = None,
) -> RegressionDisparity:
    """Groupwise mean and error gaps for a continuous quantity.

    Classification parity metrics are not applied to regression outputs, since
    they would require an unstated binarisation threshold. `values` carries
    either the measured target (data level) or the model prediction (model
    level); both are reported in target units so they are comparable.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask).astype(bool)
    grp, comp = values[mask], values[~mask]
    n_g, n_c = grp.size, comp.size

    mean_g = float(grp.mean()) if n_g else float("nan")
    mean_c = float(comp.mean()) if n_c else float("nan")

    d = p_mean = float("nan")
    if n_g > 1 and n_c > 1:
        pooled = np.sqrt(((n_g - 1) * grp.var(ddof=1) + (n_c - 1) * comp.var(ddof=1))
                         / (n_g + n_c - 2))
        d = (mean_g - mean_c) / pooled if pooled > 0 else float("nan")
        p_mean = stats.ttest_ind(grp, comp, equal_var=False).pvalue

    mae_g = mae_c = mae_gap = p_mae = float("nan")
    if y_true is not None:
        residual = np.abs(np.asarray(y_true, dtype=float) - values)
        rg, rc = residual[mask], residual[~mask]
        if rg.size and rc.size:
            mae_g, mae_c = float(rg.mean()), float(rc.mean())
            mae_gap = mae_g - mae_c
            if rg.size > 1 and rc.size > 1:
                p_mae = stats.ttest_ind(rg, rc, equal_var=False).pvalue

    return RegressionDisparity(
        n_group=n_g, n_comp=n_c, mean_group=mean_g, mean_comp=mean_c,
        mean_gap=mean_g - mean_c, cohens_d=d, p_mean_gap=p_mean,
        mae_group=mae_g, mae_comp=mae_c, mae_gap=mae_gap, p_mae_gap=p_mae,
    )


def amplification(gap_reference: float, gap_target: float) -> float:
    """Change in disparity magnitude from reference to target.

    Positive means the target model carries a larger absolute disparity than
    the reference; negative means the disparity is reduced. Signed gaps are
    retained separately for interpretation, since a sign flip is not the same
    finding as a magnitude increase.
    """
    if np.isnan(gap_reference) or np.isnan(gap_target):
        return float("nan")
    return abs(gap_target) - abs(gap_reference)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05):
    """BH step-up procedure. Returns (rejected, adjusted p-values).

    NaN entries are carried through as NaN and excluded from the ranking, so a
    group with an undefined test does not shift the correction for the others.
    """
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    rejected = np.zeros(p.shape, dtype=bool)

    finite = np.isfinite(p)
    if not finite.any():
        return rejected, adjusted

    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = ranked.size
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    out = np.empty(m)
    out[order] = adj
    adjusted[finite] = out
    rejected[finite] = out <= alpha
    return rejected, adjusted
