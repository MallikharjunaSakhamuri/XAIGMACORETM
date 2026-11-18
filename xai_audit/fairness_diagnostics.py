"""Group fairness diagnostics for Figures 7 and 8.

Figure 8, classification: per-group performance (accuracy, TPR, TNR, precision)
and five parity gaps, all computed here from the confusion matrices. These are
elementary definitions and are implemented directly so that every number in the
figure can be traced to a formula in this file.

Figure 7, regression: independence, separation and sufficiency. These are DALEX
constructs rather than closed-form ratios, so they are delegated to the `dalex`
package when it is installed. Without it the prediction-distribution panels are
still produced and the diagnostic panels are skipped with a message, rather
than being replaced by a different statistic under the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np

EPS = 1e-10

CLASSIFICATION_GAPS = {
    "statistical_parity": "positive_rate",
    "equal_opportunity": "tpr",
    "predictive_equality": "fpr",
    "predictive_parity": "ppv",
    "accuracy_equality": "accuracy",
}

RADAR_METRICS = ["accuracy", "tpr", "tnr", "ppv"]
RADAR_LABELS = ["Accuracy", "True positive rate", "True negative rate", "Precision"]


def _safe(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else float("nan")


@dataclass
class GroupMetrics:
    n: int
    positives: int
    accuracy: float
    tpr: float
    tnr: float
    fpr: float
    ppv: float
    positive_rate: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def classification_metrics(y_true: Sequence[float], y_pred_label: Sequence[float]
                           ) -> GroupMetrics:
    """Confusion-matrix metrics for one subgroup."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_label = np.asarray(y_pred_label, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred_label)
    y_true, y_pred_label = y_true[mask], y_pred_label[mask]

    tp = int(np.sum((y_true == 1) & (y_pred_label == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_label == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_label == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_label == 0)))
    total = tp + tn + fp + fn

    return GroupMetrics(
        n=total,
        positives=int(np.sum(y_true == 1)),
        accuracy=_safe(tp + tn, total),
        tpr=_safe(tp, tp + fn),
        tnr=_safe(tn, tn + fp),
        fpr=_safe(fp, fp + tn),
        ppv=_safe(tp, tp + fp),
        positive_rate=_safe(tp + fp, total),
    )


def classification_fairness(y_true: Sequence[float], y_score: Sequence[float],
                            group: Sequence[int], threshold: float = 0.5
                            ) -> Dict[str, object]:
    """Figure 8 for one dataset and model.

    `group` is G(x): 1 for non-compliant, 0 for compliant. Gaps are reported as
    non-compliant minus compliant, so zero is parity, matching the figure.
    """
    y_true = np.asarray(y_true, dtype=float)
    labels = (np.asarray(y_score, dtype=float) >= threshold).astype(float)
    group = np.asarray(group, dtype=int)

    non_compliant = classification_metrics(y_true[group == 1], labels[group == 1])
    compliant = classification_metrics(y_true[group == 0], labels[group == 0])

    gaps = {}
    for name, metric in CLASSIFICATION_GAPS.items():
        a = getattr(non_compliant, metric)
        b = getattr(compliant, metric)
        gaps[name] = float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")

    return {"non_compliant": non_compliant.as_dict(),
            "compliant": compliant.as_dict(),
            "gaps": gaps,
            "threshold": float(threshold)}


def regression_group_summary(y_true: Sequence[float], y_pred: Sequence[float],
                             group: Sequence[int]) -> Dict[str, Dict[str, float]]:
    """Per-group prediction and residual summary behind Figure 7a and 7c."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    group = np.asarray(group, dtype=int)

    out: Dict[str, Dict[str, float]] = {}
    for key, mask in (("non_compliant", group == 1), ("compliant", group == 0)):
        yt, yp = y_true[mask], y_pred[mask]
        residual = yt - yp
        out[key] = {
            "n": int(mask.sum()),
            "mean_prediction": float(np.mean(yp)) if yp.size else float("nan"),
            "std_prediction": float(np.std(yp, ddof=1)) if yp.size > 1 else float("nan"),
            "mean_target": float(np.mean(yt)) if yt.size else float("nan"),
            "mean_residual": float(np.mean(residual)) if residual.size else float("nan"),
            "mae": float(np.mean(np.abs(residual))) if residual.size else float("nan"),
            "rmse": float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan"),
        }
    return out


def dalex_regression_fairness(y_true: Sequence[float], y_pred: Sequence[float],
                              group: Sequence[int],
                              privileged: str = "compliant") -> Optional[Dict[str, float]]:
    """Independence, separation and sufficiency, via the dalex package.

    Returns None when dalex is not installed, so the caller can draw the
    distribution panels and skip the diagnostic panels rather than substituting
    a different statistic. Install with `pip install dalex`.
    """
    try:
        import dalex as dx
        import pandas as pd
    except ImportError:
        return None

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    group = np.asarray(group, dtype=int)
    protected = np.where(group == 1, "non_compliant", "compliant")

    # dalex needs an Explainer over a model; the predictions are already
    # computed, so a lookup model over an index column reproduces them exactly
    # without re-running the GNN.
    frame = pd.DataFrame({"index": np.arange(len(y_pred), dtype=float)})
    lookup = dict(zip(frame["index"].to_numpy(), y_pred))

    class _Lookup:
        """Returns precomputed predictions, so dalex never calls the GNN."""

        def predict(self, data):
            keys = np.asarray(data["index"], dtype=float)
            return np.array([lookup[k] for k in keys], dtype=float)

    def predict_function(model, data):
        return model.predict(data)

    explainer = dx.Explainer(
        _Lookup(), frame, y_true, predict_function=predict_function,
        model_type="regression", model_class="xai_audit.LookupModel",
        label="precomputed", verbose=False,
    )
    fairness = explainer.model_fairness(protected=protected, privileged=privileged)
    metrics = fairness.result

    out: Dict[str, float] = {}
    for criterion in ("independence", "separation", "sufficiency"):
        if criterion in metrics.columns:
            series = metrics[criterion].dropna()
            if len(series):
                out[criterion] = float(series.iloc[0])
        elif criterion in metrics.index:
            out[criterion] = float(np.asarray(metrics.loc[criterion]).ravel()[0])
    return out or None
