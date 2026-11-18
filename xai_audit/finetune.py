"""Fine-tune a pretrained encoder and emit per-molecule test predictions.

Protocol follows 11_downstream_eval_runpod.ipynb: deterministic scaffold split,
a dropout plus linear head on the pooled representation, Adam with separate
head and encoder learning rates, early stopping on the validation primary
metric.

ONE DELIBERATE DEPARTURE. In the notebook, regression predictions are
inverse-transformed back to the original units while the targets remain
standardized, so RMSE, MAE and R2 are computed between quantities on different
scales. Here both sides are returned in original units. Pearson and Spearman
are unaffected by the difference, which is why they looked reasonable in the
notebook's output while R2 did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .gmacore_model import (
    DownstreamModel,
    GraphEncoder,
    load_encoder,
    load_feature_stats,
    scaffold_split,
)


@dataclass
class FineTuneConfig:
    epochs: int = 100
    patience: int = 30
    batch_size: int = 32
    encoder_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-6
    head_dropout: float = 0.2
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PredictionSet:
    """Per-molecule test-set predictions, in original target units."""

    dataset: str
    model: str
    seed: int
    smiles: List[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    split: List[str]
    task: str
    valid_score: float
    epochs_run: int
    gate: Dict[str, float] = field(default_factory=dict)


def _masked_loss(outputs, targets, task: str):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return outputs.sum() * 0.0
    if task == "classification":
        return F.binary_cross_entropy_with_logits(outputs[mask], targets[mask])
    return F.mse_loss(outputs[mask], targets[mask])


@torch.no_grad()
def _collect(model, loader, device: str):
    model.eval()
    outputs, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        outputs.append(model(batch).cpu().numpy())
        targets.append(batch.y.view(batch.num_graphs, -1).float().cpu().numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def _primary_score(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> float:
    """Validation score for early stopping: ROC-AUC, or negative RMSE."""
    from sklearn.metrics import roc_auc_score

    scores = []
    for column in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, column])
        y, p = y_true[mask, column], y_pred[mask, column]
        if y.size == 0:
            continue
        if task == "classification":
            if np.unique(y).size < 2:
                continue
            scores.append(roc_auc_score(y, 1.0 / (1.0 + np.exp(-p))))
        else:
            scores.append(-float(np.sqrt(np.mean((y - p) ** 2))))
    return float(np.mean(scores)) if scores else float("-inf")


def finetune_and_predict(
    graphs: Sequence,
    task: str,
    checkpoint_path: Optional[str],
    stats_path: str,
    model_name: str,
    dataset_name: str,
    seed: int = 0,
    config: Optional[FineTuneConfig] = None,
    num_tasks: int = 1,
) -> PredictionSet:
    """Fine-tune on one benchmark and return test-set predictions."""
    config = config or FineTuneConfig()
    device = config.device

    torch.manual_seed(seed)
    np.random.seed(seed)

    smiles = [g.smiles for g in graphs]
    train_idx, valid_idx, test_idx = scaffold_split(smiles)

    # Targets are standardized for training using TRAIN statistics only. The
    # originals are kept so that both predictions and targets are reported in
    # original units.
    original_y = np.stack([g.y.view(-1).numpy() for g in graphs])
    scaler = None
    if task == "regression":
        train_y = original_y[train_idx]
        mean = np.nanmean(train_y, axis=0)
        std = np.nanstd(train_y, axis=0)
        std[std < 1e-8] = 1.0
        scaler = (mean, std)
        for index, graph in enumerate(graphs):
            graph.y = torch.tensor(
                ((original_y[index] - mean) / std)[None, :], dtype=torch.float
            )

    loaders = [
        DataLoader(Subset(graphs, idx), batch_size=config.batch_size,
                   shuffle=(position == 0), drop_last=False)
        for position, idx in enumerate((train_idx, valid_idx, test_idx))
    ]

    stats = load_feature_stats(stats_path, device=device)
    if checkpoint_path is None:
        encoder = GraphEncoder(dropout=0.1)
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
        encoder = encoder.to(device)
    else:
        encoder, _ = load_encoder(checkpoint_path, device=device)

    model = DownstreamModel(encoder, num_tasks, stats, config.head_dropout).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.head.parameters(), "lr": config.head_lr},
            {"params": model.encoder.parameters(), "lr": config.encoder_lr},
        ],
        weight_decay=config.weight_decay,
    )

    best_valid, stalled, epoch = float("-inf"), 0, 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in loaders[0]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(
                model(batch),
                batch.y.view(batch.num_graphs, -1).float(),
                task,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        valid_pred, valid_true = _collect(model, loaders[1], device)
        if scaler is not None:
            valid_pred = valid_pred * scaler[1] + scaler[0]
            valid_true = valid_true * scaler[1] + scaler[0]
        score = _primary_score(valid_true, valid_pred, task)

        if score > best_valid and np.isfinite(score):
            best_valid, stalled = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stalled += 1
            if stalled >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predictions for every molecule, labelled by split. The test split is the
    # clean estimate and remains the primary analysis; the train and validation
    # molecules are included because a disparity measured over eight functional
    # groups on a 150-molecule test set is badly underpowered, and the question
    # being asked of them is about the model's input-output behaviour rather
    # than its generalization. Any use of the pooled set must say so.
    assignment = np.empty(len(graphs), dtype=object)
    for name, indices in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
        for index in indices:
            assignment[index] = name

    ordered = list(train_idx) + list(valid_idx) + list(test_idx)
    all_loader = DataLoader(Subset(graphs, ordered), batch_size=config.batch_size,
                            shuffle=False, drop_last=False)
    all_pred, all_true = _collect(model, all_loader, device)

    if scaler is not None:
        all_pred = all_pred * scaler[1] + scaler[0]
        all_true = all_true * scaler[1] + scaler[0]
        # Restore the originals so a later seed re-standardizes from raw values.
        for index, graph in enumerate(graphs):
            graph.y = torch.tensor(original_y[index][None, :], dtype=torch.float)
    else:
        all_pred = 1.0 / (1.0 + np.exp(-all_pred))  # probabilities

    split_labels = [assignment[i] for i in ordered]
    is_test = np.array([label == "test" for label in split_labels])

    return PredictionSet(
        dataset=dataset_name,
        model=model_name,
        seed=seed,
        smiles=[smiles[i] for i in ordered],
        y_true=all_true,
        y_pred=all_pred,
        split=split_labels,
        task=task,
        valid_score=float(best_valid),
        epochs_run=epoch,
        gate=performance_gate(all_true[is_test], all_pred[is_test], task),
    )


def performance_gate(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    """Signal check computed from predictions already in hand.

    No extra compute and no comparison against any previously published
    figure: the only question asked is whether the predictions carry signal
    at all, since a disparity measured on predictions that do not would be
    measuring noise. Classification passes above chance, regression passes
    when it beats the mean predictor.
    """
    from sklearn.metrics import r2_score, roc_auc_score

    out: Dict[str, float] = {}
    scores: List[float] = []
    for column in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, column])
        y, p = y_true[mask, column], y_pred[mask, column]
        if y.size < 3:
            continue
        if task == "classification":
            if np.unique(y).size < 2:
                continue
            scores.append(roc_auc_score(y, p))
        else:
            scores.append(r2_score(y, p))

    if task == "classification":
        out["roc_auc"] = float(np.mean(scores)) if scores else float("nan")
        out["passes"] = float(out["roc_auc"] > 0.5) if scores else 0.0
    else:
        out["r2"] = float(np.mean(scores)) if scores else float("nan")
        rmse = float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))
        baseline = float(np.nanstd(y_true))
        out["rmse"] = rmse
        out["mean_predictor_rmse"] = baseline
        out["passes"] = float(rmse < baseline)
    return out


def train_downstream_model(
    graphs: Sequence,
    task: str,
    checkpoint_path: Optional[str],
    stats_path: str,
    seed: int = 0,
    config: Optional[FineTuneConfig] = None,
    num_tasks: int = 1,
):
    """Fine-tune and return the model itself, for attribution work.

    Same protocol as `finetune_and_predict`. That function returns predictions
    and discards the model; stage 4 needs the model, so this returns
    (model, stats, split_indices) and leaves the targets in original units.
    """
    config = config or FineTuneConfig()
    device = config.device
    torch.manual_seed(seed)
    np.random.seed(seed)

    smiles = [g.smiles for g in graphs]
    train_idx, valid_idx, test_idx = scaffold_split(smiles)

    original_y = np.stack([g.y.view(-1).numpy() for g in graphs])
    scaler = None
    if task == "regression":
        train_y = original_y[train_idx]
        mean = np.nanmean(train_y, axis=0)
        std = np.nanstd(train_y, axis=0)
        std[std < 1e-8] = 1.0
        scaler = (mean, std)
        for index, graph in enumerate(graphs):
            graph.y = torch.tensor(
                ((original_y[index] - mean) / std)[None, :], dtype=torch.float
            )

    loaders = [
        DataLoader(Subset(graphs, idx), batch_size=config.batch_size,
                   shuffle=(position == 0), drop_last=False)
        for position, idx in enumerate((train_idx, valid_idx, test_idx))
    ]

    stats = load_feature_stats(stats_path, device=device)
    encoder, _ = load_encoder(checkpoint_path, device=device)
    model = DownstreamModel(encoder, num_tasks, stats, config.head_dropout).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.head.parameters(), "lr": config.head_lr},
            {"params": model.encoder.parameters(), "lr": config.encoder_lr},
        ],
        weight_decay=config.weight_decay,
    )

    best_valid, stalled, best_state = float("-inf"), 0, None
    for _ in range(1, config.epochs + 1):
        model.train()
        for batch in loaders[0]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(
                model(batch), batch.y.view(batch.num_graphs, -1).float(), task
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        valid_pred, valid_true = _collect(model, loaders[1], device)
        if scaler is not None:
            valid_pred = valid_pred * scaler[1] + scaler[0]
            valid_true = valid_true * scaler[1] + scaler[0]
        score = _primary_score(valid_true, valid_pred, task)
        if score > best_valid and np.isfinite(score):
            best_valid, stalled = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stalled += 1
            if stalled >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    for index, graph in enumerate(graphs):
        graph.y = torch.tensor(original_y[index][None, :], dtype=torch.float)

    return model.eval(), stats, (train_idx, valid_idx, test_idx), float(best_valid)
