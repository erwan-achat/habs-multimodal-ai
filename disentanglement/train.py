import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from disentanglement.losses import total_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader: DataLoader, optimizer, device, weights: dict[str, float]) -> dict[str, float]:
    model.train()
    history: dict[str, list[float]] = {}
    for batch in loader:
        inputs = {name: batch[name].to(device) for name in model.modalities}
        targets = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        components = model(inputs, return_components=True)
        loss, values = total_loss(components, targets, weights)
        loss.backward()
        optimizer.step()

        for name, value in values.items():
            history.setdefault(name, []).append(value)
    return {name: float(np.mean(values)) for name, values in history.items()}


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    loss_values = []
    for batch in loader:
        inputs = {name: batch[name].to(device) for name in model.modalities}
        targets = batch["label"].to(device)
        logits = model(inputs)
        loss_values.append(F.cross_entropy(logits, targets).item())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(targets.cpu().tolist())

    if not labels:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    class_ids = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )
    return {
        "loss": float(np.mean(loss_values)) if loss_values else 0.0,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    path: Path,
    scheduler=None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    model,
    path: Path,
    device,
    optimizer=None,
    scheduler=None,
) -> int:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    return int(state.get("epoch", 0))
