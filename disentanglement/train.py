import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from disentanglement.losses import total_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader: DataLoader, optimizer, device, weights: dict[str, float]) -> dict[str, float]:
    model.train()
    history: dict[str, list[float]] = {}
    for batch in loader:
        inputs = {name: batch[name].to(device) for name in ("eeg", "gsr", "respiration")}
        targets = batch["label"].to(device)
        subjects = batch["subject"].to(device)

        optimizer.zero_grad(set_to_none=True)
        components = model(inputs, subjects, return_components=True)
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
    use_subject_ids: bool = True,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_values = []
    for batch in loader:
        inputs = {name: batch[name].to(device) for name in ("eeg", "gsr", "respiration")}
        targets = batch["label"].to(device)
        subjects = batch["subject"].to(device) if use_subject_ids else None
        logits = model(inputs, subjects)
        loss_values.append(F.cross_entropy(logits, targets).item())
        correct += (logits.argmax(dim=-1) == targets).sum().item()
        total += targets.numel()
    return {
        "accuracy": correct / max(total, 1),
        "loss": float(np.mean(loss_values)) if loss_values else 0.0,
    }


def save_checkpoint(model, optimizer, epoch: int, path: Path) -> None:
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
