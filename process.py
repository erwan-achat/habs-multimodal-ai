import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from disentanglement.config import PreprocessingConfig, TrainingConfig
from disentanglement.data_loading import build_trial_index, resolve_labels_path
from disentanglement.dataset import EmotionWindowDataset
from disentanglement.losses import total_loss
from disentanglement.models import MultimodalDisentanglement
from disentanglement.normalization import fit_normalization
from disentanglement.train import seed_everything

OVERFIT_PARTICIPANT = os.environ.get("OVERFIT_PARTICIPANT", "P001")
OVERFIT_MAX_TRIALS = 4
OVERFIT_MAX_WINDOWS = 32
OVERFIT_EPOCHS = 300
OVERFIT_BATCH_SIZE = 8
OVERFIT_TARGET_LOSS = 1e-3


def _label_universe(labels: pd.DataFrame) -> list[str]:
    values = labels["final_emotion_label"].dropna().astype(str).str.strip()
    return sorted({value for value in values if value and value.upper() != "TESTSUBJECT"})


def _select_participant(labels: pd.DataFrame, requested: str) -> str:
    available = sorted(labels["participant_id"].dropna().astype(str).unique())
    if not available:
        raise ValueError("labels_train.csv has no participant_id values")
    return requested if requested in available else available[0]


def process(data_dir: Path, out_dir: Path) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = PreprocessingConfig()
    training = TrainingConfig()
    seed_everything(training.seed)

    labels_path = resolve_labels_path(data_dir, training.labels_filename)
    labels = pd.read_csv(labels_path)
    label_to_id = {emotion: index for index, emotion in enumerate(_label_universe(labels))}

    participant = _select_participant(labels, OVERFIT_PARTICIPANT)
    trial_index = build_trial_index(
        data_dir,
        labels_path,
        preprocessing,
        participant_ids=[participant],
    )
    if trial_index.empty:
        raise ValueError(f"No valid preprocessed emotion trials found for {participant}")

    emotion = trial_index["final_emotion_label"].astype(str).value_counts().idxmax()
    subset = trial_index[
        trial_index["final_emotion_label"].astype(str) == emotion
    ].head(OVERFIT_MAX_TRIALS).copy()
    videos = set(subset["video"].astype(str))
    normalization = fit_normalization(
        subset,
        {participant},
        videos,
        max_trials=len(subset),
        seed=training.seed,
        config=preprocessing,
    )

    dataset = EmotionWindowDataset(
        subset,
        {participant},
        videos,
        label_to_id,
        normalization,
        config=preprocessing,
    )
    dataset.items = dataset.items[:OVERFIT_MAX_WINDOWS]
    if not dataset.items:
        raise ValueError("No windows were created for the overfit test")

    loader = DataLoader(
        dataset,
        batch_size=min(OVERFIT_BATCH_SIZE, len(dataset)),
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_channels = int(subset["n_eeg_channels"].mode().iloc[0])
    model = MultimodalDisentanglement(
        num_classes=len(label_to_id),
        eeg_channels=eeg_channels,
        modalities=preprocessing.modalities,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    weights = {
        "alpha": training.alpha,
        "beta": training.beta,
        "gamma": training.gamma,
    }

    print(
        f"overfit test: emotion={emotion!r} participant={participant} "
        f"trials={len(subset)} windows={len(dataset)} device={device}"
    )

    train_loss = float("nan")
    train_accuracy = 0.0
    for epoch in range(1, OVERFIT_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in model.modalities}
            targets = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            components = model(inputs, return_components=True)
            loss, values = total_loss(components, targets, weights)
            loss.backward()
            optimizer.step()
            loss_sum += values["task"] * len(targets)
            correct += (components["logits"].argmax(dim=-1) == targets).sum().item()
            total += len(targets)

        train_loss = loss_sum / total
        train_accuracy = correct / total
        if epoch == 1 or epoch % 10 == 0 or train_accuracy == 1.0:
            print(f"epoch {epoch:3d}: task_loss={train_loss:.6f} accuracy={train_accuracy:.3f}")
        if train_loss < OVERFIT_TARGET_LOSS and train_accuracy == 1.0:
            print(f"overfit OK at epoch {epoch}")
            return

    print(
        f"overfit incomplete after {OVERFIT_EPOCHS} epochs: "
        f"task_loss={train_loss:.6f} accuracy={train_accuracy:.3f}"
    )


if __name__ == "__main__":
    process(
        Path(os.environ.get("HABS_DATA_DIR", "sample")),
        Path(os.environ.get("HABS_OUT_DIR", "out")),
    )
