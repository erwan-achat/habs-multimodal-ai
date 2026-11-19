import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from disentanglement.config import PreprocessingConfig, TrainingConfig
from disentanglement.data_loading import build_trial_index
from disentanglement.dataset import EmotionWindowDataset
from disentanglement.models import MultimodalDisentanglement
from disentanglement.normalization import fit_normalization
from disentanglement.splits import save_split, subject_video_split
from disentanglement.train import evaluate, save_checkpoint, seed_everything, train_epoch


def process(data_dir: Path, out_dir: Path) -> None:
    """Run bounded emotion training and write all experiment artifacts."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = PreprocessingConfig()
    training = TrainingConfig()
    seed_everything(training.seed)

    labels = data_dir / "labels_train.csv"
    if not labels.is_file():
        labels = data_dir.parent / "labels_train.csv"
    if not labels.is_file():
        labels = data_dir.parent / "data" / "labels_train.csv"

    trial_index = build_trial_index(data_dir, labels)
    trial_index.to_csv(out_dir / "trial_index.csv", index=False)
    trial_index = trial_index[
        trial_index["final_emotion_label"].str.upper() != "TESTSUBJECT"
    ].copy()

    emotions = sorted(trial_index["final_emotion_label"].dropna().unique().tolist())
    label_to_id = {emotion: index for index, emotion in enumerate(emotions)}

    split = subject_video_split(trial_index, seed=training.seed)
    save_split(split, out_dir / "split.json")

    train_subjects = set(split["train_subjects"])
    train_videos = set(split["train_videos"])
    subject_rng = random.Random(training.seed)
    video_rng = random.Random(training.seed + 1)
    subjects = sorted(train_subjects)
    videos = sorted(train_videos)
    subject_rng.shuffle(subjects)
    video_rng.shuffle(videos)
    subject_validation_count = max(1, round(len(subjects) * 0.2))
    video_validation_count = max(1, round(len(videos) * 0.2))
    validation_subjects = set(subjects[:subject_validation_count])
    validation_videos = set(videos[:video_validation_count])
    training_subjects = train_subjects - validation_subjects
    training_videos = train_videos - validation_videos
    private_subjects = train_subjects

    subject_to_id = {
        subject: index
        for index, subject in enumerate(sorted(private_subjects))
    }
    normalization = fit_normalization(
        trial_index,
        training_subjects,
        training_videos,
        max_trials=64,
    )

    np.savez(
        out_dir / "normalization.npz",
        **{
            f"{name}_{kind}": value
            for name, (mean, std) in normalization.items()
            for kind, value in (("mean", mean), ("std", std))
        },
    )

    train_dataset = EmotionWindowDataset(
        trial_index,
        training_subjects,
        training_videos,
        label_to_id,
        subject_to_id,
        normalization,
        max_trials=training.max_trials_per_epoch,
    )
    validation_dataset = EmotionWindowDataset(
        trial_index,
        validation_subjects,
        validation_videos,
        label_to_id,
        subject_to_id,
        normalization,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_channels = 62
    model = MultimodalDisentanglement(
        num_classes=len(label_to_id),
        eeg_channels=eeg_channels,
        training_subjects=sorted(private_subjects),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    weights = {
        "alpha": training.alpha,
        "beta": training.beta,
        "gamma": training.gamma,
        "delta": training.delta,
        "epsilon": training.epsilon,
    }

    history = []
    best_accuracy = -1.0
    for epoch in range(training.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, weights)
        validation_metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch + 1,
            "device": str(device),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)

        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            save_checkpoint(model, optimizer, epoch + 1, out_dir / "best_model.pt")
        save_checkpoint(model, optimizer, epoch + 1, out_dir / "final_model.pt")
        print(f"epoch {epoch + 1}: {row}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "labels.json").write_text(
        json.dumps(label_to_id, indent=2), encoding="utf-8"
    )
    (out_dir / "training_config.json").write_text(
        json.dumps(training.__dict__, indent=2), encoding="utf-8"
    )
    (out_dir / "preprocessing_config.json").write_text(
        json.dumps(preprocessing.__dict__, indent=2), encoding="utf-8"
    )

    print(
        f"{len(trial_index)} valid trials; "
        f"{len(train_dataset)} train windows; "
        f"{len(validation_dataset)} validation windows"
    )
