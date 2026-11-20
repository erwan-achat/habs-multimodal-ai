import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from disentanglement.config import PreprocessingConfig, TrainingConfig
from disentanglement.data_loading import build_trial_index
from disentanglement.dataset import EmotionWindowDataset
from disentanglement.models import MultimodalDisentanglement
from disentanglement.normalization import fit_normalization
from disentanglement.splits import build_partitions, save_split
from disentanglement.dataset import RecordingGroupedSampler
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

    partitions = build_partitions(trial_index, train_fraction=0.8, seed=training.seed)
    save_split(
        {
            "train_subjects": set(partitions["train_subjects"]),
            "test_subjects": set(partitions["test_subjects"]),
            "train_videos": set(partitions["train_videos"]),
            "test_videos": set(partitions["test_videos"]),
        },
        out_dir / "split.json",
    )

    training_subjects = set(partitions["train_subjects"])
    training_videos = set(partitions["train_videos"])
    private_subjects = training_subjects
    validation_index = partitions["validation"]

    subject_to_id = {
        subject: index
        for index, subject in enumerate(sorted(private_subjects))
    }

    train_index = partitions["train"].copy()
    validation_pairs = set(
        zip(validation_index["participant_id"], validation_index["video"], strict=True)
    )
    train_index = train_index[
        [
            (participant, video) not in validation_pairs
            for participant, video in zip(
                train_index["participant_id"],
                train_index["video"],
                strict=True,
            )
        ]
    ].copy()

    normalization = fit_normalization(
        train_index,
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
        train_index,
        training_subjects,
        training_videos,
        label_to_id,
        subject_to_id,
        normalization,
        max_trials=0,
    )
    validation_dataset = EmotionWindowDataset(
        validation_index,
        set(validation_index["participant_id"].unique()),
        set(validation_index["video"].unique()),
        label_to_id,
        subject_to_id,
        normalization,
    )

    train_sampler = RecordingGroupedSampler(
        train_dataset.items,
        seed=training.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        sampler=train_sampler,
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

    evaluation_metrics = {}
    for partition_name in ("A", "B", "C"):
        partition_index = partitions[partition_name]
        partition_subjects = set(partition_index["participant_id"].unique())
        partition_videos = set(partition_index["video"].unique())
        partition_dataset = EmotionWindowDataset(
            partition_index,
            partition_subjects,
            partition_videos,
            label_to_id,
            subject_to_id,
            normalization,
        )
        partition_loader = DataLoader(
            partition_dataset,
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
        )
        evaluation_metrics[partition_name] = evaluate(
            model,
            partition_loader,
            device,
            use_subject_ids=partition_name == "B",
        )
        partition_index.to_csv(
            out_dir / f"partition_{partition_name}.csv",
            index=False,
        )
    (out_dir / "evaluation.json").write_text(
        json.dumps(evaluation_metrics, indent=2),
        encoding="utf-8",
    )

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
