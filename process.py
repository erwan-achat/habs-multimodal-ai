import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from disentanglement.config import PreprocessingConfig, TrainingConfig
from disentanglement.data_loading import build_trial_index, resolve_labels_path
from disentanglement.dataset import EmotionWindowDataset, RecordingGroupedSampler
from disentanglement.models import MultimodalDisentanglement
from disentanglement.normalization import fit_normalization
from disentanglement.splits import build_subject_partitions, save_split
from disentanglement.train import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


def _save_plots(history: list[dict], out_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    loss_keys = (
        "total",
        "task",
        "modality_similarity",
        "modality_difference",
        "modality_reconstruction",
    )
    plt.figure(figsize=(10, 6))
    for key in loss_keys:
        column = f"train_{key}"
        if column in history[0]:
            plt.plot(epochs, [row[column] for row in history], label=column)
    if "validation_loss" in history[0]:
        plt.plot(epochs, [row["validation_loss"] for row in history], label="validation_loss", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png")
    plt.close()

    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        column = f"validation_{key}"
        if column in history[0]:
            plt.plot(epochs, [row[column] for row in history], label=column)
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_metrics.png")
    plt.close()


def process(data_dir: Path, out_dir: Path) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = PreprocessingConfig()
    training = TrainingConfig()
    seed_everything(training.seed)

    labels = resolve_labels_path(data_dir, training.labels_filename)
    trial_index = build_trial_index(data_dir, labels, preprocessing)
    if trial_index.empty:
        raise ValueError("No valid preprocessed emotion trials were found")
    trial_index.to_csv(out_dir / "trial_index.csv", index=False)

    emotions = sorted(trial_index["final_emotion_label"].dropna().unique().tolist())
    label_to_id = {emotion: index for index, emotion in enumerate(emotions)}

    partitions = build_subject_partitions(
        trial_index,
        pd.read_csv(labels),
        subject_count=training.subject_count,
        video_train_fraction=training.video_train_fraction,
        validation_fraction=training.validation_fraction,
        seed=training.seed,
    )
    save_split(
        {
            "train_subjects": set(partitions["train_subjects"]),
            "validation_subjects": set(partitions["validation_subjects"]),
            "train_videos": set(partitions["train_videos"]),
            "validation_videos": set(partitions["validation_videos"]),
            "train_genders": partitions["train_genders"],
            "validation_genders": partitions["validation_genders"],
        },
        out_dir / "split.json",
    )

    training_subjects = set(partitions["train_subjects"])
    validation_subjects = set(partitions["validation_subjects"])
    training_videos = set(partitions["train_videos"])
    validation_videos = set(partitions["validation_videos"])
    train_index = partitions["train"]
    validation_index = partitions["validation"]
    if train_index.empty:
        raise ValueError("No training trials found for selected subjects")
    if validation_index.empty:
        raise ValueError("No validation trials found for selected subjects")

    normalization = fit_normalization(
        train_index,
        training_subjects,
        training_videos,
        max_trials=training.max_normalization_trials,
        seed=training.seed,
        config=preprocessing,
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
        normalization,
        config=preprocessing,
    )
    validation_dataset = EmotionWindowDataset(
        validation_index,
        validation_subjects,
        validation_videos,
        label_to_id,
        normalization,
        config=preprocessing,
    )
    if not train_dataset.items:
        raise ValueError("No training windows were created")
    if not validation_dataset.items:
        raise ValueError("No validation windows were created")

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
    eeg_channels = int(train_index["n_eeg_channels"].mode().iloc[0])
    model = MultimodalDisentanglement(
        num_classes=len(label_to_id),
        eeg_channels=eeg_channels,
        modalities=preprocessing.modalities,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training.lr_scheduler_factor,
        patience=training.lr_scheduler_patience,
        min_lr=training.min_learning_rate,
    )
    weights = {
        "alpha": training.alpha,
        "beta": training.beta,
        "gamma": training.gamma,
    }

    history = []
    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(training.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, device, weights)
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            num_classes=len(label_to_id),
        )
        metric = validation_metrics[training.primary_metric]
        row = {
            "epoch": epoch + 1,
            "device": str(device),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        scheduler.step(metric)

        if metric > best_metric + training.min_delta:
            best_metric = metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                out_dir / "best_model.pt",
                scheduler,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            out_dir / "final_model.pt",
            scheduler,
        )
        print(f"epoch {epoch + 1}: {row}")
        if epochs_without_improvement >= training.patience:
            break

    if (out_dir / "best_model.pt").is_file():
        best_epoch = load_checkpoint(
            model,
            out_dir / "best_model.pt",
            device,
            optimizer,
            scheduler,
        )

    evaluation_metrics = {
        "C": evaluate(
            model,
            validation_loader,
            device,
            num_classes=len(label_to_id),
        )
    }
    validation_index.to_csv(out_dir / "partition_C.csv", index=False)
    for partition_name in ("A", "B"):
        partition_index = partitions[partition_name]
        partition_subjects = (
            validation_subjects if partition_name == "A" else training_subjects
        )
        partition_videos = (
            training_videos if partition_name == "A" else validation_videos
        )
        partition_dataset = EmotionWindowDataset(
            partition_index,
            partition_subjects,
            partition_videos,
            label_to_id,
            normalization,
            config=preprocessing,
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
            num_classes=len(label_to_id),
        )
        partition_index.to_csv(
            out_dir / f"partition_{partition_name}.csv",
            index=False,
        )

    (out_dir / "evaluation.json").write_text(
        json.dumps(evaluation_metrics, indent=2),
        encoding="utf-8",
    )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out_dir / "history.csv", index=False)
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (out_dir / "labels.json").write_text(
        json.dumps(label_to_id, indent=2),
        encoding="utf-8",
    )
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "eeg_channels": eeg_channels,
                "modalities": list(preprocessing.modalities),
                "feature_dim": 128,
                "num_classes": len(label_to_id),
                "best_epoch": best_epoch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "training_config.json").write_text(
        json.dumps(training.__dict__, indent=2),
        encoding="utf-8",
    )
    (out_dir / "preprocessing_config.json").write_text(
        json.dumps(preprocessing.__dict__, indent=2),
        encoding="utf-8",
    )
    _save_plots(history, out_dir)

    print(
        f"{len(trial_index)} valid trials; "
        f"{len(train_dataset)} train windows; "
        f"{len(validation_dataset)} validation windows; "
        f"best epoch {best_epoch}"
    )
