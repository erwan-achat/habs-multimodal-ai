import json
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from disentanglement.config import PreprocessingConfig, TrainingConfig
from disentanglement.data_loading import build_trial_index, resolve_labels_path
from disentanglement.dataset import EmotionWindowDataset, RecordingGroupedSampler
from disentanglement.models import MultimodalDisentanglement
from disentanglement.normalization import fit_normalization
from disentanglement.splits import (
    build_subject_partitions,
    save_split,
    select_cohort_candidates,
)
from disentanglement.train import (
    collect_predictions,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_epoch,
)

SMOKE_MODE = os.environ.get("SMOKE_MODE", "0") == "1"


def _preprocessed_root(data_dir: Path) -> Path:
    candidate = data_dir / "PREPROCESSED"
    return candidate if candidate.is_dir() else data_dir


def _label_universe(labels: pd.DataFrame) -> list[str]:
    values = labels["final_emotion_label"].dropna().astype(str).str.strip()
    return sorted({value for value in values if value and value.upper() != "TESTSUBJECT"})


def _class_weights(labels: list[int], num_classes: int, device) -> tuple[torch.Tensor, list[float]]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    weights = np.ones(num_classes, dtype=float)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device), weights.tolist()


def _prepare_trial_index(labels, labels_path, preprocessing, data_dir, training, require_gender_balance):
    pool_size = training.subject_count if not require_gender_balance else training.subject_count * 2
    root = _preprocessed_root(data_dir)
    last_error: Exception | None = None
    for attempt in range(3):
        candidates = select_cohort_candidates(
            labels,
            root,
            pool_size=pool_size,
            seed=training.seed + attempt,
            require_gender_balance=require_gender_balance,
        )
        trial_index = build_trial_index(
            data_dir,
            labels_path,
            preprocessing,
            participant_ids=candidates,
        )
        if trial_index.empty:
            last_error = ValueError("No valid preprocessed emotion trials found in the candidate pool")
            pool_size *= 2
            continue
        try:
            partitions = build_subject_partitions(
                trial_index,
                labels,
                subject_count=training.subject_count,
                video_train_fraction=training.video_train_fraction,
                validation_fraction=training.validation_fraction,
                seed=training.seed,
                require_gender_balance=require_gender_balance,
            )
            return trial_index, partitions, candidates
        except ValueError as error:
            last_error = error
            pool_size *= 2
    raise last_error if last_error is not None else ValueError("Could not build partitions")


def _dataset(index, subjects, videos, label_to_id, normalization, preprocessing):
    return EmotionWindowDataset(
        index,
        subjects,
        videos,
        label_to_id,
        normalization,
        config=preprocessing,
    )


def _save_plots(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
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


def _save_confusion(labels, predictions, path: Path, title: str) -> dict:
    present = sorted(set(labels) | set(predictions))
    if not present:
        return {}
    matrix = confusion_matrix(labels, predictions, labels=present)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(present)), present, rotation=90)
    plt.yticks(range(len(present)), present)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return {
        "classes": present,
        "matrix": matrix.tolist(),
        "report": classification_report(
            labels,
            predictions,
            labels=present,
            zero_division=0,
            output_dict=True,
        ),
    }


def process(data_dir: Path, out_dir: Path) -> None:
    started = time.perf_counter()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = PreprocessingConfig()
    training = TrainingConfig()
    if SMOKE_MODE:
        training = replace(
            training,
            subject_count=2,
            epochs=1,
            validation_fraction=0.5,
            batch_size=8,
            max_normalization_trials=8,
            num_workers=0,
        )
    require_gender_balance = not SMOKE_MODE
    seed_everything(training.seed)

    labels_path = resolve_labels_path(data_dir, training.labels_filename)
    labels = pd.read_csv(labels_path)
    label_to_id = {emotion: index for index, emotion in enumerate(_label_universe(labels))}
    id_to_label = {index: emotion for emotion, index in label_to_id.items()}

    index_started = time.perf_counter()
    trial_index, partitions, candidates = _prepare_trial_index(
        labels,
        labels_path,
        preprocessing,
        data_dir,
        training,
        require_gender_balance,
    )
    index_seconds = time.perf_counter() - index_started
    trial_index.to_csv(out_dir / "trial_index.csv", index=False)

    save_split(
        {
            "train_subjects": set(partitions["train_subjects"]),
            "validation_subjects": set(partitions["validation_subjects"]),
            "train_videos": set(partitions["train_videos"]),
            "validation_videos": set(partitions["validation_videos"]),
            "train_genders": partitions["train_genders"],
            "validation_genders": partitions["validation_genders"],
            "candidates": set(candidates),
            "smoke_mode": SMOKE_MODE,
        },
        out_dir / "split.json",
    )

    train_subjects = set(partitions["train_subjects"])
    validation_subjects = set(partitions["validation_subjects"])
    train_videos = set(partitions["train_videos"])
    validation_videos = set(partitions["validation_videos"])
    train_index = partitions["train"]
    validation_index = partitions["validation"]
    if train_index.empty:
        raise ValueError("No training trials found for selected subjects")
    if validation_index.empty:
        raise ValueError("No validation trials found for selected subjects")

    normalization_started = time.perf_counter()
    normalization = fit_normalization(
        train_index,
        train_subjects,
        train_videos,
        max_trials=training.max_normalization_trials,
        seed=training.seed,
        config=preprocessing,
    )
    normalization_seconds = time.perf_counter() - normalization_started
    np.savez(
        out_dir / "normalization.npz",
        **{
            f"{name}_{kind}": value
            for name, (mean, std) in normalization.items()
            for kind, value in (("mean", mean), ("std", std))
        },
    )

    train_dataset = _dataset(train_index, train_subjects, train_videos, label_to_id, normalization, preprocessing)
    validation_dataset = _dataset(
        validation_index,
        validation_subjects,
        validation_videos,
        label_to_id,
        normalization,
        preprocessing,
    )
    if not train_dataset.items:
        raise ValueError("No training windows were created")
    if not validation_dataset.items:
        raise ValueError("No validation windows were created")

    train_sampler = RecordingGroupedSampler(train_dataset.items, seed=training.seed)
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
    class_weights, class_weight_values = _class_weights(
        [item["label"] for item in train_dataset.items],
        len(label_to_id),
        device,
    )
    eeg_channels = int(train_index["n_eeg_channels"].mode().iloc[0])
    model = MultimodalDisentanglement(
        num_classes=len(label_to_id),
        eeg_channels=eeg_channels,
        modalities=preprocessing.modalities,
        dropout=training.dropout,
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
    weights = {"alpha": training.alpha, "beta": training.beta, "gamma": training.gamma}

    history: list[dict] = []
    best_metric = -float("inf")
    best_epoch = 0
    training_seconds = 0.0
    for epoch in range(training.epochs):
        epoch_started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights,
            class_weights=class_weights,
            label_smoothing=training.label_smoothing,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            num_classes=len(label_to_id),
        )
        training_seconds += time.perf_counter() - epoch_started
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

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, epoch + 1, out_dir / "best_model.pt", scheduler)
        print(f"epoch {epoch + 1}: {row}")

    save_checkpoint(model, optimizer, training.epochs, out_dir / "final_model.pt", scheduler)
    if (out_dir / "best_model.pt").is_file():
        best_epoch = load_checkpoint(model, out_dir / "best_model.pt", device, optimizer, scheduler)

    evaluation: dict[str, dict] = {}
    evaluation["C"] = evaluate(model, validation_loader, device, num_classes=len(label_to_id))
    validation_index.to_csv(out_dir / "partition_C.csv", index=False)

    for partition_name in ("A", "B"):
        partition_index = partitions[partition_name]
        partition_subjects = validation_subjects if partition_name == "A" else train_subjects
        partition_videos = train_videos if partition_name == "A" else validation_videos
        partition_dataset = _dataset(
            partition_index,
            partition_subjects,
            partition_videos,
            label_to_id,
            normalization,
            preprocessing,
        )
        partition_loader = DataLoader(
            partition_dataset,
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
        )
        evaluation[partition_name] = evaluate(
            model,
            partition_loader,
            device,
            num_classes=len(label_to_id),
        )
        partition_index.to_csv(out_dir / f"partition_{partition_name}.csv", index=False)

    labels_c, predictions_c, _ = collect_predictions(model, validation_loader, device)
    confusion = _save_confusion(
        labels_c,
        predictions_c,
        out_dir / "confusion_matrix_C.png",
        "Validation C",
    )
    (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (out_dir / "confusion_C.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out_dir / "history.csv", index=False)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "eeg_channels": eeg_channels,
                "modalities": list(preprocessing.modalities),
                "feature_dim": 128,
                "num_classes": len(label_to_id),
                "dropout": training.dropout,
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

    run_report = {
        "smoke_mode": SMOKE_MODE,
        "require_gender_balance": require_gender_balance,
        "device": str(device),
        "candidates": sorted(candidates),
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(validation_subjects),
        "seed": training.seed,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "primary_metric": training.primary_metric,
        "class_weights": dict(zip(label_to_id, class_weight_values)),
        "counts": {
            "candidate_trials": len(trial_index),
            "train_trials": len(train_index),
            "train_windows": len(train_dataset),
            "validation_trials": len(validation_index),
            "validation_windows": len(validation_dataset),
        },
        "timings_seconds": {
            "index": index_seconds,
            "normalization": normalization_seconds,
            "training": training_seconds,
            "total": time.perf_counter() - started,
        },
    }
    (out_dir / "run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print(
        f"{len(trial_index)} candidate trials; "
        f"{len(train_dataset)} train windows; "
        f"{len(validation_dataset)} validation windows; "
        f"best epoch {best_epoch}; "
        f"C {evaluation['C']}"
    )


if __name__ == "__main__":
    process(
        Path(os.environ.get("HABS_DATA_DIR", "sample")),
        Path(os.environ.get("HABS_OUT_DIR", "out")),
    )
