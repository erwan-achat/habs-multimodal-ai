import json
import random
from pathlib import Path

import pandas as pd


def subject_video_split(
    trial_index: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, set[str]]:
    subjects = sorted(trial_index["participant_id"].unique())
    videos = sorted(trial_index["video"].unique())

    subject_rng = random.Random(seed)
    video_rng = random.Random(seed + 1)
    subject_rng.shuffle(subjects)
    video_rng.shuffle(videos)
    subject_cut = max(1, min(len(subjects) - 1 if len(subjects) > 1 else 1,
                           round(len(subjects) * train_fraction)))
    video_cut = max(1, min(len(videos) - 1 if len(videos) > 1 else 1,
                           round(len(videos) * train_fraction)))
    return {
        "train_subjects": set(subjects[:subject_cut]),
        "test_subjects": set(subjects[subject_cut:]),
        "train_videos": set(videos[:video_cut]),
        "test_videos": set(videos[video_cut:]),
    }


def build_partitions(
    trial_index: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame | set[str]]:
    split = subject_video_split(trial_index, train_fraction=train_fraction, seed=seed)
    train_subjects = split["train_subjects"]
    test_subjects = split["test_subjects"]
    train_videos = split["train_videos"]
    test_videos = split["test_videos"]

    train = trial_index[
        trial_index["participant_id"].isin(train_subjects)
        & trial_index["video"].isin(train_videos)
    ].copy()

    pair_keys = sorted(set(zip(train["participant_id"], train["video"], strict=True)))
    pair_rng = random.Random(seed + 2)
    pair_rng.shuffle(pair_keys)
    validation_count = (
        max(1, round(len(pair_keys) * validation_fraction))
        if validation_fraction > 0.0
        else 0
    )
    validation_pairs = set(pair_keys[:validation_count])
    validation = train[
        [
            (participant, video) in validation_pairs
            for participant, video in zip(
                train["participant_id"],
                train["video"],
                strict=True,
            )
        ]
    ].copy()
    a = trial_index[
        trial_index["participant_id"].isin(test_subjects)
        & trial_index["video"].isin(train_videos)
    ].copy()
    b = trial_index[
        trial_index["participant_id"].isin(train_subjects)
        & trial_index["video"].isin(test_videos)
    ].copy()
    c = trial_index[
        trial_index["participant_id"].isin(test_subjects)
        & trial_index["video"].isin(test_videos)
    ].copy()

    return {
        "train": train,
        "validation": validation,
        "validation_pairs": validation_pairs,
        "A": a,
        "B": b,
        "C": c,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_videos": train_videos,
        "test_videos": test_videos,
    }


def save_split(split: dict, path: Path) -> None:
    serializable = {}
    for key, values in split.items():
        if isinstance(values, set):
            serializable[key] = sorted(values)
        else:
            serializable[key] = values
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
