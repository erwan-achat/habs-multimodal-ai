import json
from pathlib import Path

import pandas as pd


def subject_video_split(
    trial_index: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, set[str]]:
    """Split unique subjects and videos independently."""
    subjects = sorted(trial_index["participant_id"].unique())
    videos = sorted(trial_index["video"].unique())

    import random

    subject_rng = random.Random(seed)
    video_rng = random.Random(seed + 1)
    subject_rng.shuffle(subjects)
    video_rng.shuffle(videos)
    subject_cut = max(1, round(len(subjects) * train_fraction))
    video_cut = max(1, round(len(videos) * train_fraction))
    train_subjects = set(subjects[:subject_cut])
    test_subjects = set(subjects[subject_cut:])
    train_videos = set(videos[:video_cut])
    test_videos = set(videos[video_cut:])

    return {
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_videos": train_videos,
        "test_videos": test_videos,
    }


def save_split(split: dict[str, set[str]], path: Path) -> None:
    path.write_text(
        json.dumps({key: sorted(values) for key, values in split.items()}, indent=2),
        encoding="utf-8",
    )
