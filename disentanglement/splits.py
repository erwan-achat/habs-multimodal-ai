import json
import random
from pathlib import Path

import pandas as pd


def gender_by_subject(demographics: pd.DataFrame) -> dict[str, str]:
    if not {"participant_id", "gender"}.issubset(demographics.columns):
        raise ValueError("Demographics require participant_id and gender columns")
    rows = demographics[["participant_id", "gender"]].dropna().copy()
    rows["participant_id"] = rows["participant_id"].astype(str)
    rows["gender"] = rows["gender"].astype(str).str.strip().str.lower()
    result: dict[str, str] = {}
    for participant, group in rows.groupby("participant_id"):
        genders = set(group["gender"])
        if len(genders) == 1 and next(iter(genders)) in {"female", "male"}:
            result[participant] = next(iter(genders))
    return result


def select_cohort_candidates(
    demographics: pd.DataFrame,
    preprocessed_root: Path,
    pool_size: int,
    seed: int = 42,
    require_gender_balance: bool = True,
) -> set[str]:
    genders = gender_by_subject(demographics)
    root = Path(preprocessed_root)
    available_by_gender = {
        gender: sorted(
            participant
            for participant, value in genders.items()
            if value == gender and (root / participant).is_dir()
        )
        for gender in ("female", "male")
    }
    rng = random.Random(seed)
    if require_gender_balance:
        per_gender = max(1, pool_size // 2)
        for gender, subjects in available_by_gender.items():
            if len(subjects) < per_gender:
                raise ValueError(
                    f"Need {per_gender} candidate subjects of gender {gender}; "
                    f"found {len(subjects)} under {root}"
                )
        selected: set[str] = set()
        for subjects in available_by_gender.values():
            selected.update(rng.sample(subjects, per_gender))
        return selected

    available = sorted(set().union(*available_by_gender.values()))
    if not available:
        raise ValueError(f"No participants with gender found under {root}")
    return set(rng.sample(available, min(pool_size, len(available))))


def build_subject_partitions(
    trial_index: pd.DataFrame,
    demographics: pd.DataFrame,
    subject_count: int = 20,
    video_train_fraction: float = 0.8,
    validation_fraction: float = 0.2,
    seed: int = 42,
    require_gender_balance: bool = True,
) -> dict[str, pd.DataFrame | set[str] | dict[str, str]]:
    if subject_count <= 0 or (require_gender_balance and subject_count % 2):
        raise ValueError("subject_count must be a positive even number")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < video_train_fraction < 1.0:
        raise ValueError("video_train_fraction must be between 0 and 1")

    genders = gender_by_subject(demographics)
    eligible_subjects = set(trial_index["participant_id"].astype(str))
    available_by_gender = {
        gender: sorted(
            participant for participant, value in genders.items()
            if value == gender and participant in eligible_subjects
        )
        for gender in ("female", "male")
    }
    rng = random.Random(seed)
    if require_gender_balance:
        per_gender = subject_count // 2
        if any(len(subjects) < per_gender for subjects in available_by_gender.values()):
            raise ValueError(
                f"Need {per_gender} valid subjects of each gender; "
                f"found female={len(available_by_gender['female'])}, "
                f"male={len(available_by_gender['male'])}"
            )
        validation_per_gender = round(per_gender * validation_fraction)
        if validation_per_gender < 1 or validation_per_gender >= per_gender:
            raise ValueError("validation_fraction must leave train and validation subjects per gender")
        train_subjects: set[str] = set()
        validation_subjects: set[str] = set()
        for subjects in available_by_gender.values():
            selected = rng.sample(subjects, per_gender)
            rng.shuffle(selected)
            validation_subjects.update(selected[:validation_per_gender])
            train_subjects.update(selected[validation_per_gender:])
    else:
        available = sorted(set().union(*available_by_gender.values()))
        if len(available) < subject_count:
            raise ValueError(
                f"Need {subject_count} valid subjects; found {len(available)}"
            )
        selected = rng.sample(available, subject_count)
        validation_count = round(len(selected) * validation_fraction)
        if validation_count < 1 or validation_count >= len(selected):
            raise ValueError("validation_fraction must leave train and validation subjects")
        rng.shuffle(selected)
        validation_subjects = set(selected[:validation_count])
        train_subjects = set(selected[validation_count:])

    participant_ids = trial_index["participant_id"].astype(str)
    cohort_trials = trial_index.loc[
        participant_ids.isin(train_subjects | validation_subjects)
    ].copy()
    videos = sorted(cohort_trials["video"].astype(str).unique())
    if len(videos) < 2:
        raise ValueError("At least two distinct videos are required for a video split")
    random.Random(seed + 1).shuffle(videos)
    video_cut = max(1, min(len(videos) - 1, round(len(videos) * video_train_fraction)))
    train_videos = set(videos[:video_cut])
    validation_videos = set(videos[video_cut:])

    subject_ids = cohort_trials["participant_id"].astype(str)
    video_ids = cohort_trials["video"].astype(str)
    train = cohort_trials.loc[
        subject_ids.isin(train_subjects) & video_ids.isin(train_videos)
    ].copy()
    validation = cohort_trials.loc[
        subject_ids.isin(validation_subjects) & video_ids.isin(validation_videos)
    ].copy()
    a = cohort_trials.loc[
        subject_ids.isin(validation_subjects) & video_ids.isin(train_videos)
    ].copy()
    b = cohort_trials.loc[
        subject_ids.isin(train_subjects) & video_ids.isin(validation_videos)
    ].copy()
    c = validation.copy()
    return {
        "train": train,
        "validation": validation,
        "A": a,
        "B": b,
        "C": c,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "train_videos": train_videos,
        "validation_videos": validation_videos,
        "train_genders": {
            subject: genders[subject] for subject in sorted(train_subjects)
        },
        "validation_genders": {
            subject: genders[subject] for subject in sorted(validation_subjects)
        },
    }


def save_split(split: dict, path: Path) -> None:
    serializable = {}
    for key, values in split.items():
        if isinstance(values, set):
            serializable[key] = sorted(values)
        else:
            serializable[key] = values
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
