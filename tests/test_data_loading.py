import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from disentanglement.data_loading import build_trial_index
from disentanglement.dataset import RecordingGroupedSampler
from disentanglement.preprocessing import window_offsets
from disentanglement.splits import (
    build_subject_partitions,
    select_cohort_candidates,
)


def _write_record_map(path: Path, n_eeg: int = 62) -> None:
    records = {
        "eeg": {
            "channel_ranges": [0, n_eeg - 1],
            "correction": {
                "RecordType": "eeg",
                "Channels": [{"name": f"E{index}"} for index in range(n_eeg)],
            },
        },
        "ECG": {
            "channel_indices": [n_eeg],
            "correction": {"RecordType": "ECG", "Channels": [{"name": "ECG"}]},
        },
        "RESP": {
            "channel_indices": [n_eeg + 1],
            "correction": {"RecordType": "Respiration", "Channels": [{"name": "Resp"}]},
        },
        "GSR": {
            "channel_indices": [n_eeg + 2],
            "correction": {"RecordType": "GSR", "Channels": [{"name": "GSR"}]},
        },
    }
    path.write_text(json.dumps({"stream_map": {"0": records}}), encoding="utf-8")


def test_window_offsets_cover_a_ten_second_trial() -> None:
    assert window_offsets(duration_s=10.0, window_s=3.0, stride_s=2.0) == [
        (0.0, 3.0),
        (2.0, 5.0),
        (4.0, 7.0),
        (6.0, 9.0),
    ]


def test_trial_index_filters_outliers_and_reads_metadata(tmp_path: Path) -> None:
    labels = tmp_path / "labels_train.csv"
    pd.DataFrame(
        [
            {
                "participant_id": "P001",
                "day": 1,
                "video": "keep.mp4",
                "start_marker": 10100,
                "final_emotion_label": "joy",
                "flagged_outliers": 0,
            },
            {
                "participant_id": "P001",
                "day": 1,
                "video": "drop.mp4",
                "start_marker": 10200,
                "final_emotion_label": "fear",
                "flagged_outliers": 1,
            },
            {
                "participant_id": "P001",
                "day": 1,
                "video": "sentinel.mp4",
                "start_marker": 10300,
                "final_emotion_label": "anger",
                "flagged_outliers": "TESTSUBJECT",
            },
        ]
    ).to_csv(labels, index=False)

    preprocessed = tmp_path / "PREPROCESSED"
    metadata_dir = preprocessed / "P001"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "P001_day-1_metadata.json").write_text(
        json.dumps({
            "dataset_export": {
                "participant_id": "P001",
                "day": 1,
                "segments": {
                    "emotion-video": {
                        "file": "P001_day-1_task-emotion-video_multimodal.bdf",
                        "metadata_time_origin_s": 0.0,
                        "duration_s": 30.0,
                    },
                },
            },
            "events": [
                {
                    "marker_id": 10100,
                    "onset_s": 12.5,
                    "duration_s": 10.0,
                    "trial_type": "emotion_video",
                    "media_file": "keep.mp4",
                },
            ],
        }),
        encoding="utf-8",
    )
    bdf = metadata_dir / "P001_day-1_task-emotion-video_multimodal.bdf"
    bdf.write_bytes(b"")
    _write_record_map(bdf.with_suffix(".record_map"))

    index = build_trial_index(tmp_path, labels)

    assert len(index) == 1
    assert index.iloc[0]["participant_id"] == "P001"
    assert index.iloc[0]["video"] == "keep.mp4"
    assert index.iloc[0]["start_s"] == 12.5
    assert index.iloc[0]["n_eeg_channels"] == 62


def test_trial_index_rejects_segments_outside_metadata(tmp_path: Path) -> None:
    labels = tmp_path / "labels_train.csv"
    pd.DataFrame(
        [
            {
                "participant_id": "P001",
                "day": 1,
                "video": "late.mp4",
                "start_marker": 10100,
                "final_emotion_label": "joy",
                "flagged_outliers": 0,
            },
        ]
    ).to_csv(labels, index=False)

    preprocessed = tmp_path / "PREPROCESSED"
    metadata_dir = preprocessed / "P001"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "P001_day-1_metadata.json").write_text(
        json.dumps({
            "dataset_export": {
                "segments": {
                    "emotion-video": {
                        "file": "P001_day-1_task-emotion-video_multimodal.bdf",
                        "metadata_time_origin_s": 0.0,
                        "duration_s": 15.0,
                    },
                },
            },
            "events": [
                {
                    "marker_id": 10100,
                    "onset_s": 12.0,
                    "duration_s": 10.0,
                    "trial_type": "emotion_video",
                    "media_file": "late.mp4",
                },
            ],
        }),
        encoding="utf-8",
    )
    bdf = metadata_dir / "P001_day-1_task-emotion-video_multimodal.bdf"
    bdf.write_bytes(b"")
    _write_record_map(bdf.with_suffix(".record_map"))

    index = build_trial_index(tmp_path, labels)

    assert index.empty


def test_trial_index_uses_start_and_end_markers(tmp_path: Path, monkeypatch) -> None:
    labels = tmp_path / "labels_train.csv"
    pd.DataFrame([{
        "participant_id": "P001",
        "day": 1,
        "video": "keep.mp4",
        "start_marker": 10100,
        "final_emotion_label": "joy",
        "flagged_outliers": 0,
    }]).to_csv(labels, index=False)

    metadata_dir = tmp_path / "PREPROCESSED" / "P001"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "P001_day-1_metadata.json").write_text(
        json.dumps({
            "dataset_export": {
                "segments": {
                    "emotion-video": {
                        "file": "P001_day-1_task-emotion-video_multimodal.bdf",
                        "metadata_time_origin_s": 0.0,
                        "duration_s": 30.0,
                    },
                },
            },
            "events": [{
                "marker_id": 10100,
                "onset_s": 12.5,
                "duration_s": 10.0,
                "trial_type": "emotion_video",
                "media_file": "keep.mp4",
            }],
        }),
        encoding="utf-8",
    )
    bdf = metadata_dir / "P001_day-1_task-emotion-video_multimodal.bdf"
    bdf.write_bytes(b"")
    _write_record_map(bdf.with_suffix(".record_map"))

    raw = SimpleNamespace(
        annotations=SimpleNamespace(
            onset=np.array([12.5, 14.0]),
            description=np.array(["10100", "10101"]),
        ),
        first_time=0.0,
        n_times=30_000,
        info={"sfreq": 1000.0},
        close=lambda: None,
    )
    monkeypatch.setattr(
        "mne.io.read_raw_bdf",
        lambda *args, **kwargs: raw,
    )

    index = build_trial_index(tmp_path, labels)

    assert len(index) == 1
    assert index.iloc[0]["start_s"] == 12.5
    assert index.iloc[0]["duration_s"] == 1.5


def test_trial_index_rejects_marker_segment_beyond_bdf(tmp_path: Path, monkeypatch) -> None:
    labels = tmp_path / "labels_train.csv"
    pd.DataFrame([{
        "participant_id": "P378",
        "day": 1,
        "video": "late.mp4",
        "start_marker": 11200,
        "final_emotion_label": "neutral",
        "flagged_outliers": 0,
    }]).to_csv(labels, index=False)

    metadata_dir = tmp_path / "PREPROCESSED" / "P378"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "P378_day-1_metadata.json").write_text(
        json.dumps({
            "dataset_export": {
                "segments": {
                    "emotion-video": {
                        "file": "P378_day-1_task-emotion-video_multimodal.bdf",
                        "metadata_time_origin_s": 0.0,
                        "duration_s": 978.0,
                    },
                },
            },
            "events": [{
                "marker_id": 11200,
                "onset_s": 968.81,
                "duration_s": 10.0,
                "trial_type": "emotion_video",
                "media_file": "late.mp4",
            }],
        }),
        encoding="utf-8",
    )
    bdf = metadata_dir / "P378_day-1_task-emotion-video_multimodal.bdf"
    bdf.write_bytes(b"")
    _write_record_map(bdf.with_suffix(".record_map"))

    raw = SimpleNamespace(
        annotations=SimpleNamespace(
            onset=np.array([968.81, 978.81]),
            description=np.array(["11200", "11201"]),
        ),
        first_time=0.0,
        n_times=977_000,
        info={"sfreq": 1000.0},
        close=lambda: None,
    )
    monkeypatch.setattr(
        "mne.io.read_raw_bdf",
        lambda *args, **kwargs: raw,
    )

    index = build_trial_index(tmp_path, labels)

    assert index.empty


def test_recording_grouped_sampler_preserves_full_window_coverage() -> None:
    items = [
        {"bdf_path": "P001.bdf", "trial_start_s": 10.0, "trial_duration_s": 10.0},
        {"bdf_path": "P001.bdf", "trial_start_s": 10.0, "trial_duration_s": 10.0},
        {"bdf_path": "P002.bdf", "trial_start_s": 20.0, "trial_duration_s": 10.0},
    ]
    sampler = RecordingGroupedSampler(items, seed=42)
    order = list(sampler)

    assert sorted(order) == [0, 1, 2]
    assert order.index(0) + 1 == order.index(1) or order.index(1) + 1 == order.index(0)


def test_select_cohort_candidates_filters_existing_preprocessed_dirs(tmp_path: Path) -> None:
    demographics = pd.DataFrame({
        "participant_id": ["P001", "P002", "P003", "P004"],
        "gender": ["female", "female", "male", "male"],
    })
    root = tmp_path / "PREPROCESSED"
    for participant in ("P001", "P002", "P003"):
        (root / participant).mkdir(parents=True)

    balanced = select_cohort_candidates(
        demographics,
        root,
        pool_size=2,
        seed=42,
        require_gender_balance=True,
    )
    assert len(balanced) == 2
    assert "P003" in balanced
    assert balanced & {"P001", "P002"}

    unbalanced = select_cohort_candidates(
        demographics,
        root,
        pool_size=4,
        seed=42,
        require_gender_balance=False,
    )
    assert unbalanced == {"P001", "P002", "P003"}


def test_partitions_support_unbalanced_gender_mode() -> None:
    demographics = pd.DataFrame({
        "participant_id": [f"P{i:03}" for i in range(6)],
        "gender": ["female"] * 6,
    })
    trial_index = pd.DataFrame({
        "participant_id": [f"P{i:03}" for i in range(6) for _ in range(10)],
        "video": [f"V{j:02}" for _ in range(6) for j in range(10)],
    })

    partitions = build_subject_partitions(
        trial_index,
        demographics,
        subject_count=2,
        validation_fraction=0.5,
        seed=42,
        require_gender_balance=False,
    )

    assert len(partitions["train_subjects"]) == 1
    assert len(partitions["validation_subjects"]) == 1
    assert set(partitions["train_genders"].values()) == {"female"}
    assert set(partitions["validation_genders"].values()) == {"female"}


def test_subject_video_partitions_are_balanced_disjoint_and_reproducible() -> None:
    demographics = pd.DataFrame({
        "participant_id": [f"P{i:03}" for i in range(40)],
        "gender": ["female"] * 20 + ["male"] * 20,
    })
    trial_index = pd.DataFrame({
        "participant_id": [f"P{i:03}" for i in range(40) for _ in range(10)],
        "video": [f"V{j:02}" for _ in range(40) for j in range(10)],
        "trial": list(range(400)),
    })

    partitions = build_subject_partitions(
        trial_index,
        demographics,
        subject_count=20,
        validation_fraction=0.2,
        seed=42,
    )
    repeated = build_subject_partitions(
        trial_index,
        demographics,
        subject_count=20,
        validation_fraction=0.2,
        seed=42,
    )

    train_subjects = set(partitions["train_subjects"])
    validation_subjects = set(partitions["validation_subjects"])
    assert len(train_subjects | validation_subjects) == 20
    assert len(train_subjects) == 16
    assert len(validation_subjects) == 4
    assert train_subjects.isdisjoint(validation_subjects)
    assert len(partitions["train_videos"]) == 8
    assert len(partitions["validation_videos"]) == 2
    assert set(partitions["train_videos"]).isdisjoint(partitions["validation_videos"])
    assert set(partitions["train"]["participant_id"]) == train_subjects
    assert set(partitions["train"]["video"]) == set(partitions["train_videos"])
    assert set(partitions["validation"]["participant_id"]) == validation_subjects
    assert set(partitions["validation"]["video"]) == set(partitions["validation_videos"])
    assert set(partitions["A"]["participant_id"]) == validation_subjects
    assert set(partitions["A"]["video"]) == set(partitions["train_videos"])
    assert set(partitions["B"]["participant_id"]) == train_subjects
    assert set(partitions["B"]["video"]) == set(partitions["validation_videos"])
    assert set(partitions["C"]["participant_id"]) == validation_subjects
    assert set(partitions["C"]["video"]) == set(partitions["validation_videos"])
    assert partitions["train_genders"] == repeated["train_genders"]
    assert train_subjects == set(repeated["train_subjects"])
    assert validation_subjects == set(repeated["validation_subjects"])
    assert partitions["train_videos"] == repeated["train_videos"]
    assert partitions["validation_videos"] == repeated["validation_videos"]
    assert sum(gender == "female" for gender in partitions["train_genders"].values()) == 8
    assert sum(gender == "male" for gender in partitions["train_genders"].values()) == 8
    assert sum(gender == "female" for gender in partitions["validation_genders"].values()) == 2
    assert sum(gender == "male" for gender in partitions["validation_genders"].values()) == 2
