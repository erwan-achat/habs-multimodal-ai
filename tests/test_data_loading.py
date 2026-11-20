import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from disentanglement.data_loading import build_trial_index
from disentanglement.dataset import (
    EpochTrialSubsetSampler,
    RecordingGroupedSampler,
)
from disentanglement.preprocessing import window_offsets
from disentanglement.splits import build_partitions


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
        {"bdf_path": "P001.bdf", "trial_start_s": 10.0},
        {"bdf_path": "P001.bdf", "trial_start_s": 10.0},
        {"bdf_path": "P002.bdf", "trial_start_s": 20.0},
    ]
    sampler = RecordingGroupedSampler(items, seed=42)
    order = list(sampler)

    assert sorted(order) == [0, 1, 2]
    assert order.index(0) + 1 == order.index(1) or order.index(1) + 1 == order.index(0)


def test_epoch_trial_sampler_resamples_complete_trials_and_changes_between_epochs() -> None:
    items = []
    for trial in range(6):
        for window in range(4):
            items.append({
                "bdf_path": f"P{trial // 2}.bdf",
                "trial_start_s": float(trial),
                "trial_duration_s": 10.0,
            })

    sampler = EpochTrialSubsetSampler(items, trials_per_epoch=3, seed=42)
    sampler.set_epoch(0)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    selected_zero = {
        items[index]["trial_start_s"]
        for index in epoch_zero
    }
    selected_one = {
        items[index]["trial_start_s"]
        for index in epoch_one
    }
    assert len(selected_zero) == 3
    assert len(selected_one) == 3
    assert selected_zero != selected_one
    assert len(epoch_zero) == 12


def test_subject_video_partitions_are_disjoint() -> None:
    trial_index = pd.DataFrame({
        "participant_id": ["S1", "S1", "S2", "S2"],
        "video": ["V1", "V2", "V1", "V2"],
    })
    partitions = build_partitions(trial_index, train_fraction=0.5, seed=42)

    train = partitions["train"]
    a = partitions["A"]
    b = partitions["B"]
    c = partitions["C"]
    train_subjects = set(train["participant_id"])
    train_videos = set(train["video"])
    assert set(a["video"]) <= train_videos
    assert set(a["participant_id"]).isdisjoint(train_subjects)
    assert set(b["participant_id"]) <= train_subjects
    assert set(b["video"]).isdisjoint(train_videos)
    assert set(c["participant_id"]).isdisjoint(train_subjects)
    assert set(c["video"]).isdisjoint(train_videos)
