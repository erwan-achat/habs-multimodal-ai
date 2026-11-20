from pathlib import Path

import pandas as pd

from disentanglement.data_loading import build_trial_index
from disentanglement.preprocessing import window_offsets
from disentanglement.splits import build_partitions
from disentanglement.dataset import RecordingGroupedSampler
import torch


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
        ]
    ).to_csv(labels, index=False)

    preprocessed = tmp_path / "PREPROCESSED"
    metadata_dir = preprocessed / "P001"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "P001_day-1_metadata.json").write_text(
        '{"dataset_export":{"participant_id":"P001","day":1},'
        '"events":[{"marker_id":10100,"onset_s":12.5,"duration_s":10.0,'
        '"trial_type":"emotion_video","media_file":"keep.mp4"}]}',
        encoding="utf-8",
    )

    index = build_trial_index(tmp_path, labels)

    assert len(index) == 1
    assert index.iloc[0]["participant_id"] == "P001"
    assert index.iloc[0]["video"] == "keep.mp4"
    assert index.iloc[0]["start_s"] == 12.5


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


def test_subject_video_partitions_are_disjoint() -> None:
    import pandas as pd

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
