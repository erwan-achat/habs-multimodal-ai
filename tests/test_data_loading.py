from pathlib import Path

import pandas as pd

from disentanglement.data_loading import build_trial_index
from disentanglement.preprocessing import window_offsets


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
