import json
from pathlib import Path

import pandas as pd


def _preprocessed_root(data_dir: Path) -> Path:
    candidate = data_dir / "PREPROCESSED"
    return candidate if candidate.is_dir() else data_dir


def build_trial_index(
    data_dir: Path,
    labels_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build a lightweight index of valid emotion trials."""
    data_dir = Path(data_dir)
    if labels_path is None:
        labels_path = data_dir / "labels_train.csv"
        if not labels_path.is_file():
            labels_path = data_dir.parent / "labels_train.csv"
        if not labels_path.is_file():
            labels_path = data_dir.parent / "data" / "labels_train.csv"
    labels_path = Path(labels_path)

    labels = pd.read_csv(labels_path)
    required = {"participant_id", "day", "video", "start_marker", "final_emotion_label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"labels_train.csv missing columns: {sorted(missing)}")

    outlier_column = "flagged_outliers" if "flagged_outliers" in labels else "flagged_outlier"
    if outlier_column not in labels:
        raise ValueError("labels_train.csv has no flagged outlier column")
    labels = labels.loc[pd.to_numeric(labels[outlier_column], errors="coerce").fillna(0) != 1].copy()
    labels = labels.loc[
        labels["participant_id"].notna()
        & pd.to_numeric(labels["day"], errors="coerce").notna()
    ].copy()

    root = _preprocessed_root(data_dir)
    rows: list[dict] = []
    for row in labels.itertuples(index=False):
        participant = str(row.participant_id)
        day = int(float(row.day))
        metadata_path = root / participant / f"{participant}_day-{day}_metadata.json"
        if not metadata_path.is_file():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        segment = metadata.get("dataset_export", {}).get("segments", {}).get("emotion-video", {})
        origin = float(segment.get("metadata_time_origin_s", metadata.get("metadata_time_origin_s", 0.0)))
        events = metadata.get("events", [])
        marker = int(row.start_marker)
        matches = [
            event for event in events
            if int(event.get("marker_id", -1)) == marker
            and event.get("trial_type") == "emotion_video"
        ]
        if not matches:
            continue

        event = matches[0]
        media_file = event.get("media_file") or event.get("instruction_text")
        if media_file and str(media_file) != str(row.video):
            continue
        start_s = float(event["onset_s"]) - origin
        duration_s = float(event.get("duration_s", 10.0))
        if not 0.0 <= start_s:
            continue

        bdf_path = (
            root / participant
            / f"{participant}_day-{day}_task-emotion-video_multimodal.bdf"
        )

        rows.append({
            "participant_id": participant,
            "day": day,
            "video": str(row.video),
            "start_marker": marker,
            "final_emotion_label": str(row.final_emotion_label),
            "start_s": start_s,
            "duration_s": duration_s,
            "bdf_path": str(bdf_path),
        })

    return pd.DataFrame(rows)
