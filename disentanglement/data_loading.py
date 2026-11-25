import json
from collections.abc import Iterable
from pathlib import Path

import mne
import pandas as pd

from disentanglement.config import PreprocessingConfig
from disentanglement.preprocessing import channel_names


def _preprocessed_root(data_dir: Path) -> Path:
    candidate = data_dir / "PREPROCESSED"
    return candidate if candidate.is_dir() else data_dir


def resolve_labels_path(data_dir: Path, filename: str = "labels_train.csv") -> Path:
    candidates = (
        data_dir / filename,
        data_dir.parent / filename,
        data_dir.parent / "data" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} near {data_dir}")


def _as_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_trial_index(
    data_dir: Path,
    labels_path: Path | str | None = None,
    config: PreprocessingConfig | None = None,
    participant_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    config = config or PreprocessingConfig()
    data_dir = Path(data_dir)
    labels_path = (
        resolve_labels_path(data_dir)
        if labels_path is None
        else Path(labels_path)
    )

    labels = pd.read_csv(labels_path)
    required = {"participant_id", "day", "video", "start_marker", "final_emotion_label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"labels_train.csv missing columns: {sorted(missing)}")

    outlier_column = "flagged_outliers" if "flagged_outliers" in labels else "flagged_outlier"
    if outlier_column not in labels:
        raise ValueError("labels_train.csv has no flagged outlier column")

    numeric_flag = pd.to_numeric(labels[outlier_column], errors="coerce")
    text_flag = labels[outlier_column].astype(str).str.strip().str.upper()
    labels = labels.loc[
        (numeric_flag != 1)
        & (text_flag != "TESTSUBJECT")
        & (labels["final_emotion_label"].astype(str).str.upper() != "TESTSUBJECT")
    ].copy()
    labels = labels.loc[
        labels["participant_id"].notna()
        & pd.to_numeric(labels["day"], errors="coerce").notna()
        & labels["video"].notna()
        & labels["final_emotion_label"].notna()
        & (labels["final_emotion_label"].astype(str).str.strip() != "")
    ].copy()
    if participant_ids is not None:
        allowed = {str(value) for value in participant_ids}
        labels = labels.loc[labels["participant_id"].astype(str).isin(allowed)].copy()

    root = _preprocessed_root(data_dir)
    rows: list[dict] = []
    recording_bounds: dict[Path, tuple[float, list[tuple[str, float]]]] = {}

    def get_recording_bounds(
        bdf_path: Path,
    ) -> tuple[float, list[tuple[str, float]]] | None:
        if bdf_path in recording_bounds:
            return recording_bounds[bdf_path]
        try:
            raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
            try:
                duration_s = raw.n_times / float(raw.info["sfreq"])
                annotations = [
                    (str(description), float(onset - raw.first_time))
                    for onset, description in zip(
                        raw.annotations.onset,
                        raw.annotations.description,
                        strict=True,
                    )
                ]
                recording_bounds[bdf_path] = (duration_s, annotations)
                return recording_bounds[bdf_path]
            finally:
                raw.close()
        except Exception:
            return None

    for row in labels.itertuples(index=False):
        participant = str(row.participant_id)
        day_value = _as_int(row.day)
        marker = _as_int(row.start_marker)
        if day_value is None or marker is None:
            continue

        metadata_path = root / participant / f"{participant}_day-{day_value}_metadata.json"
        if not metadata_path.is_file():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        segment = metadata.get("dataset_export", {}).get("segments", {}).get("emotion-video", {})
        origin = segment.get("metadata_time_origin_s")
        if origin is None:
            origin = metadata.get("metadata_time_origin_s")
        origin = 0.0 if origin is None else float(origin)
        events = metadata.get("events", [])
        matches = [
            event for event in events
            if _as_int(event.get("marker_id")) == marker
            and event.get("trial_type") == "emotion_video"
        ]
        if not matches:
            continue

        video = str(row.video)
        event = next(
            (
                item for item in matches
                if str(item.get("media_file") or "") == video
            ),
            matches[0],
        )
        media_file = event.get("media_file") or event.get("instruction_text")
        if media_file and str(media_file) != video:
            continue

        metadata_start_s = float(event["onset_s"]) - origin
        metadata_duration_s = float(
            event.get("duration_s") or config.trial_duration_s
        )
        segment_duration = segment.get("duration_s")
        if (
            segment_duration is None
            and "requested_start_s" in segment
            and "requested_end_s" in segment
        ):
            segment_duration = (
                float(segment["requested_end_s"])
                - float(segment["requested_start_s"])
            )
        if (
            metadata_start_s < 0.0
            or metadata_duration_s <= 0.0
            or (
                segment_duration is not None
                and metadata_start_s + metadata_duration_s
                > float(segment_duration) + 1e-6
            )
        ):
            continue

        filename = segment.get("file") or (
            f"{participant}_day-{day_value}_task-emotion-video_multimodal.bdf"
        )
        bdf_path = root / participant / str(filename)
        record_map = bdf_path.with_suffix(".record_map")
        if not bdf_path.is_file() or not record_map.is_file():
            continue

        bounds = get_recording_bounds(bdf_path)
        if bounds is None:
            recording_duration_s, annotations = float("inf"), []
        else:
            recording_duration_s, annotations = bounds

        start_matches = [
            onset for description, onset in annotations
            if description == str(marker)
        ]
        if start_matches:
            start_s = float(start_matches[0])
        else:
            metadata_start_s = float(event["onset_s"]) - origin
            anchor = None
            for candidate in events:
                if candidate.get("trial_type") != "emotion_video":
                    continue
                candidate_marker = _as_int(candidate.get("marker_id"))
                if candidate_marker is None:
                    continue
                candidate_matches = [
                    onset for description, onset in annotations
                    if description == str(candidate_marker)
                ]
                if candidate_matches:
                    anchor = (
                        float(candidate["onset_s"]) - origin,
                        float(candidate_matches[0]),
                    )
                    break
            if anchor is not None:
                start_s = anchor[1] + (metadata_start_s - anchor[0])
            else:
                start_s = metadata_start_s

        end_matches = [
            onset for description, onset in annotations
            if description == str(marker + 1) and onset > start_s
        ]
        if end_matches:
            end_s = float(end_matches[0])
        else:
            later_thirteen = [
                onset for description, onset in annotations
                if onset > start_s and description.endswith("13")
            ]
            if later_thirteen:
                end_s = float(min(later_thirteen))
            else:
                end_s = start_s + float(event.get("duration_s") or config.trial_duration_s)

        duration_s = end_s - start_s
        if (
            start_s < 0.0
            or duration_s <= 0.0
            or end_s > recording_duration_s + 1e-6
        ):
            continue

        try:
            eeg_names, _, _, _ = channel_names(record_map)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

        rows.append({
            "participant_id": participant,
            "day": day_value,
            "video": video,
            "start_marker": marker,
            "final_emotion_label": str(row.final_emotion_label),
            "start_s": start_s,
            "duration_s": duration_s,
            "bdf_path": str(bdf_path),
            "n_eeg_channels": len(eeg_names),
        })

    return pd.DataFrame(rows)
