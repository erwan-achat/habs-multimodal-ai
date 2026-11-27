"""Build the trial manifest and verify the join before extracting features."""

import json
import logging
import re
from collections import defaultdict
from contextlib import ExitStack
from fnmatch import fnmatchcase
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import pyedflib

from . import VIDEO_PATTERN
from .signals import FEATURE_COLUMNS, FEATURES, TIMING_TOLERANCE_S, WINDOW_S, Signals

TARGETS = ("rating_arousal", "rating_valence")
LOG = logging.getLogger(__name__)
KEY = ["participant_id", "day", "start_marker", "occurrence"]
MANIFEST_COLUMNS = [
    *KEY, "cohort", "video", "metadata_video", "end_marker", "labels_end_marker", "label_day_source",
    "bdf", "metadata", "record_map", "bdf_start_s", "bdf_end_s", "metadata_start_s",
    "metadata_end_s", "metadata_duration_source", "bdf_minus_metadata_s",
    "bdf_end_minus_metadata_s", "window_start_s", "window_end_s",
    "n_bdf_events", "n_metadata_events", "n_label_rows", "trial_status", "flagged_outliers",
    *TARGETS, *(f"{name}_status" for name in TARGETS),
    *(f"{sensor}_{field}" for sensor in FEATURES for field in ("status", "trace_status", "channels", "issues")),
]
FILENAME = re.compile(r"(?P<id>P[^_]+)_day-(?P<day>\d+)_")


def normalize_clip(value):
    return value.replace("\\", "/").split("/")[-1].strip().casefold()


def integer(text):
    match = re.fullmatch(r"([0-9]+)(?:\.0+)?", str(text).strip())
    return int(match[1]) if match else None


def day_number(text):
    return integer(str(text).strip().lower().removeprefix("day-"))


def read_markers(path):
    codes = pd.read_csv(path, dtype=str, keep_default_na=False)
    names = ["Start Marker", "End Marker"]
    pairs = {}
    for start_text, end_text in codes[names].itertuples(index=False, name=None):
        start, end = integer(start_text), integer(end_text)
        if start is not None and start >= 10100:
            if end is None or start in pairs:
                raise ValueError(f"Marqueur vidéo invalide ou dupliqué dans {path.name}: {start}")
            pairs[start] = end
    if not pairs or set(pairs) & set(pairs.values()) or len(set(pairs.values())) != len(pairs):
        raise ValueError(f"Marqueurs vidéo non appariables dans {path.name}")
    return pairs


def read_labels(path):
    labels = defaultdict(lambda: defaultdict(list))
    invalid = []
    if not path.is_file():
        return labels, invalid
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"participant_id", "day", "start_marker", "end_marker", "video", *TARGETS}
    if missing := required - set(table.columns):
        raise ValueError(f"Colonnes manquantes dans {path.name}: {', '.join(sorted(missing))}")
    for source_row, row in enumerate(table.to_dict("records"), 2):
        participant = row["participant_id"].strip().upper()
        day = day_number(row["day"])
        marker = integer(row["start_marker"])
        if participant and day is not None and marker is not None:
            labels[participant, day][marker].append(row)
        else:
            invalid.append({**label_fields(row), "participant_id": participant or None,
                            "day": day, "start_marker": marker, "occurrence": -source_row,
                            "n_label_rows": 1, "trial_status": "invalid_label_key"})
    return labels, invalid


def label_fields(row):
    fields = {
        "cohort": None if row.get("cohort") in (None, "", "TESTSUBJECT") else row["cohort"],
        "video": normalize_clip(row.get("video", "")) or None,
        "labels_end_marker": integer(row.get("end_marker")),
        "label_day_source": "csv" if row.get("day", "").strip() else None,
        "flagged_outliers": integer(row.get("flagged_outliers")),
    }
    for target in TARGETS:
        value = row.get(target, "").strip()
        number = pd.to_numeric(value, errors="coerce")
        if not row:
            status = "missing_label_row"
        elif value.upper() == "TESTSUBJECT":
            status = "held_out_release"
        elif not value:
            status = "missing"
        elif not np.isfinite(number):
            status = "invalid"
        elif not 1 <= number <= 9:
            status = "out_of_range"
        else:
            status = "ok"
        fields[target] = float(number) if status == "ok" else np.nan
        fields[f"{target}_status"] = status
    return fields


def bdf_trials(reader, pairs):
    starts, _, descriptions = reader.readAnnotations()
    boundaries = set(pairs) | set(pairs.values())
    events = [(float(time), code) for time, text in zip(starts, descriptions, strict=True)
              if (code := integer(text)) in boundaries]
    result = defaultdict(list)
    for i, (start, marker) in enumerate(events):
        if marker in pairs:
            end, end_marker = events[i + 1] if i + 1 < len(events) else (np.nan, None)
            valid_end = end_marker == pairs[marker] and end > start
            result[marker].append((start, end if valid_end else np.nan))
    return result


def metadata_trials(path):
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    segment = data["dataset_export"]["segments"].get("emotion-video")
    if segment is None:
        return {}
    origin = float(segment["metadata_time_origin_s"])
    result = defaultdict(list)
    for event in data["events"]:
        if str(event.get("trial_type", "")).replace("-", "_") != "emotion_video":
            continue
        marker = integer(event.get("marker_id"))
        onset, duration = event.get("onset_s"), event.get("duration_s")
        start = float(onset) - origin if onset is not None else np.nan
        end = start + float(duration) if duration is not None else np.nan
        clip = normalize_clip(event.get("media_file") or "")
        result[marker].append((start, end, clip, event.get("duration_source")))
    return result


def find_session_files(root):
    sessions = defaultdict(dict)
    for participant in sorted((root / "PREPROCESSED").iterdir()):
        if not participant.name.startswith("P") or not participant.is_dir():
            continue
        for path in sorted(participant.iterdir()):
            match = FILENAME.match(path.name)
            if not match:
                continue
            if fnmatchcase(path.name, VIDEO_PATTERN):
                kind = "bdf"
            elif path.name.endswith("_metadata.json"):
                kind = "metadata"
            else:
                continue
            key = match["id"].upper(), int(match["day"])
            if kind in sessions[key]:
                raise ValueError(f"Deux fichiers {kind} pour la session {key}")
            sessions[key][kind] = path
    return sessions


def timing_status(row, signals):
    if not np.isfinite(row["bdf_end_s"]):
        return "unpaired_bdf_markers"
    if not (np.isfinite(row["metadata_start_s"]) and np.isfinite(row["metadata_end_s"])):
        return "missing_metadata_times"
    # A protocol duration describes the clip. The end annotation can mark
    # the later task transition or end earlier when a clip was interrupted.
    measured_end = row["metadata_duration_source"] != "protocol_csv"
    if (abs(row["bdf_minus_metadata_s"]) > TIMING_TOLERANCE_S
            or (measured_end and abs(row["bdf_end_minus_metadata_s"]) > TIMING_TOLERANCE_S)):
        return "discordant_timing"
    sample = 1 / signals.master_rate
    start = row["bdf_start_s"]
    metadata_duration = row["metadata_end_s"] - row["metadata_start_s"]
    stop = min(start + min(WINDOW_S, metadata_duration), row["bdf_end_s"], signals.duration)
    if start < -sample / 2 or stop - start < WINDOW_S - sample - 1e-9:
        return "truncated_window"
    row["window_start_s"], row["window_end_s"] = max(start, 0.0), stop
    return "ok"


def trial_status(row, signals, marker_known, problem):
    if max(row["n_bdf_events"], row["n_metadata_events"], row["n_label_rows"]) > 1:
        return "ambiguous_occurrence"
    if not marker_known:
        return "unknown_start_marker"
    if problem:
        return problem
    if not row["n_metadata_events"]:
        return "missing_metadata_event"
    if not row["n_bdf_events"]:
        return "missing_bdf_event"
    if row["n_label_rows"]:
        if row["labels_end_marker"] != row["end_marker"]:
            return "label_end_marker_mismatch"
        if not row["video"] or row["video"] != row["metadata_video"]:
            return "clip_mismatch"
    return timing_status(row, signals)


def session_trials(root, key, files, labels, pairs):
    participant, day = key
    bdf = files.get("bdf")
    map_path = bdf.with_suffix(".record_map") if bdf else None
    metadata_path = files.get("metadata")
    metadata = metadata_trials(metadata_path) if metadata_path else {}
    with ExitStack() as stack:
        signals, events = None, {}
        if bdf is None:
            problem = "missing_video_bdf"
        else:
            reader = stack.enter_context(pyedflib.EdfReader(str(bdf)))
            events = bdf_trials(reader, pairs)
            problem = None
            if map_path.is_file():
                signals = Signals(reader, map_path)
            else:
                problem = "missing_record_map"
        for marker in sorted(set(metadata) | set(events) | set(labels), key=lambda value: (value is None, value)):
            notes = labels.get(marker, [])
            bdf_events = events.get(marker, [])
            meta_events = metadata.get(marker, [])
            for i in range(max(len(notes), len(bdf_events), len(meta_events))):
                note = notes[i] if i < len(notes) else {}
                start, end = bdf_events[i] if i < len(bdf_events) else (np.nan, np.nan)
                meta_start, meta_end, meta_clip, duration_source = (
                    meta_events[i] if i < len(meta_events) else (np.nan, np.nan, None, None)
                )
                row = {
                    **label_fields(note), "participant_id": participant, "day": day,
                    "start_marker": marker, "occurrence": i + 1, "end_marker": pairs.get(marker),
                    "metadata_video": meta_clip, "bdf_start_s": start, "bdf_end_s": end,
                    "metadata_start_s": meta_start, "metadata_end_s": meta_end,
                    "metadata_duration_source": duration_source,
                    "bdf_minus_metadata_s": start - meta_start, "bdf_end_minus_metadata_s": end - meta_end,
                    "n_bdf_events": len(bdf_events),
                    "n_metadata_events": len(meta_events), "n_label_rows": len(notes),
                    "window_start_s": np.nan, "window_end_s": np.nan,
                    "bdf": str(bdf.relative_to(root)) if bdf else None,
                    "metadata": str(metadata_path.relative_to(root)) if metadata_path else None,
                    "record_map": str(map_path.relative_to(root)) if map_path else None,
                }
                if not note:
                    row["video"] = meta_clip
                row["trial_status"] = trial_status(row, signals, marker in pairs, problem)
                features = {}
                if row["trial_status"] == "ok":
                    features, details = signals.extract(row["window_start_s"], row["window_end_s"])
                    row.update(details)
                yield row, features


def build_manifest(root, sessions, pairs, labels, invalid):
    manifest_rows = list(invalid)
    feature_rows = []
    video_count = sum("bdf" in files for files in sessions.values())
    processed = 0
    last_log = monotonic()
    for key in sorted(set(sessions) | set(labels)):
        files = sessions.get(key, {})
        for row, features in session_trials(root, key, files, labels.get(key, {}), pairs):
            manifest_rows.append(row)
            if row["trial_status"] == "ok":
                feature_rows.append({**{name: row[name] for name in KEY}, **features})
        if "bdf" in files:
            processed += 1
            now = monotonic()
            if processed in (1, video_count) or processed % 25 == 0 or now - last_log >= 15:
                LOG.info("Extraction %d/%d BDF vidéo. %d fenêtres vérifiées.",
                         processed, video_count, len(feature_rows))
                last_log = now
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    features = pd.DataFrame(feature_rows, columns=[*KEY, *FEATURE_COLUMNS])
    return recover_missing_days(manifest), features


def recover_missing_days(manifest):
    """Attach a dayless label only when exactly one recorded trial matches."""
    key = ["participant_id", "start_marker", "video", "labels_end_marker"]
    pending = manifest.loc[
        manifest.trial_status.eq("invalid_label_key") & manifest.day.isna() & manifest.label_day_source.isna(),
        key,
    ].dropna().drop_duplicates(key, keep=False)
    if pending.empty:
        return manifest
    # Include failed candidates in the ambiguity check, then require the
    # remaining candidate to have a verified BDF interval and an empty label slot.
    recording_columns = ["participant_id", "start_marker", "metadata_video", "end_marker"]
    recorded = manifest.loc[manifest.n_metadata_events.gt(0), recording_columns].dropna()
    recorded.columns = key
    unique = recorded.drop_duplicates(key, keep=False)
    ready = unique.loc[
        manifest.loc[unique.index, "trial_status"].eq("ok")
        & manifest.loc[unique.index, "n_label_rows"].eq(0)
    ]
    matches = pending.rename_axis("label_index").reset_index().merge(
        ready.rename_axis("recording_index").reset_index(), on=key, validate="one_to_one"
    )
    columns = ["cohort", "video", "labels_end_marker", "flagged_outliers",
               *TARGETS, *(f"{target}_status" for target in TARGETS)]
    for label_index, recording_index in matches[["label_index", "recording_index"]].itertuples(index=False, name=None):
        manifest.loc[recording_index, columns] = manifest.loc[label_index, columns]
        manifest.at[recording_index, "label_day_source"] = "verified_recording"
        manifest.at[recording_index, "n_label_rows"] = 1
    return manifest.drop(index=matches.label_index).reset_index(drop=True)
