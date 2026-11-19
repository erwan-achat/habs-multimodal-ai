from pathlib import Path
import json

import mne
import numpy as np
from scipy.signal import resample_poly


def window_offsets(
    duration_s: float,
    window_s: float = 3.0,
    stride_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Return non-overlapping-start windows inside a segment."""
    starts = np.arange(0.0, duration_s - window_s + 1e-9, stride_s)
    return [(float(start), float(start + window_s)) for start in starts]


def _channel_names(record_map: Path) -> tuple[list[str], str, str]:
    data = json.loads(record_map.read_text(encoding="utf-8"))
    eeg_names: list[str] = []
    gsr_name = ""
    respiration_name = ""

    def add_range(first: int, last: int) -> list[int]:
        return list(range(first, last + 1))

    for stream in data["stream_map"].values():
        for _, spec in stream.items():
            correction = spec.get("correction", {})
            kind = str(correction.get("RecordType", "")).lower()
            if "channel_ranges" in spec:
                indices = add_range(*spec["channel_ranges"])
            else:
                indices = list(spec.get("channel_indices", []))
            details = correction.get("Channels", [])
            names = [str(detail["name"]) for detail in details]
            if kind == "eeg":
                eeg_names.extend(names[: len(indices)])
            elif kind == "gsr" and names:
                gsr_name = names[0]
            elif kind in {"resp", "respiration"} and names:
                respiration_name = names[0]

    if not eeg_names:
        raise ValueError("No EEG channels found in the record map")
    if not gsr_name or not respiration_name:
        raise ValueError("GSR or respiration channel metadata is missing")
    return eeg_names, gsr_name, respiration_name


def resample_signal(values: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(source_sfreq, target_sfreq):
        return np.asarray(values, dtype=np.float32)
    from math import gcd

    divisor = gcd(round(source_sfreq), round(target_sfreq))
    up = round(target_sfreq) // divisor
    down = round(source_sfreq) // divisor
    return resample_poly(values, up=up, down=down, axis=-1).astype(np.float32)


def load_preprocessed_segment(
    bdf_path: Path,
    start_s: float,
    duration_s: float = 10.0,
) -> dict[str, np.ndarray]:
    """Load only the requested segment and return EEG/GSR/respiration arrays."""
    record_map = bdf_path.with_suffix(".record_map")
    eeg_names, gsr_name, respiration_name = _channel_names(record_map)

    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
    source_sfreq = float(raw.info["sfreq"])
    start = max(0, round(start_s * source_sfreq))
    stop = min(raw.n_times, round((start_s + duration_s) * source_sfreq))

    eeg = raw.get_data(picks=eeg_names, start=start, stop=stop)
    gsr = raw.get_data(picks=[gsr_name], start=start, stop=stop)[0]
    respiration = raw.get_data(picks=[respiration_name], start=start, stop=stop)[0]

    eeg = resample_signal(eeg, source_sfreq, 256.0)
    gsr = resample_signal(gsr, source_sfreq, 32.0)
    respiration = resample_signal(respiration, source_sfreq, 32.0)
    return {"eeg": eeg, "gsr": gsr[np.newaxis, :], "respiration": respiration[np.newaxis, :]}
