import atexit
import json
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from math import gcd
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

from disentanglement.config import PreprocessingConfig


class RawCache:
    def __init__(self, maxsize: int = 4):
        self.maxsize = max(1, maxsize)
        self._items: OrderedDict[Path, mne.io.BaseRaw] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, bdf_path: Path) -> mne.io.BaseRaw:
        bdf_path = Path(bdf_path)
        with self._lock:
            cached = self._items.get(bdf_path)
            if cached is not None:
                self._items.move_to_end(bdf_path)
                return cached

        raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
        evicted = None
        with self._lock:
            if len(self._items) >= self.maxsize:
                _, evicted = self._items.popitem(last=False)
            self._items[bdf_path] = raw
        if evicted is not None:
            evicted.close()
        return raw

    def close(self) -> None:
        with self._lock:
            items = list(self._items.values())
            self._items.clear()
        for raw in items:
            raw.close()


RAW_CACHE = RawCache(int(os.environ.get("BDF_CACHE_SIZE", "4")))
atexit.register(RAW_CACHE.close)


def get_raw(bdf_path: Path) -> mne.io.BaseRaw:
    return RAW_CACHE.get(Path(bdf_path))


def window_offsets(
    duration_s: float,
    window_s: float = 3.0,
    stride_s: float = 2.0,
) -> list[tuple[float, float]]:
    starts = np.arange(0.0, duration_s - window_s + 1e-9, stride_s)
    return [(float(start), float(start + window_s)) for start in starts]


@lru_cache(maxsize=256)
def channel_names(record_map: Path) -> tuple[tuple[str, ...], str, str, str]:
    data = json.loads(record_map.read_text(encoding="utf-8"))
    eeg_names: list[str] = []
    ecg_name = ""
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
            elif kind == "ecg" and names:
                ecg_name = names[0]
            elif kind == "gsr" and names:
                gsr_name = names[0]
            elif kind in {"resp", "respiration"} and names:
                respiration_name = names[0]

    if not eeg_names:
        raise ValueError("No EEG channels found in the record map")
    if not all((ecg_name, gsr_name, respiration_name)):
        raise ValueError("ECG, GSR or respiration channel metadata is missing")
    return tuple(eeg_names), ecg_name, gsr_name, respiration_name


def resample_signal(values: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(source_sfreq, target_sfreq):
        return np.asarray(values, dtype=np.float32)

    divisor = gcd(round(source_sfreq), round(target_sfreq))
    up = round(target_sfreq) // divisor
    down = round(source_sfreq) // divisor
    return resample_poly(values, up=up, down=down, axis=-1).astype(np.float32)


def load_preprocessed_segment(
    bdf_path: Path,
    start_s: float,
    duration_s: float | None = None,
    config: PreprocessingConfig | None = None,
) -> dict[str, np.ndarray]:
    config = config or PreprocessingConfig()
    duration_s = config.trial_duration_s if duration_s is None else duration_s
    record_map = bdf_path.with_suffix(".record_map")
    eeg_names, ecg_name, gsr_name, respiration_name = channel_names(record_map)

    raw = get_raw(bdf_path)
    source_sfreq = float(raw.info["sfreq"])
    start = max(0, round(start_s * source_sfreq))
    stop = min(raw.n_times, round((start_s + duration_s) * source_sfreq))
    if start_s < 0 or stop <= start:
        raise ValueError(f"Invalid segment bounds: start={start_s}, duration={duration_s}")
    if stop != round((start_s + duration_s) * source_sfreq):
        raise ValueError(
            f"Segment extends beyond {bdf_path.name}: "
            f"{start_s + duration_s:.3f}s > {raw.n_times / source_sfreq:.3f}s"
        )

    eeg = raw.get_data(picks=list(eeg_names), start=start, stop=stop)
    ecg = raw.get_data(picks=[ecg_name], start=start, stop=stop)[0]
    gsr = raw.get_data(picks=[gsr_name], start=start, stop=stop)[0]
    respiration = raw.get_data(picks=[respiration_name], start=start, stop=stop)[0]

    eeg = resample_signal(eeg, source_sfreq, config.eeg_sfreq)
    ecg = resample_signal(ecg, source_sfreq, config.ecg_sfreq)
    gsr = resample_signal(gsr, source_sfreq, config.gsr_sfreq)
    respiration = resample_signal(respiration, source_sfreq, config.respiration_sfreq)
    return {
        "eeg": eeg,
        "ecg": ecg[np.newaxis, :],
        "gsr": gsr[np.newaxis, :],
        "respiration": respiration[np.newaxis, :],
    }
