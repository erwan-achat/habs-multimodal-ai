"""Read only the channels and sample intervals used by V0."""

import json
from pathlib import Path

import numpy as np
import pyedflib
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from starting_notebooks.bdf_loader import uV_to_uS

WINDOW_S = 10.0
TIMING_TOLERANCE_S = 0.020
REGIONS = {
    "frontal": ("F3", "Fz", "F4"),
    "central": ("C3", "Cz", "C4"),
    "parietal": ("P3", "Pz", "P4"),
    "occipital": ("O1", "Oz", "O2"),
}
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
QC = {
    "flat_span_min": 1e-9,
    "rail_fraction_max": 0.01,
    "eeg_min_channels_per_region": 2,
    "ecg_filter_hz": (5, 20),
    "ecg_envelope_s": 0.08,
    "ecg_rr_s": (0.4, 1.5),
    "ecg_min_rr": 6,
    "resp_filter_hz": (0.08, 0.7),
    "resp_peak_distance_s": 1.25,
    "resp_prominence_std": 0.5,
    "resp_cycle_s": (1.5, 8.0),
    "resp_min_cycles": 2,
    "welch_s": 2.0,
    "welch_overlap": 0.5,
}
FEATURES = {
    "eeg": [f"eeg_{region}_{band}" for region in REGIONS for band in BANDS],
    "ecg": ["ecg_bpm"],
    "gsr": ["gsr_median_us", "gsr_slope_us_s"],
    "resp": ["resp_cpm"],
}
FEATURE_COLUMNS = [name for columns in FEATURES.values() for name in columns]
TO_UV = {"v": 1e6, "mv": 1e3, "uv": 1.0, "nv": 1e-3}


def unit_key(unit):
    return unit.strip().lower().replace("μ", "u").replace("µ", "u")


class Signals:
    def __init__(self, reader: pyedflib.EdfReader, map_path: Path):
        self.reader = reader
        record_map = json.loads(map_path.read_text(encoding="utf-8"))
        self.channels = {modality: {} for modality in FEATURES}
        self.total_samples = record_map.get("TotalSamples")
        self.rates = reader.getSampleFrequencies()
        self.counts = reader.getNSamples()
        self.master_rate = max(self.rates)
        self.duration = float(reader.file_duration)
        if isinstance(self.total_samples, int) and self.total_samples > 0:
            self.duration = min(self.duration, self.total_samples / self.master_rate)

        labels = reader.getSignalLabels()
        electrodes = {name.casefold() for region in REGIONS.values() for name in region}
        for stream in record_map["stream_map"].values():
            for group, spec in stream.items():
                correction = spec.get("correction", {})
                modality = correction.get("RecordType", group).casefold()
                modality = {"respiration": "resp", "eda": "gsr"}.get(modality, modality)
                if modality not in FEATURES:
                    continue
                indices = spec.get("channel_indices", ())
                if "channel_ranges" in spec:
                    first, last = spec["channel_ranges"]
                    indices = range(first, last + 1)
                for index, detail in zip(indices, correction.get("Channels", []), strict=True):
                    name = detail["name"]
                    if not 0 <= index < len(labels) or name != labels[index]:
                        raise ValueError(f"record_map et BDF divergent pour le canal {name}")
                    if modality != "eeg" or name.casefold() in electrodes:
                        self.channels[modality][name.casefold()] = index

    def read(self, index, start, end):
        rate = self.rates[index]
        first, stop = round(start * rate), round(end * rate)
        if first < 0 or stop > self.counts[index] or stop - first < round(WINDOW_S * rate) - 1:
            return None, "samples_out_of_bounds"
        values = self.reader.readSignal(index, first, stop - first)
        if not np.isfinite(values).all():
            return None, "non_finite"
        if np.ptp(values) <= QC["flat_span_min"]:
            return None, "flat"
        low = self.reader.getPhysicalMinimum(index)
        high = self.reader.getPhysicalMaximum(index)
        digital_span = self.reader.getDigitalMaximum(index) - self.reader.getDigitalMinimum(index)
        half_step = (high - low) / (2 * digital_span)
        if np.mean((values <= low + half_step) | (values >= high - half_step)) > QC["rail_fraction_max"]:
            return None, "saturated"
        return values, "ok"

    def native_window(self, sensor, names, start, end):
        """Read in caller-specified order. Only trace quality gates apply here."""
        samples = []
        rate = None
        for name in names:
            index = self.channels[sensor].get(name.casefold())
            if index is None:
                raise ValueError(f"{sensor}/{name}: sensor_absent")
            values, status = self.read(index, start, end)
            if status != "ok":
                raise ValueError(f"{sensor}/{name}: {status}")
            current_rate = self.rates[index]
            if rate is not None and rate != current_rate:
                raise ValueError(f"{sensor}: frequencies differ within branch")
            rate = current_rate
            unit = unit_key(self.reader.getPhysicalDimension(index))
            if sensor == "gsr":
                values = conductance_us(values, unit)
            elif sensor in ("eeg", "ecg") and unit in TO_UV:
                values = values * (TO_UV[unit] / (1000 if sensor == "ecg" else 1))
            elif sensor != "resp" or unit not in ("au", "a.u.", "a.u"):
                raise ValueError(f"{sensor}/{name}: unsupported_unit {unit!r}")
            samples.append(values)
        return np.stack(samples), rate

    def extract(self, start, end):
        features = {}
        details = {}
        for modality, channels in self.channels.items():
            used = []
            rejected = []
            trace_checks = []
            if modality == "eeg":
                powers = {}
                for name, index in channels.items():
                    values, status = self.read(index, start, end)
                    trace_checks.append(status)
                    if status == "ok":
                        powers[name], status = eeg_power(
                            values, self.rates[index], self.reader.getPhysicalDimension(index)
                        )
                    if status != "ok":
                        rejected.append(f"{name}:{status}")
                for region, names in REGIONS.items():
                    valid = [name for name in names if powers.get(name.casefold()) is not None]
                    if len(valid) >= QC["eeg_min_channels_per_region"]:
                        used.extend(valid)
                        for band in BANDS:
                            features[f"eeg_{region}_{band}"] = np.mean(
                                [powers[name.casefold()][band] for name in valid]
                            )
            elif channels:
                if len(channels) != 1:
                    rejected.append("ambiguous_channel")
                    trace_checks.append("ambiguous_channel")
                else:
                    (name, index), = channels.items()
                    values, status = self.read(index, start, end)
                    trace_checks.append(status)
                    if status == "ok":
                        extracted, status = PERIPHERAL[modality](
                            values, self.rates[index], self.reader.getPhysicalDimension(index)
                        )
                        features.update(extracted)
                    if status == "ok":
                        used.append(self.reader.getLabel(index))
                    else:
                        rejected.append(f"{name}:{status}")
            if used:
                status = "ok"
            elif not channels:
                status = "sensor_absent"
            elif modality == "eeg" and any(power is not None for power in powers.values()):
                status = "insufficient_region_channels"
            else:
                status = "unusable"
            details[f"{modality}_status"] = status
            details[f"{modality}_trace_status"] = (
                "ok" if "ok" in trace_checks else "|".join(sorted(set(trace_checks))) or "sensor_absent"
            )
            details[f"{modality}_channels"] = ",".join(used)
            details[f"{modality}_issues"] = "|".join(rejected)
        return features, details


def eeg_power(values, rate, unit):
    factor = TO_UV.get(unit_key(unit))
    if factor is None:
        return None, "unsupported_unit"
    if rate / 2 < max(high for _, high in BANDS.values()):
        return None, "rate_too_low"
    size = round(QC["welch_s"] * rate)
    frequencies, density = welch(
        values * factor, fs=rate, window="hann", nperseg=size,
        noverlap=round(QC["welch_overlap"] * size),
    )
    powers = {}
    for band, (low, high) in BANDS.items():
        within = (frequencies > low) & (frequencies < high)
        grid = np.r_[low, frequencies[within], high]
        power = trapezoid(np.interp(grid, frequencies, density), grid)
        if power <= 0 or not np.isfinite(power):
            return None, "undefined_band_power"
        powers[band] = np.log(power)
    return powers, "ok"


def ecg_features(values, rate, _unit):
    if rate / 2 <= QC["ecg_filter_hz"][1]:
        return {}, "rate_too_low"
    # The notebook uses a rectified QRS envelope to handle inverted ECG traces.
    sos = butter(2, QC["ecg_filter_hz"], btype="bandpass", fs=rate, output="sos")
    width = max(1, round(QC["ecg_envelope_s"] * rate))
    filtered = sosfiltfilt(sos, values)
    envelope = np.convolve(np.abs(filtered), np.ones(width) / width, mode="same")
    low, high = QC["ecg_rr_s"]
    peaks, _ = find_peaks(envelope, distance=max(1, round(low * rate)), prominence=envelope.std())
    rr = np.diff(peaks) / rate
    rr = rr[(rr >= low) & (rr <= high)]
    if len(rr) < QC["ecg_min_rr"]:
        return {}, "insufficient_rr_intervals"
    return {"ecg_bpm": np.median(60 / rr)}, "ok"


def conductance_us(values, unit):
    key = unit_key(unit)
    if key in TO_UV:
        return uV_to_uS(values * TO_UV[key])
    if key == "us":
        return values
    raise ValueError(f"GSR: unsupported_unit {unit!r}")


def gsr_features(values, rate, unit):
    key = unit_key(unit)
    if key not in TO_UV and key != "us":
        return {}, "unsupported_unit"
    conductance = conductance_us(values, key)
    if key in TO_UV and np.mean((conductance <= 1) | (conductance >= 40)) > QC["rail_fraction_max"]:
        return {}, "outside_calibration_range"
    times = np.arange(len(values)) / rate
    times -= times.mean()
    slope = (times @ (conductance - conductance.mean())) / (times @ times)
    return {"gsr_median_us": np.median(conductance), "gsr_slope_us_s": slope}, "ok"


def respiratory_features(values, rate, _unit):
    if rate / 2 <= QC["resp_filter_hz"][1] or len(values) <= 15:
        return {}, "rate_too_low"
    sos = butter(2, QC["resp_filter_hz"], btype="bandpass", fs=rate, output="sos")
    trace = sosfiltfilt(sos, values)
    peaks, _ = find_peaks(
        trace, distance=max(1, round(QC["resp_peak_distance_s"] * rate)),
        prominence=QC["resp_prominence_std"] * trace.std(),
    )
    low, high = QC["resp_cycle_s"]
    cycles = np.diff(peaks) / rate
    cycles = cycles[(cycles >= low) & (cycles <= high)]
    if len(cycles) < QC["resp_min_cycles"]:
        return {}, "insufficient_cycles"
    return {"resp_cpm": 60 / np.median(cycles)}, "ok"


PERIPHERAL = {"ecg": ecg_features, "gsr": gsr_features, "resp": respiratory_features}
