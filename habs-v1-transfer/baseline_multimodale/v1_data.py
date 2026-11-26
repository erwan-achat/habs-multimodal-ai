"""Frozen V0 population, partitions and session-level signal cache for V1."""

import hashlib
import json
import logging
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import resample_poly

from . import evaluation
from .signals import FEATURE_COLUMNS, Signals, WINDOW_S, gsr_features
from .trials import KEY, TARGETS

EEG = ("F3", "Fz", "F4", "C3", "C4", "P3", "Pz", "P4", "O1", "O2")
RATES = {"eeg": 250, "ecg": 250, "gsr": 25, "resp": 25}
REFERENCES = ("median", "four_modalities", "clip_ridge")
LOG = logging.getLogger(__name__)


@contextmanager
def atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def digest_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_table(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def frozen_study(v0_dir):
    manifest = read_table(v0_dir / "manifest.csv")
    features = read_table(v0_dir / "features.csv")
    folds = read_table(v0_dir / "folds.csv").query("population == 'four_modalities'")
    predictions = read_table(v0_dir / "predictions.csv").query("population == 'four_modalities'")
    eligible = manifest[[f"eligible_{t}" for t in TARGETS]]
    if not all(pd.api.types.is_bool_dtype(eligible[col]) for col in eligible):
        raise ValueError("Invalid V0 eligibility masks")
    rows = manifest[eligible.any(axis=1)].merge(features, on=KEY, how="left",
                                               validate="one_to_one", indicator=True)
    if rows.empty or not rows._merge.eq("both").all() or rows[KEY].isna().any().any():
        raise ValueError("Missing V0 four-modality trials or features")
    rows = rows.drop(columns="_merge").reset_index(drop=True)
    if folds.duplicated().any():
        raise ValueError("Duplicate V0 fold assignment")
    assignment = folds[folds.role.eq("assignment")].set_index("participant_id").outer_fold
    if not assignment.index.is_unique:
        raise ValueError("Ambiguous external assignment")

    partitions = {}
    for target in TARGETS:
        selected = rows[rows[f"eligible_{target}"]]
        target_folds = folds[folds.target.eq(target)]
        refs = predictions[predictions.target.eq(target)]
        if selected.empty and refs.empty:
            continue
        if not set(REFERENCES).issubset(refs.model.unique()):
            raise ValueError(f"{target}: missing V0 reference predictions")
        expected = selected[KEY + [target, "video", "flagged_outliers"]].rename(columns={target: "rating"})
        expected = expected.assign(outer_fold=selected.participant_id.map(assignment))
        if expected.outer_fold.isna().any():
            raise ValueError("Missing external assignment")
        cols = ["rating", "outer_fold", "video", "flagged_outliers"]
        for _, ref in refs.groupby("model"):
            evaluation.check_trial_alignment(expected, ref, cols)
        ids = set(selected.participant_id)
        covered = set()
        for fold, group in target_folds.groupby("outer_fold"):
            external = group[group.inner_fold.isna()]
            train_ids = set(external.loc[external.role.eq("train"), "participant_id"])
            test_ids = set(external.loc[external.role.eq("test"), "participant_id"])
            assigned = set(assignment[assignment.eq(fold)].index) & ids
            if test_ids != assigned or not test_ids or train_ids != ids - test_ids or covered & test_ids:
                raise ValueError(f"{target}/{fold}: inconsistent outer roles")
            covered.update(test_ids)
            inner = []
            validated = set()
            for _, split in group.dropna(subset=["inner_fold"]).groupby("inner_fold"):
                fit = set(split.loc[split.role.eq("train"), "participant_id"])
                val = set(split.loc[split.role.eq("validation"), "participant_id"])
                if not val or not fit or val - train_ids or fit != train_ids - val or val & validated:
                    raise ValueError(f"{target}/{fold}: inconsistent inner roles")
                validated.update(val)
                inner.append((selected.index[selected.participant_id.isin(fit)].to_numpy(),
                              selected.index[selected.participant_id.isin(val)].to_numpy()))
            if validated != train_ids or len(inner) < 2:
                raise ValueError(f"{target}/{fold}: missing inner validation IDs")
            partitions[target, int(fold)] = (selected.index[selected.participant_id.isin(train_ids)].to_numpy(),
                                             selected.index[selected.participant_id.isin(test_ids)].to_numpy(), inner)
        if covered != ids:
            raise ValueError(f"{target}: incomplete external test coverage")
    return rows, partitions, predictions


def resample_window(values, native_rate, target_rate, duration):
    ratio = Fraction(target_rate) / Fraction(str(native_rate))
    converted = resample_poly(values, ratio.numerator, ratio.denominator, axis=-1, padtype="line")
    # Samples at times k / rate strictly before the verified end of the window.
    n = min(converted.shape[-1], int(np.ceil(duration * target_rate - 1e-8)), int(WINDOW_S * target_rate))
    return converted[..., :n].astype(np.float32)


def extract_session(bdf, map_path, rows):
    packed = {s: np.zeros((len(rows), len(EEG) if s == "eeg" else 1, int(WINDOW_S * r)),
                          dtype=np.float32) for s, r in RATES.items()}
    lengths = np.zeros((len(rows), len(RATES)), dtype=np.int32)
    with pyedflib.EdfReader(str(bdf)) as reader:
        signals = Signals(reader, map_path)
        for i, row in enumerate(rows.itertuples()):
            for col, (sensor, rate) in enumerate(RATES.items()):
                names = getattr(row, sensor + "_channels").split(",")
                if sensor == "eeg":
                    if not set(EEG).issubset(names):
                        raise ValueError(f"{bdf.name} marker {row.start_marker}: incomplete V1 EEG montage")
                    names = EEG
                elif len(names) != 1:
                    raise ValueError(f"{bdf.name}: ambiguous {sensor} channel")
                native, source_rate = signals.native_window(sensor, names, row.window_start_s, row.window_end_s)
                if sensor == "gsr":
                    measured, _ = gsr_features(native[0], source_rate, "uS")
                    if not all(np.isclose(value, getattr(row, key), rtol=1e-7, atol=1e-9)
                               for key, value in measured.items()):
                        raise ValueError(f"{bdf.name} marker {row.start_marker}: native GSR disagrees with V0")
                window = resample_window(native, source_rate, rate, row.window_end_s - row.window_start_s)
                lengths[i, col] = window.shape[-1]
                packed[sensor][i, :, :window.shape[-1]] = window
    return {**packed, "lengths": lengths}


def load_windows(root, cache, rows):
    """Store only physical-unit signals. Fold-specific normalization happens in the model."""
    import scipy

    recipe = [digest_file(Path(__file__)), digest_file(Path(__file__).with_name("signals.py")),
              digest_file(Path(__file__).resolve().parents[1] / "starting_notebooks/bdf_loader.py"),
              scipy.__version__, np.__version__, pyedflib.__version__]
    data = {s: np.empty((len(rows), len(EEG) if s == "eeg" else 1, int(WINDOW_S * r)),
                        dtype=np.float32) for s, r in RATES.items()}
    lengths = np.empty((len(rows), len(RATES)), dtype=np.int32)
    stamps = []
    for number, (relative, group) in enumerate(rows.groupby("bdf", sort=True), 1):
        bdf = root / relative
        map_path = root / group.record_map.iloc[0]
        stat = bdf.stat()
        fields = KEY + ["window_start_s", "window_end_s", "gsr_median_us", "gsr_slope_us_s",
                        *(f"{s}_channels" for s in RATES)]
        stamp = hashlib.sha256(json.dumps([recipe, relative, stat.st_size, stat.st_mtime_ns,
                                           digest_file(map_path), group[fields].to_csv(index=False)]).encode()).hexdigest()
        stamps.append(stamp)
        path = cache / f"{stamp}.npz"
        if not path.exists():
            session = extract_session(bdf, map_path, group)
            with atomic_file(path) as temp, temp.open("wb") as handle:
                np.savez(handle, **session)
        with np.load(path, allow_pickle=False) as session:
            for sensor in RATES:
                data[sensor][group.index] = session[sensor]
            lengths[group.index] = session["lengths"]
        if number % 50 == 0:
            LOG.info("%d sessions V1 chargées", number)
    return data, lengths, hashlib.sha256("".join(stamps).encode()).hexdigest()
