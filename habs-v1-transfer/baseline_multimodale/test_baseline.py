"""Regression tests for trial alignment and grouped evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

from . import evaluation, pipeline, trials

ROOT = Path(__file__).resolve().parents[1]
RATE = 100
CODES = (10100, 10102, 10104)


def synthetic_session(root, number, *, flat_resp=False, full_montage=False):
    participant = f"P{number:03}"
    session = root / "PREPROCESSED" / participant
    session.mkdir(parents=True)
    prefix = f"{participant}_day-1"
    bdf = session / f"{prefix}_task-emotion-video_multimodal.bdf"
    times = np.arange(33 * RATE) / RATE
    headers = []
    channels = []
    maps = {}
    layout = [
        ("EEG", "F3", "uV", 4 * np.sin(2 * np.pi * 10 * times)),
        ("EEG", "F4", "uV", 6 * np.sin(2 * np.pi * 10 * times + 0.5)),
        ("ECG", "ECG", "mV", np.exp(-(((times + 0.4) % 0.8 - 0.4) / 0.021) ** 2)),
        ("GSR", "GSR", "uV", -1.8e6 + 9000 * np.sin(times / 4) + 1000 * number),
        ("Respiration", "Resp", "AU", np.zeros_like(times) if flat_resp else np.sin(2 * np.pi * 0.37 * times)),
    ]
    if full_montage:
        for name in ("Fz", "C3", "C4", "P3", "Pz", "P4", "O1", "O2"):
            layout.append(("EEG", name, "uV", 5 * np.sin(2 * np.pi * 9 * times)))
    for i, (sensor, name, unit, values) in enumerate(layout):
        channels.append(values)
        headers.append({
            "label": name, "dimension": unit, "sample_frequency": RATE,
            "physical_min": -2000000 if sensor == "GSR" else -20,
            "physical_max": 2000000 if sensor == "GSR" else 20,
            "digital_min": -8388608, "digital_max": 8388607,
            "transducer": "", "prefilter": "",
        })
        entry = maps.setdefault(sensor, {
            "channel_indices": [], "correction": {"RecordType": sensor, "Channels": []},
        })
        entry["channel_indices"].append(i)
        entry["correction"]["Channels"].append({"name": name, "units": unit})
    events = []
    labels = []
    with pyedflib.EdfWriter(str(bdf), len(channels), file_type=pyedflib.FILETYPE_BDFPLUS) as writer:
        writer.setSignalHeaders(headers)
        writer.writeSamples(channels)
        for i, marker in enumerate(CODES):
            start = i * 11.0
            video = f"clip_{i}.mp4" if i < 2 else f"unique_{number}.mp4"
            writer.writeAnnotation(start, -1, str(marker))
            writer.writeAnnotation(start + 10 + (0.25 if i == 1 else 0), -1, str(marker + 1))
            events.append({
                "marker_id": marker, "duration_s": 10,
                "onset_s": 40 + start + (0.08 if number == 6 and i == 2 else 0),
                "trial_type": "emotion_video", "media_file": video,
                "duration_source": "protocol_csv" if i == 1 else "marker",
            })
            labels.append({
                "participant_id": participant, "cohort": 1, "day": "day-1",
                "start_marker": marker, "end_marker": marker + 1, "video": video.upper(),
                "rating_arousal": 1 + ((number + i) % 8),
                "rating_valence": "" if number == 4 and i == 0 else 9 - ((number + i) % 8),
                "flagged_outliers": int(number == 2 and i == 0),
            })
    bdf.with_suffix(".record_map").write_text(json.dumps({"stream_map": {"s": maps}, "TotalSamples": len(times)}))
    metadata = {
        "events": events, "dataset_export": {"segments": {"emotion-video": {"metadata_time_origin_s": 40}}},
    }
    (session / f"{prefix}_metadata.json").write_text(json.dumps(metadata))
    return labels


class BaselineTest(unittest.TestCase):
    def test_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config = pipeline.process(ROOT / "sample", out, smoke=True)
            self.assertEqual(config["status"], "missing_inputs")
            self.assertEqual(config["input"]["manifest_rows"], 36)
            manifest = pd.read_csv(out / "manifest.csv")
            self.assertTrue(manifest.trial_status.eq("missing_video_bdf").all())
            self.assertAlmostEqual(manifest.metadata_start_s.min(), 0)
            self.assertTrue(pd.read_csv(out / "predictions.csv").empty)
            inspection = json.loads((out / "sample_smoke.json").read_text())
            self.assertEqual(len(inspection), 1)
            self.assertTrue(all(inspection[0]["channels_first_10s"][f"{sensor}_status"] == "ok"
                                for sensor in evaluation.SENSORS))

    def test_full_bdf_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            out = Path(directory) / "first"
            repeat = Path(directory) / "repeat"
            labels = [row for i in range(1, 7) for row in synthetic_session(root, i)]
            labels.append(dict(labels[0]))
            labels[4]["day"] = ""
            pd.DataFrame(labels).to_csv(root / "labels_train.csv", index=False)
            config = pipeline.process(root, out)
            self.assertEqual(config["status"], "evaluated")
            manifest = pd.read_csv(out / "manifest.csv")
            self.assertEqual(manifest.trial_status.value_counts().to_dict(),
                             {"ok": 16, "ambiguous_occurrence": 2, "discordant_timing": 1})
            self.assertEqual(config["input"]["label_rows"], 19)
            self.assertEqual(config["input"]["label_days_recovered"], 1)
            recovered = manifest.loc[manifest.label_day_source.eq("verified_recording")]
            self.assertEqual(recovered.day.item(), 1)
            self.assertEqual(recovered.rating_arousal.item(), labels[4]["rating_arousal"])
            self.assertEqual(manifest.eligible_rating_arousal.sum(), 16)
            self.assertEqual(manifest.eligible_rating_valence.sum(), 15)
            delayed = manifest.loc[manifest.start_marker.eq(10102)]
            self.assertTrue(delayed.trial_status.eq("ok").all())
            self.assertTrue(np.allclose(delayed.bdf_end_minus_metadata_s, 0.25))
            self.assertTrue(np.allclose(delayed.window_end_s - delayed.window_start_s, 10))
            features = pd.read_csv(out / "features.csv")
            self.assertTrue(features.ecg_bpm.between(74, 76).all())
            self.assertTrue(features.resp_cpm.between(20, 25).all())
            self.assertTrue(features.eeg_central_alpha.isna().all())
            self.assertTrue((features.eeg_frontal_alpha > features.eeg_frontal_beta).all())

            predictions = pd.read_csv(out / "predictions.csv")
            self.assertTrue(predictions.prediction.notna().all())
            for target, rows in predictions.groupby("target"):
                required = 16 if target == "rating_arousal" else 15
                self.assertTrue(rows.groupby("model").size().eq(required).all())
                self.assertFalse(rows.duplicated(trials.KEY + ["model"]).any())
                unknown = rows[rows.model.eq("clip_median") & rows.video.str.startswith("unique")]
                self.assertFalse(unknown.clip_seen.any())
                for fold, unseen in unknown.groupby("outer_fold"):
                    train = rows[rows.outer_fold.ne(fold) & rows.model.eq("median")]
                    self.assertTrue(np.allclose(unseen.prediction, train.rating.median()))

            folds = pd.read_csv(out / "folds.csv")
            active = folds[folds.role.ne("assignment")]
            for _, outer in active.groupby(["target", "outer_fold"]):
                outer_test = set(outer.loc[outer.role.eq("test"), "participant_id"])
                self.assertFalse(outer_test & set(outer.loc[outer.inner_fold.notna(), "participant_id"]))
                for _, inner in outer.dropna(subset=["inner_fold"]).groupby("inner_fold"):
                    self.assertFalse(set(inner.loc[inner.role.eq("train"), "participant_id"])
                                     & set(inner.loc[inner.role.eq("validation"), "participant_id"]))

            metrics = pd.read_csv(out / "metrics.csv")
            gains = metrics[metrics.kind.eq("paired_gain") & metrics.subset.eq("all")]
            self.assertEqual(len(gains), 4)
            arousal = predictions[predictions.target.eq("rating_arousal")]
            mae = (arousal.assign(error=(arousal.rating - arousal.prediction).abs())
                   .groupby(["participant_id", "model"]).error.mean().unstack())
            expected_gain = (mae[evaluation.BEST] - mae[evaluation.PRIMARY]).mean()
            measured = gains[gains.target.eq("rating_arousal") & gains.name.str.startswith(evaluation.BEST)]
            self.assertAlmostEqual(measured.gain.item(), expected_gain)

            from process import process as habs_process

            habs_process(root, repeat)
            for name in ("manifest.csv", "folds.csv", "predictions.csv", "metrics.csv", "config.json"):
                self.assertEqual((out / name).read_bytes(), (repeat / name).read_bytes())

    def test_missing_respiration_uses_separate_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            out = Path(directory) / "output"
            labels = [row for i in range(1, 5) for row in synthetic_session(root, i, flat_resp=True)]
            pd.DataFrame(labels).to_csv(root / "labels_train.csv", index=False)
            config = pipeline.process(root, out, development_ids=["P004"])
            self.assertEqual(config["status"], "insufficient_main_population")
            predictions = pd.read_csv(out / "predictions.csv")
            self.assertEqual(set(predictions.population), {"ecg_gsr"})
            self.assertEqual(set(predictions.participant_id), {"P001", "P002", "P003"})
            self.assertIn("ecg_gsr", set(predictions.model))
            manifest = pd.read_csv(out / "manifest.csv")
            self.assertTrue(manifest.resp_issues.eq("resp:flat").all())
            self.assertFalse(manifest.eligible_rating_arousal.any())

    def test_missing_day_is_unresolved_when_two_sessions_match(self):
        manifest = pd.DataFrame([
            {"participant_id": "P001", "start_marker": 10102, "day": day, "n_metadata_events": 1,
             "trial_status": "ok", "n_label_rows": 0, "end_marker": 10103, "metadata_video": "clip.mp4"}
            for day in (1, 2)
        ] + [
            {"participant_id": "P001", "start_marker": 10102, "video": "clip.mp4",
             "labels_end_marker": 10103, "trial_status": "invalid_label_key"}
        ], columns=trials.MANIFEST_COLUMNS)
        result = trials.recover_missing_days(manifest)
        self.assertEqual(result.trial_status.tolist(), ["ok", "ok", "invalid_label_key"])
        self.assertTrue(result.label_day_source.isna().all())
