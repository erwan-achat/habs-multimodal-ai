"""Regression tests for frozen-fold V1 training, boundary masking and physical inputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from . import evaluation, pipeline, v1
from .signals import FEATURE_COLUMNS
from .test_baseline import synthetic_session
from .v1_data import EEG, RATES, frozen_study, resample_window
from .v1_model import Banks, Regressor


class V1Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)

    def test_native_rates_and_anti_alias(self):
        for rate in (998, 999, 1000):
            for seconds in (10, 9.999):
                t = np.arange(round(rate * seconds)) / rate
                x = np.stack([np.sin(2 * np.pi * f * t) for f in (5, 180)])
                down = resample_window(x, rate, 250, seconds)
                self.assertEqual(down.shape, (2, 2500))
                expected = np.sin(2 * np.pi * 5 * np.arange(2500) / 250)
                np.testing.assert_allclose(down[0, 20:-20], expected[20:-20], atol=0.003)
                self.assertLess(np.max(np.abs(down[1, 20:-20])), 0.005)

    def test_padding_does_not_change_valid_tokens_or_regression(self):
        model = Regressor("v1_four_modalities").eval()
        single, lengths, batch, batch_lengths = {}, {}, {}, {}
        for sensor, hz in RATES.items():
            n = hz * 10 - 1
            x = torch.randn(1, len(EEG) if sensor == "eeg" else 1, n)
            single[sensor], lengths[sensor] = x, torch.tensor([n])
            batch[sensor] = torch.cat([torch.nn.functional.pad(x, (0, 19), value=99.0),
                                       torch.randn(1, x.size(1), n + 19)])
            batch_lengths[sensor] = torch.tensor([n, n + 19])
            a, m = model.branches[sensor](x, lengths[sensor])
            b, _ = model.branches[sensor](batch[sensor], batch_lengths[sensor])
            self.assertEqual(a.shape, (1, 5, 32))
            torch.testing.assert_close(a[m], b[:1][m])
        torch.testing.assert_close(model(single, lengths), model(batch, batch_lengths)[:1])

    def test_overfit_and_training_statistics(self):
        n = 8
        values = np.repeat(np.arange(n, dtype=np.float32)[:, None, None], 2500, axis=2)
        rows = pd.DataFrame(np.zeros((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        rows.iloc[:4, 0] = np.nan
        rows.iloc[4:, 0] = 1e8
        lengths = np.full((n, 4), 2500)
        banks = Banks({"ecg": values}, lengths, rows, "cpu")
        train = np.arange(4)
        descriptor = Regressor("descriptor_mlp")
        banks.configure(descriptor, train)
        self.assertEqual(descriptor.feature_used[0].item(), False)
        labels = torch.arange(n, dtype=torch.float32) + 1
        model, _ = v1.fit("v1_ecg", banks, labels, train, None, 0.005, 0.0001, 160, v1.Training())
        self.assertAlmostEqual(model.branches["ecg"].mean.item(), 1.5)
        pred = v1.predict(model, banks, torch.tensor(train), 4)
        self.assertLess(np.mean(abs(pred - labels[train].numpy())), 0.15)

    def test_full_study_replay_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            v0, out = Path(tmp) / "v0", Path(tmp) / "v1"
            labels = [r for n in range(1, 5) for r in synthetic_session(root, n, full_montage=True)]
            pd.DataFrame(labels).to_csv(root / "labels_train.csv", index=False)
            pipeline.process(root, v0)
            settings = v1.Training(epochs=1, rates=(0.001,), decays=(0.0001,))
            config = v1.process(root, out, v0, device="cpu", settings=settings)
            self.assertEqual(config["status"], "evaluated")
            predictions = pd.read_csv(out / "predictions.csv")
            for target, g in predictions.groupby("target"):
                ref = g[g.model.eq("four_modalities")]
                for _, result in g.groupby("model"):
                    evaluation.check_paired_rows(ref, result)
                for damaged in (ref.iloc[:-1], pd.concat([ref, ref.iloc[:1]]), ref.assign(outer_fold=0),
                                ref.assign(rating=0)):
                    with self.assertRaises(ValueError):
                        evaluation.check_paired_rows(ref, damaged)
            selection = pd.read_csv(out / "selection.csv")
            self.assertEqual(len(selection), 4 * 2 * 7)
            for s in selection.itertuples():
                _, saved = v1.load_model(out / s.checkpoint)
                test = predictions.query("target == @s.target and outer_fold == @s.outer_fold")
                self.assertFalse(set(test.participant_id) & set(saved["train_ids"]))
            before = (out / "predictions.csv").read_bytes()
            with patch.object(v1, "fit", side_effect=AssertionError("Unexpected refit")):
                v1.process(root, out, v0, device="cpu", settings=settings)
            self.assertEqual(before, (out / "predictions.csv").read_bytes())
            folds = pd.read_csv(v0 / "folds.csv")
            index = folds.index[folds.role.eq("validation")][0]
            folds.loc[index, "role"] = "train"
            folds.to_csv(v0 / "folds.csv", index=False)
            with self.assertRaisesRegex(ValueError, "inner roles"):
                frozen_study(v0)

    def test_habs_gpu_entrypoint_builds_reference_then_runs_v1(self):
        import os
        from process import process

        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "glob", return_value=iter(["video"])), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(pipeline, "process") as baseline, patch.object(v1, "process") as neural:
            process(Path("/data"), Path("/out"))
        baseline.assert_called_once_with(Path("/data"), Path("/out/v0"))
        settings = v1.Training(epochs=3, rates=(0.001,), decays=(0.0001,))
        neural.assert_called_once_with(Path("/data"), Path("/out/v1"), Path("/out/v0"),
                                       device="cuda", settings=settings)

        with patch.dict(os.environ, {"HABS_V0_DIR": "/reference"}, clear=True), \
             patch.object(Path, "glob", return_value=iter(["video"])), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(pipeline, "process") as baseline, patch.object(v1, "process") as neural:
            process(Path("/data"), Path("/out"))
        baseline.assert_not_called()
        neural.assert_called_once_with(Path("/data"), Path("/out/v1"), Path("/reference"),
                                       device="cuda", settings=settings)
