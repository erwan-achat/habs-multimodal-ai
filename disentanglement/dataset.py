from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from disentanglement.config import PreprocessingConfig
from disentanglement.preprocessing import load_preprocessed_segment, window_offsets


class EmotionWindowDataset(Dataset):
    def __init__(
        self,
        trial_index: pd.DataFrame,
        partition_subjects: set[str],
        partition_videos: set[str],
        label_to_id: dict[str, int],
        subject_to_id: dict[str, int],
        normalization: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        config: PreprocessingConfig | None = None,
        max_trials: int = 0,
    ):
        self.config = config or PreprocessingConfig()
        self.trials = trial_index[
            trial_index["participant_id"].isin(partition_subjects)
            & trial_index["video"].isin(partition_videos)
        ].copy()
        if max_trials > 0:
            self.trials = self.trials.head(max_trials)

        self.label_to_id = label_to_id
        self.subject_to_id = subject_to_id
        self.normalization = normalization or {}
        self.items: list[dict] = []
        for trial in self.trials.itertuples(index=False):
            emotion = str(trial.final_emotion_label)
            if emotion not in label_to_id:
                continue
            for start_s, stop_s in window_offsets(
                duration_s=float(trial.duration_s),
                window_s=self.config.window_duration_s,
                stride_s=self.config.window_stride_s,
            ):
                self.items.append({
                    "bdf_path": Path(trial.bdf_path),
                    "trial_start_s": float(trial.start_s),
                    "trial_duration_s": float(trial.duration_s),
                    "window_start_s": float(start_s),
                    "window_stop_s": float(stop_s),
                    "label": label_to_id[emotion],
                    "subject": subject_to_id.get(str(trial.participant_id), 0),
                })
        self.cache: OrderedDict[tuple[Path, float, float], dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        item = self.items[index]
        bdf_path = item["bdf_path"]
        trial_start = float(item["trial_start_s"])
        trial_duration = float(item["trial_duration_s"])
        cache_key = (bdf_path, trial_start, trial_duration)
        segment = self.cache.get(cache_key)
        if segment is None:
            segment = load_preprocessed_segment(
                bdf_path,
                start_s=trial_start,
                duration_s=trial_duration,
                config=self.config,
            )
            self.cache[cache_key] = segment
            if len(self.cache) > 3:
                self.cache.popitem(last=False)

        start = item["window_start_s"]
        stop = item["window_stop_s"]
        sfreqs = {
            "eeg": self.config.eeg_sfreq,
            "ecg": self.config.ecg_sfreq,
            "gsr": self.config.gsr_sfreq,
            "respiration": self.config.respiration_sfreq,
        }
        tensors = {}
        for modality in self.config.modalities:
            sfreq = sfreqs[modality]
            values = segment[modality][
                :,
                round(start * sfreq):round(stop * sfreq),
            ]
            values = np.asarray(values, dtype=np.float32)
            if modality in self.normalization:
                mean, std = self.normalization[modality]
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                if mean.shape != values.shape:
                    mean = mean.reshape(-1, 1)
                    std = std.reshape(-1, 1)
                values = (values - mean) / (std + 1e-6)
            tensors[modality] = torch.from_numpy(np.asarray(values, dtype=np.float32))

        return {
            **tensors,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "subject": torch.tensor(item["subject"], dtype=torch.long),
        }


class RecordingGroupedSampler(Sampler[int]):
    def __init__(self, items: list[dict], seed: int = 42):
        self.items = items
        self.seed = seed
        groups: dict[str, list[int]] = OrderedDict()
        for index, item in enumerate(items):
            key = str(item["bdf_path"])
            groups.setdefault(key, []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = [list(group) for group in self.groups]
        rng.shuffle(groups)
        order = []
        for group in groups:
            rng.shuffle(group)
            order.extend(group)
        return iter(order)

    def __len__(self) -> int:
        return len(self.items)


class EpochTrialSubsetSampler(Sampler[int]):
    def __init__(
        self,
        items: list[dict],
        trials_per_epoch: int = 512,
        trial_fraction_per_epoch: float = 0.0,
        seed: int = 42,
    ):
        self.items = items
        self.trials_per_epoch = trials_per_epoch
        self.trial_fraction_per_epoch = trial_fraction_per_epoch
        self.seed = seed
        self.epoch = 0

        trial_groups: dict[tuple[str, float, float], list[int]] = OrderedDict()
        for index, item in enumerate(items):
            trial_key = (
                str(item["bdf_path"]),
                float(item["trial_start_s"]),
                float(item["trial_duration_s"]),
            )
            trial_groups.setdefault(trial_key, []).append(index)
        self.trial_groups = list(trial_groups.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _selected_groups(self) -> list[list[int]]:
        trial_count = len(self.trial_groups)
        if trial_count == 0:
            return []
        if self.trials_per_epoch > 0:
            selected_count = min(trial_count, self.trials_per_epoch)
        elif self.trial_fraction_per_epoch > 0.0:
            selected_count = max(
                1,
                min(trial_count, round(trial_count * self.trial_fraction_per_epoch)),
            )
        else:
            selected_count = trial_count

        rng = random.Random(self.seed + self.epoch)
        trial_groups = [list(group) for group in self.trial_groups]
        rng.shuffle(trial_groups)
        return trial_groups[:selected_count]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected = self._selected_groups()
        recording_groups: dict[str, list[list[int]]] = OrderedDict()
        for group in selected:
            bdf_path = str(self.items[group[0]]["bdf_path"])
            recording_groups.setdefault(bdf_path, []).append(group)

        recordings = list(recording_groups.values())
        rng.shuffle(recordings)
        order: list[int] = []
        for trial_groups in recordings:
            rng.shuffle(trial_groups)
            for group in trial_groups:
                rng.shuffle(group)
                order.extend(group)
        return iter(order)

    def __len__(self) -> int:
        return sum(len(group) for group in self._selected_groups())
