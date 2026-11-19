from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        max_trials: int = 0,
    ):
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
                window_s=3.0,
                stride_s=2.0,
            ):
                self.items.append({
                    "bdf_path": Path(trial.bdf_path),
                    "trial_start_s": float(trial.start_s),
                    "window_start_s": float(start_s),
                    "window_stop_s": float(stop_s),
                    "label": label_to_id[emotion],
                    "subject": subject_to_id.get(str(trial.participant_id), 0),
                })
        self.cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        item = self.items[index]
        bdf_path = item["bdf_path"]
        segment = self.cache.get(bdf_path)
        if segment is None:
            segment = load_preprocessed_segment(
                bdf_path,
                start_s=float(item["trial_start_s"]),
                duration_s=10.0,
            )
            self.cache[bdf_path] = segment
            if len(self.cache) > 2:
                self.cache.popitem(last=False)

        start = item["window_start_s"]
        eeg = segment["eeg"][:, round(start * 256):round(item["window_stop_s"] * 256)]
        gsr = segment["gsr"][:, round(start * 32):round(item["window_stop_s"] * 32)]
        respiration = segment["respiration"][:, round(start * 32):round(item["window_stop_s"] * 32)]

        modality_data = {"eeg": eeg, "gsr": gsr, "respiration": respiration}
        tensors = {}
        for modality, values in modality_data.items():
            values = np.asarray(values, dtype=np.float32)
            if modality in self.normalization:
                mean, std = self.normalization[modality]
                if mean.shape[-1] == values.shape[-1]:
                    values = (values - mean) / (std + 1e-6)
                else:
                    values = (values - mean.reshape(-1, 1)) / (std.reshape(-1, 1) + 1e-6)
            tensors[modality] = torch.from_numpy(np.asarray(values, dtype=np.float32))

        return {
            **tensors,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "subject": torch.tensor(item["subject"], dtype=torch.long),
        }
