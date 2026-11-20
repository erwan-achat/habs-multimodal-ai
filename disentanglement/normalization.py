from pathlib import Path
import random

import numpy as np
import pandas as pd

from disentanglement.config import PreprocessingConfig
from disentanglement.preprocessing import load_preprocessed_segment


def fit_normalization(
    trial_index: pd.DataFrame,
    subjects: set[str],
    videos: set[str],
    max_trials: int = 64,
    seed: int = 42,
    config: PreprocessingConfig | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    config = config or PreprocessingConfig()
    train = trial_index[
        trial_index["participant_id"].isin(subjects)
        & trial_index["video"].isin(videos)
    ]
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    selected = train.iloc[indices[:max_trials]] if max_trials > 0 else train

    sums: dict[str, np.ndarray] = {}
    squares: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for row in selected.itertuples(index=False):
        segment = load_preprocessed_segment(
            Path(row.bdf_path),
            start_s=float(row.start_s),
            duration_s=float(row.duration_s),
            config=config,
        )
        for modality, values in segment.items():
            values = np.asarray(values, dtype=np.float64)
            if modality not in sums:
                sums[modality] = np.zeros(values.shape[0], dtype=np.float64)
                squares[modality] = np.zeros(values.shape[0], dtype=np.float64)
                counts[modality] = 0
            sums[modality] += values.sum(axis=-1)
            squares[modality] += (values ** 2).sum(axis=-1)
            counts[modality] += values.shape[-1]

    return {
        modality: (
            sums[modality] / counts[modality],
            np.sqrt(np.maximum(
                squares[modality] / counts[modality]
                - (sums[modality] / counts[modality]) ** 2,
                0,
            )),
        )
        for modality in sums
    }
