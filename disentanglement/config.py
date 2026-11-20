from dataclasses import dataclass


MODALITIES: tuple[str, ...] = ("eeg", "ecg", "gsr", "respiration")


@dataclass(frozen=True)
class PreprocessingConfig:
    trial_duration_s: float = 10.0
    window_duration_s: float = 3.0
    window_stride_s: float = 2.0
    eeg_sfreq: float = 256.0
    ecg_sfreq: float = 256.0
    gsr_sfreq: float = 32.0
    respiration_sfreq: float = 32.0
    modalities: tuple[str, ...] = MODALITIES


@dataclass(frozen=True)
class TrainingConfig:
    labels_filename: str = "labels_train.csv"
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    num_workers: int = 0
    train_fraction: float = 0.8
    validation_fraction: float = 0.2
    max_normalization_trials: int = 64
    primary_metric: str = "macro_f1"
    patience: int = 6
    min_delta: float = 1e-4
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    alpha: float = 0.002
    beta: float = 0.1
    gamma: float = 0.04
    delta: float = 0.5
    epsilon: float = 0.05
