from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    trial_duration_s: float = 10.0
    window_duration_s: float = 3.0
    window_stride_s: float = 2.0
    eeg_sfreq: float = 256.0
    gsr_sfreq: float = 32.0
    respiration_sfreq: float = 32.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    max_trials_per_epoch: int = 0
    alpha: float = 0.002
    beta: float = 0.1
    gamma: float = 0.04
    delta: float = 0.5
    epsilon: float = 0.05
