from typing import Iterable

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    def __init__(self, modality: str, channels: int, input_sfreq: int, feature_dim: int = 128):
        super().__init__()
        if modality == "eeg":
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 7), padding="same"),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 16)),
                nn.Flatten(),
                nn.Linear(16 * 4 * 16, feature_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(channels, 16, kernel_size=5, padding="same"),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten(),
                nn.Linear(16 * 16, feature_dim),
            )
        self.modality = modality

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.modality == "eeg":
            values = values.unsqueeze(1)
        return self.net(values)


class DisentangledMultimodal(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
        self.specific = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
            for _ in range(3)
        ])
        self.modality_reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, modality_features: list[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        invariant = [self.shared(item) for item in modality_features]
        specific = [branch(item) for branch, item in zip(self.specific, modality_features, strict=True)]
        reconstructed = [self.modality_reconstruction(item + spec) for item, spec in zip(invariant, specific, strict=True)]
        return {"invariant": invariant, "specific": specific, "reconstructed": reconstructed}


class SubjectDisentanglement(nn.Module):
    def __init__(self, feature_dim: int = 128, training_subjects: Iterable[str] | None = None):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
        subjects = list(training_subjects or [])
        self.subject_ids = {subject: index for index, subject in enumerate(subjects)}
        self.private = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
            for _ in subjects
        ])
        self.reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, fused: torch.Tensor, subject_indices: torch.Tensor | None) -> dict[str, torch.Tensor]:
        shared = self.shared(fused)
        batch_size = fused.shape[0]
        if subject_indices is None:
            private = torch.zeros_like(shared)
        else:
            private = torch.stack([
                self.private[int(index)](fused_row)
                for index, fused_row in zip(subject_indices.cpu(), fused)
            ]) if len(subject_indices) else torch.zeros_like(shared)
        reconstructed = self.reconstruction(shared + private)
        return {"shared": shared, "private": private, "subject_reconstructed": reconstructed}


class MultimodalDisentanglement(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        eeg_channels: int = 62,
        training_subjects: Iterable[str] | None = None,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.eeg_encoder = ModalityEncoder("eeg", eeg_channels, 256, feature_dim)
        self.gsr_encoder = ModalityEncoder("gsr", 1, 32, feature_dim)
        self.respiration_encoder = ModalityEncoder("respiration", 1, 32, feature_dim)
        self.modality_disentanglement = DisentangledMultimodal(feature_dim)

        tokens = feature_dim
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=tokens, nhead=8, batch_first=True),
            num_layers=1,
        )
        self.subject_disentanglement = SubjectDisentanglement(feature_dim, training_subjects)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        subject_ids: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        eeg = self.eeg_encoder(inputs["eeg"])
        gsr = self.gsr_encoder(inputs["gsr"])
        respiration = self.respiration_encoder(inputs["respiration"])
        modality_features = [eeg, gsr, respiration]
        modality = self.modality_disentanglement(modality_features)

        invariant = modality["invariant"]
        specific = modality["specific"]
        tokens = torch.cat([
            torch.stack([inv, spec], dim=1)
            for inv, spec in zip(invariant, specific, strict=True)
        ], dim=1)
        fused_tokens = self.fusion(tokens)
        fused = fused_tokens.mean(dim=1)

        subject = self.subject_disentanglement(fused, subject_ids)
        logits = self.classifier(subject["shared"])
        if not return_components:
            return logits
        return {
            "logits": logits,
            "modality_features": modality_features,
            **modality,
            "fused": fused,
            **subject,
        }
