import torch
from torch import nn

from disentanglement.config import MODALITIES


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
    def __init__(self, feature_dim: int = 128, num_modalities: int = len(MODALITIES)):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
        self.specific = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
            for _ in range(num_modalities)
        ])
        self.modality_reconstruction = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, modality_features: list[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        invariant = [self.shared(item) for item in modality_features]
        specific = [
            branch(item)
            for branch, item in zip(self.specific, modality_features, strict=True)
        ]
        reconstructed = [
            self.modality_reconstruction(item + spec)
            for item, spec in zip(invariant, specific, strict=True)
        ]
        return {"invariant": invariant, "specific": specific, "reconstructed": reconstructed}


class MultimodalDisentanglement(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        eeg_channels: int = 62,
        feature_dim: int = 128,
        modalities: tuple[str, ...] = MODALITIES,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        specs = {
            "eeg": (eeg_channels, 256),
            "ecg": (1, 256),
            "gsr": (1, 32),
            "respiration": (1, 32),
        }
        unknown = set(self.modalities).difference(specs)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(name, *specs[name], feature_dim)
            for name in self.modalities
        })
        self.modality_disentanglement = DisentangledMultimodal(
            feature_dim,
            num_modalities=len(self.modalities),
        )

        tokens = feature_dim
        self.token_type = nn.Parameter(torch.zeros(1, 2 * len(self.modalities), tokens))
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=tokens, nhead=1, batch_first=True),
            num_layers=1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        modality_features = [
            self.encoders[name](inputs[name])
            for name in self.modalities
        ]
        modality = self.modality_disentanglement(modality_features)
        invariant = modality["invariant"]
        specific = modality["specific"]
        tokens = torch.cat([
            torch.stack([inv, spec], dim=1)
            for inv, spec in zip(invariant, specific, strict=True)
        ], dim=1)
        tokens = tokens + self.token_type
        fused = self.fusion(tokens).mean(dim=1)

        logits = self.classifier(fused)
        if not return_components:
            return logits
        return {
            "logits": logits,
            "modality_features": modality_features,
            **modality,
            "fused": fused,
        }
