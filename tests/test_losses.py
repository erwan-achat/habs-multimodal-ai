import torch

from disentanglement.losses import (
    central_moment_discrepancy,
    orthogonality_loss,
    total_loss,
)
from disentanglement.models import MultimodalDisentanglement


def test_central_moment_discrepancy_is_zero_for_identical_distributions() -> None:
    values = torch.randn(8, 16)
    assert central_moment_discrepancy([values, values, values]).item() == 0.0


def test_central_moment_discrepancy_detects_different_variances() -> None:
    low = torch.randn(64, 16)
    high = low * 3.0
    assert central_moment_discrepancy([low, high]).item() > 0.0


def test_orthogonality_loss_is_zero_for_orthogonal_vectors() -> None:
    left = torch.tensor([[1.0, 0.0]])
    right = torch.tensor([[0.0, 1.0]])
    assert orthogonality_loss(left, right).item() == 0.0


def test_orthogonality_loss_penalizes_aligned_vectors() -> None:
    vectors = torch.tensor([[1.0, 0.0]])
    assert orthogonality_loss(vectors, vectors).item() == 1.0


def test_total_loss_supports_class_weights_and_label_smoothing() -> None:
    model = MultimodalDisentanglement(num_classes=12, eeg_channels=62)
    inputs = {
        "eeg": torch.randn(2, 62, 768),
        "ecg": torch.randn(2, 1, 768),
        "gsr": torch.randn(2, 1, 96),
        "respiration": torch.randn(2, 1, 96),
    }
    components = model(inputs, return_components=True)
    loss, metrics = total_loss(
        components,
        torch.tensor([0, 1]),
        {"alpha": 0.002, "beta": 0.1, "gamma": 0.04},
        class_weights=torch.ones(12),
        label_smoothing=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "task" in metrics
