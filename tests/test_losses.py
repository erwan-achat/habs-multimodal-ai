import torch

from disentanglement.losses import central_moment_discrepancy, orthogonality_loss


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
