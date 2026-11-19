import torch
import torch.nn.functional as F


def orthogonality_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return (left * right).sum(dim=-1).pow(2).mean()


def central_moment_discrepancy(values: list[torch.Tensor], moments: int = 3) -> torch.Tensor:
    if len(values) < 2:
        return torch.zeros((), device=values[0].device)
    reference = values[0]
    loss = torch.zeros((), device=reference.device)
    for item in values[1:]:
        loss = loss + (item.mean(0) - reference.mean(0)).pow(2).mean()
        for moment in range(2, moments + 1):
            loss = loss + (
                item.pow(moment).mean(0) - reference.pow(moment).mean(0)
            ).pow(2).mean()
    return loss


def total_loss(
    components: dict,
    targets: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = components["logits"]
    task = F.cross_entropy(logits, targets)

    invariant = components["invariant"]
    specific = components["specific"]
    modality_difference = sum(
        orthogonality_loss(inv, spec)
        for inv, spec in zip(invariant, specific, strict=True)
    )
    modality_reconstruction = sum(
        F.mse_loss(reconstructed, original)
        for reconstructed, original in zip(
            components["reconstructed"], components["modality_features"], strict=True
        )
    )

    subject_difference = orthogonality_loss(
        components["shared"], components["private"]
    )
    subject_reconstruction = F.mse_loss(
        components["subject_reconstructed"], components["fused"]
    )

    modality_similarity = central_moment_discrepancy(invariant)
    total = (
        task
        + weights["alpha"] * modality_similarity
        + weights["beta"] * modality_difference
        + weights["gamma"] * modality_reconstruction
        + weights["delta"] * subject_difference
        + weights["epsilon"] * subject_reconstruction
    )
    values = {
        "task": task.detach().item(),
        "modality_similarity": modality_similarity.detach().item(),
        "modality_difference": modality_difference.detach().item(),
        "modality_reconstruction": modality_reconstruction.detach().item(),
        "subject_difference": subject_difference.detach().item(),
        "subject_reconstruction": subject_reconstruction.detach().item(),
        "total": total.detach().item(),
    }
    return total, values
