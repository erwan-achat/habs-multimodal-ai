import torch
import torch.nn.functional as F


def orthogonality_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return (left * right).sum(dim=-1).pow(2).mean()


def central_moment(values: torch.Tensor, order: int) -> torch.Tensor:
    mean = values.mean(dim=0, keepdim=True)
    if order == 1:
        return mean.squeeze(0)
    return (values - mean).pow(order).mean(dim=0)


def central_moment_discrepancy(values: list[torch.Tensor], moments: int = 3) -> torch.Tensor:
    if len(values) < 2:
        return torch.zeros((), device=values[0].device)
    loss = torch.zeros((), device=values[0].device)
    pairs = 0
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            for order in range(1, moments + 1):
                loss = loss + (
                    central_moment(values[left_index], order)
                    - central_moment(values[right_index], order)
                ).pow(2).mean()
            pairs += 1
    return loss / max(pairs, 1)


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
    modality_difference = modality_difference + sum(
        orthogonality_loss(specific[left], specific[right])
        for left in range(len(specific))
        for right in range(left + 1, len(specific))
    )
    modality_reconstruction = sum(
        F.mse_loss(reconstructed, original)
        for reconstructed, original in zip(
            components["reconstructed"], components["modality_features"], strict=True
        )
    )

    modality_similarity = central_moment_discrepancy(invariant)
    total = (
        task
        + weights["alpha"] * modality_similarity
        + weights["beta"] * modality_difference
        + weights["gamma"] * modality_reconstruction
    )
    values = {
        "task": task.detach().item(),
        "modality_similarity": modality_similarity.detach().item(),
        "modality_difference": modality_difference.detach().item(),
        "modality_reconstruction": modality_reconstruction.detach().item(),
        "total": total.detach().item(),
    }
    return total, values
