import torch

from disentanglement.losses import total_loss
from disentanglement.models import ModalityEncoder, MultimodalDisentanglement


def test_modality_encoders_share_a_128_dimensional_output() -> None:
    eeg = ModalityEncoder("eeg", 62, 256)
    ecg = ModalityEncoder("ecg", 1, 256)
    gsr = ModalityEncoder("gsr", 1, 32)
    resp = ModalityEncoder("respiration", 1, 32)

    eeg_output = eeg(torch.randn(2, 62, 768))
    ecg_output = ecg(torch.randn(2, 1, 768))
    gsr_output = gsr(torch.randn(2, 1, 96))
    resp_output = resp(torch.randn(2, 1, 96))

    assert eeg_output.shape == (2, 128)
    assert ecg_output.shape == (2, 128)
    assert gsr_output.shape == (2, 128)
    assert resp_output.shape == (2, 128)


def test_full_model_forward_and_backward() -> None:
    model = MultimodalDisentanglement(
        num_classes=12,
        eeg_channels=62,
    )
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
    )
    loss.backward()

    assert components["logits"].shape == (2, 12)
    assert model.fusion.layers[0].self_attn.num_heads == 4
    assert len(components["invariant"]) == 4
    assert len(components["specific"]) == 4
    assert "private" not in components
    assert "subject_reconstructed" not in components
    assert set(metrics) == {
        "task",
        "modality_similarity",
        "modality_difference",
        "modality_reconstruction",
        "total",
    }


def test_model_supports_modality_ablation() -> None:
    model = MultimodalDisentanglement(
        num_classes=12,
        eeg_channels=62,
        modalities=("eeg", "gsr"),
    )
    inputs = {
        "eeg": torch.randn(2, 62, 768),
        "gsr": torch.randn(2, 1, 96),
    }
    components = model(inputs, return_components=True)

    assert components["logits"].shape == (2, 12)
    assert len(components["invariant"]) == 2
    assert len(components["specific"]) == 2
