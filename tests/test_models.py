import torch

from disentanglement.models import ModalityEncoder, MultimodalDisentanglement


def test_modality_encoders_share_a_128_dimensional_output() -> None:
    eeg = ModalityEncoder("eeg", 62, 256)
    gsr = ModalityEncoder("gsr", 1, 32)
    resp = ModalityEncoder("respiration", 1, 32)

    eeg_output = eeg(torch.randn(2, 62, 768))
    gsr_output = gsr(torch.randn(2, 1, 96))
    resp_output = resp(torch.randn(2, 1, 96))

    assert eeg_output.shape == (2, 128)
    assert gsr_output.shape == (2, 128)
    assert resp_output.shape == (2, 128)


def test_full_model_forward_and_backward() -> None:
    model = MultimodalDisentanglement(
        num_classes=12,
        eeg_channels=62,
        training_subjects=["P001", "P002"],
    )
    inputs = {
        "eeg": torch.randn(2, 62, 768),
        "gsr": torch.randn(2, 1, 96),
        "respiration": torch.randn(2, 1, 96),
    }
    logits = model(inputs, subject_ids=torch.tensor([0, 1]))
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (2, 12)
