import torch

from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig


def test_model_forward_shapes():
    model = ForensicVoice(ForensicVoiceConfig(use_dummy_backbone=True))
    audio = torch.randn(2, 32000)
    out = model(audio, n_speakers=3)
    assert out["speaker_embedding"].shape == (2, 256)
    assert out["authenticity_logits"].shape == (2, 1)
    assert out["diarization_logits"].shape == (2, 3)


def test_model_no_diarization_head():
    model = ForensicVoice(ForensicVoiceConfig(use_dummy_backbone=True))
    audio = torch.randn(2, 32000)
    out = model(audio, n_speakers=None)
    assert out["speaker_embedding"].shape == (2, 256)
    assert out["authenticity_logits"].shape == (2, 1)
    assert out["diarization_logits"] is None
