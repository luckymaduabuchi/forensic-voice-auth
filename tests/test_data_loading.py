from src.data.dataset import ForensicAudioDataset


def test_dataset_dummy_fallback():
    ds = ForensicAudioDataset("does/not/exist.jsonl")
    sample = ds[0]
    assert "audio" in sample
    assert "speaker_id" in sample
    assert "authenticity" in sample
