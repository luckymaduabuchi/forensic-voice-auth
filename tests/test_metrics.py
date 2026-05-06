import torch

from src.evaluation.metrics import compute_authenticity_metrics


def test_metrics_range():
    probs = torch.tensor([0.9, 0.2, 0.8, 0.1])
    labels = torch.tensor([1, 0, 1, 0])
    metrics = compute_authenticity_metrics(probs, labels)
    assert 0.0 <= metrics["f1"] <= 1.0
