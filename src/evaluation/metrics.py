from __future__ import annotations

import torch


def compute_authenticity_metrics(
    probabilities: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    preds = (probabilities.view(-1) >= threshold).long()
    labels = labels.view(-1).long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
