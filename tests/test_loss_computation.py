import torch

from src.models.losses import JointForensicLoss


def test_joint_loss_is_finite():
    loss_fn = JointForensicLoss()
    embeddings = torch.nn.functional.normalize(torch.randn(8, 256), dim=1)
    auth_logits = torch.randn(8, 1)
    diar_logits = torch.randn(8, 4)
    auth_labels = torch.randint(0, 2, (8,))
    diar_labels = torch.randint(0, 4, (8,))
    out = loss_fn(embeddings, auth_logits, diar_logits, auth_labels, diar_labels)
    assert torch.isfinite(out["total_loss"]).item()
