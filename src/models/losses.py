from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Stable supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        device = embeddings.device
        batch_size = embeddings.size(0)
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_row = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(positives_per_row, min=1.0)

        valid_rows = positives_per_row > 0
        if valid_rows.sum() == 0:
            return torch.zeros((), device=device, dtype=embeddings.dtype)

        return -mean_log_prob_pos[valid_rows].mean()


class JointForensicLoss(nn.Module):
    def __init__(
        self,
        w_contrastive: float = 1.0,
        w_authenticity: float = 0.1,
        w_diarization: float = 0.05,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.w_contrastive = w_contrastive
        self.w_authenticity = w_authenticity
        self.w_diarization = w_diarization
        self.contrastive_loss = SupConLoss(temperature=temperature)
        self.authenticity_loss = nn.BCEWithLogitsLoss()
        self.diarization_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        auth_logits: torch.Tensor,
        diar_logits: torch.Tensor | None,
        auth_labels: torch.Tensor,
        diar_labels: torch.Tensor | None = None,
        diar_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | float]:
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        auth_logits = torch.nan_to_num(auth_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if diar_logits is not None:
            diar_logits = torch.nan_to_num(diar_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        if diar_mask is not None and diar_labels is not None:
            con_mask = diar_mask.view(-1).bool()
            if con_mask.sum() >= 2:
                loss_contrastive = self.contrastive_loss(embeddings[con_mask], diar_labels[con_mask].long())
            else:
                loss_contrastive = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
        else:
            contrastive_labels = diar_labels if diar_labels is not None else auth_labels
            loss_contrastive = self.contrastive_loss(embeddings, contrastive_labels.long())
        loss_authenticity = self.authenticity_loss(auth_logits, auth_labels.float().unsqueeze(1))

        if diar_logits is not None and diar_labels is not None:
            if diar_mask is not None:
                diar_mask = diar_mask.view(-1).bool()
                if int(diar_mask.sum().item()) > 0:
                    loss_diarization = self.diarization_loss(
                        diar_logits[diar_mask],
                        diar_labels.long()[diar_mask],
                    )
                else:
                    loss_diarization = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            else:
                loss_diarization = self.diarization_loss(diar_logits, diar_labels.long())
        else:
            loss_diarization = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        total_loss = (
            self.w_contrastive * loss_contrastive
            + self.w_authenticity * loss_authenticity
            + self.w_diarization * loss_diarization
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e3, neginf=-1e3)

        return {
            "total_loss": total_loss,
            "contrastive_loss": float(loss_contrastive.detach().cpu().item()),
            "authenticity_loss": float(loss_authenticity.detach().cpu().item()),
            "diarization_loss": float(loss_diarization.detach().cpu().item()),
        }
