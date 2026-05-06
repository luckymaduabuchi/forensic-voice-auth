from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pyannote.audio import Model as PyannoteModel
except Exception:  # pragma: no cover - optional dependency at import time
    PyannoteModel = None


class _DummyBackbone(nn.Module):
    """Minimal backbone for unit tests — loads instantly, no HF auth needed."""
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(16000, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, :]  # [B, T]
        if x.shape[-1] > 16000:
            x = x[:, :16000]
        elif x.shape[-1] < 16000:
            x = F.pad(x, (0, 16000 - x.shape[-1]))
        return self.proj(x)  # [B, out_dim]


@dataclass
class ForensicVoiceConfig:
    freeze_backbone: bool = True
    num_attention_heads: int = 4
    embedding_dim: int = 256
    backbone_name: str = "pyannote/embedding"
    use_dummy_backbone: bool = False


class ForensicVoice(nn.Module):
    def __init__(self, config: ForensicVoiceConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_dim = 512
        self._backbone_kind = "pyannote"

        if config.use_dummy_backbone:
            self.backbone_dim = 256
            self.backbone = _DummyBackbone(out_dim=self.backbone_dim)
            self._backbone_kind = "dummy"
        else:
            if PyannoteModel is None:
                raise RuntimeError(
                    "pyannote.audio is not installed or failed to import. "
                    "Install pyannote.audio==3.1.1 in the active environment."
                )
            try:
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                kwargs = {"use_auth_token": hf_token} if hf_token else {}
                self.backbone = PyannoteModel.from_pretrained(config.backbone_name, **kwargs)
                if self.backbone is None and not hf_token:
                    self.backbone = PyannoteModel.from_pretrained(config.backbone_name, use_auth_token=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load pyannote backbone '{config.backbone_name}'. "
                    "Check Hugging Face authentication and model access."
                ) from exc
            if self.backbone is None:
                raise RuntimeError(
                    f"Failed to load pyannote backbone '{config.backbone_name}': "
                    "pyannote returned None. Check the model id and Hugging Face access."
                )

            pyannote_dim = getattr(getattr(self.backbone, "specifications", None), "dimension", None)
            if isinstance(pyannote_dim, int):
                self.backbone_dim = pyannote_dim
            inferred_dim = self._infer_backbone_dim()
            if isinstance(inferred_dim, int):
                self.backbone_dim = inferred_dim

            if config.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.backbone_dim,
            num_heads=config.num_attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.speaker_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.embedding_dim),
        )
        self.authenticity_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.diarization_head: Optional[nn.Linear] = None
        self.n_speakers: Optional[int] = None

    def _infer_backbone_dim(self) -> Optional[int]:
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 16000)
                out = self.backbone(dummy)
                if isinstance(out, dict):
                    if "embedding" in out:
                        out = out["embedding"]
                    elif "logits" in out:
                        out = out["logits"]
                    else:
                        out = next(iter(out.values()))
                if isinstance(out, torch.Tensor):
                    if out.dim() >= 2:
                        return int(out.shape[-1])
        except Exception:
            return None
        return None

    def set_n_speakers(self, n_speakers: int) -> None:
        if self.diarization_head is None or self.n_speakers != n_speakers:
            self.diarization_head = nn.Linear(self.backbone_dim, n_speakers)
            self.n_speakers = n_speakers

    def _extract_features(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveforms = audio.unsqueeze(1)  # [B, 1, T]
        out = self.backbone(waveforms)
        if isinstance(out, dict):
            if "embedding" in out:
                out = out["embedding"]
            elif "logits" in out:
                out = out["logits"]
            else:
                out = next(iter(out.values()))
        if out.dim() == 2:
            out = out.unsqueeze(1)  # [B, 1, D]
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        n_speakers: Optional[int] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        features = self._extract_features(audio, sample_rate=sample_rate)
        if features.size(1) > 1:
            attended, _ = self.attention_pool(features, features, features)
            pooled = attended.mean(dim=1)
        else:
            pooled = features.squeeze(1)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=-1.0)

        speaker_embedding = F.normalize(self.speaker_proj(pooled), p=2, dim=1)
        speaker_embedding = torch.nan_to_num(speaker_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        authenticity_logits = torch.nan_to_num(
            self.authenticity_head(pooled),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

        diarization_logits = None
        if n_speakers is not None:
            self.set_n_speakers(n_speakers)
            if self.diarization_head.weight.device != pooled.device:
                self.diarization_head = self.diarization_head.to(pooled.device)
            diarization_logits = torch.nan_to_num(
                self.diarization_head(pooled),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )

        return {
            "speaker_embedding": speaker_embedding,
            "authenticity_logits": authenticity_logits,
            "diarization_logits": diarization_logits,
        }
