from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class ForensicAudioDataset(Dataset):
    """
    Minimal dataset that reads JSONL metadata.
    Falls back to random tensors when audio_path is unavailable.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        project_root: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.items = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                "Run scripts/build_manifests_asvspoof_protocol.py to generate manifests."
            )
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        audio_path = item.get("audio_path")
        audio = self._load_or_fallback_audio(
            audio_path,
            segment_start=item.get("segment_start"),
            segment_end=item.get("segment_end"),
        )
        speaker_id = int(item.get("speaker_id", 0))
        authenticity = int(item.get("authenticity", 1))
        n_speakers = int(item.get("n_speakers", max(2, speaker_id + 1)))
        has_diarization = int(item.get("has_diarization", 1 if n_speakers > 0 else 0))
        utt_id = item.get("utt_id", "")
        asvspoof_track = item.get("asvspoof_track", "")

        return {
            "audio": audio,
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
            "authenticity": torch.tensor(authenticity, dtype=torch.long),
            "n_speakers": torch.tensor(n_speakers, dtype=torch.long),
            "has_diarization": torch.tensor(has_diarization, dtype=torch.bool),
            "utt_id": utt_id,
            "asvspoof_track": asvspoof_track,
        }

    def _load_or_fallback_audio(
        self,
        audio_path: str | None,
        segment_start: float | None = None,
        segment_end: float | None = None,
    ) -> torch.Tensor:
        if not audio_path:
            warnings.warn("manifest item missing audio_path; substituting silence")
            return torch.zeros(self.segment_samples, dtype=torch.float32)

        path = Path(audio_path)
        if not path.is_absolute():
            path = self.project_root / path

        if not path.exists():
            warnings.warn(f"audio not found: {path}; substituting silence")
            return torch.zeros(self.segment_samples, dtype=torch.float32)

        try:
            waveform, sr = torchaudio.load(path)
        except Exception as exc:
            warnings.warn(f"failed to load {path}: {exc}; substituting silence")
            return torch.zeros(self.segment_samples, dtype=torch.float32)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if segment_start is not None and segment_end is not None:
            start = max(0, int(float(segment_start) * self.sample_rate))
            end = max(start, int(float(segment_end) * self.sample_rate))
            waveform = waveform[start:end]

        if waveform.numel() < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.numel()))
        elif waveform.numel() > self.segment_samples:
            waveform = waveform[: self.segment_samples]

        return waveform.float()
