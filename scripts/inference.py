from __future__ import annotations

"""Run inference on one file or a directory of audio files.

Reports authenticity score (0=spoof, 1=genuine) per file.
Use this for qualitative evaluation on unlabelled or single-class data
(e.g. ElevenLabs TTS where all files are known spoof).

Usage — single file:
    python scripts/inference.py \
        --model checkpoints/best_forensic_score.pth \
        --audio data/raw/Elevenlabs/ElevenLabs_voxconverse.wav

Usage — directory:
    python scripts/inference.py \
        --model checkpoints/best_forensic_score.pth \
        --audio data/raw/Elevenlabs/ \
        --out-csv checkpoints/elevenlabs_scores.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torchaudio
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_model(checkpoint: Path, model_cfg: dict, device: torch.device) -> ForensicVoice:
    model = ForensicVoice(ForensicVoiceConfig(**model_cfg)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def score_file(
    model: ForensicVoice,
    path: Path,
    device: torch.device,
    segment_seconds: float,
    sample_rate: int,
) -> float:
    """Returns mean authenticity probability over non-overlapping segments."""
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return float("nan")

    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(0)
    else:
        wav = wav.squeeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    seg_len = int(segment_seconds * sample_rate)
    if wav.numel() < seg_len:
        wav = torch.nn.functional.pad(wav, (0, seg_len - wav.numel()))

    segments = wav.unfold(0, seg_len, seg_len)  # (N, seg_len)
    scores = []
    with torch.no_grad():
        for chunk in segments.split(8):  # batch of 8
            out = model(chunk.to(device), n_speakers=None)
            probs = torch.sigmoid(out["authenticity_logits"]).cpu().view(-1)
            scores.extend(probs.tolist())

    return float(sum(scores) / len(scores)) if scores else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score audio files for authenticity.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True, help="File or directory of audio files.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg["data"].get("sample_rate", 16000)
    seg = cfg["data"].get("segment_seconds", 2.0)
    threshold = args.decision_threshold

    model = load_model(ROOT / args.model, cfg["model"], device)

    audio_path = Path(args.audio)
    if audio_path.is_dir():
        files = sorted(f for f in audio_path.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    else:
        files = [audio_path]

    if not files:
        print(f"No audio files found at: {audio_path}")
        return

    print(f"\n{'File':<45} {'Score':>7} {'Decision':>10}")
    print("-" * 65)

    rows = []
    for f in files:
        score = score_file(model, f, device, seg, sr)
        decision = "genuine" if score >= threshold else "SPOOF"
        print(f"{f.name:<45} {score:>7.4f} {decision:>10}")
        rows.append({"file": str(f), "score": score, "decision": decision})

    spoof_count = sum(1 for r in rows if r["decision"] == "SPOOF")
    genuine_count = sum(1 for r in rows if r["decision"] == "genuine")
    print(f"\nSummary: {spoof_count}/{len(rows)} flagged as SPOOF  |  "
          f"{genuine_count}/{len(rows)} classified as genuine  |  "
          f"threshold={threshold}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "score", "decision"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
