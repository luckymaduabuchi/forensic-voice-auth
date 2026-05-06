from __future__ import annotations

"""Build JSONL manifest for the In-the-Wild dataset.

In-the-Wild:
    data/raw/in_the_wild/release_in_the_wild/meta.csv
    columns: file, speaker, label  (label = "bona-fide" | "spoof")

ElevenLabs files are all spoof — no manifest needed.
Run inference.py on data/raw/Elevenlabs/ directly for a qualitative score report.

Output:
    data/manifests/test_in_the_wild_manifest.jsonl

Usage:
    python scripts/build_manifests_in_the_wild.py
"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT_DIR = ROOT / "data/manifests"


def _speaker_hash(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def build_in_the_wild() -> Path:
    meta_path = RAW / "in_the_wild/release_in_the_wild/meta.csv"
    audio_dir = RAW / "in_the_wild/release_in_the_wild"
    out_path = OUT_DIR / "test_in_the_wild_manifest.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = 0
    with meta_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["file"].strip()
            speaker = row["speaker"].strip()
            label = row["label"].strip()

            audio_path = audio_dir / fname
            if not audio_path.exists():
                missing += 1
                continue

            authenticity = 1 if label == "bona-fide" else 0
            rows.append({
                "audio_path": str(audio_path.relative_to(ROOT)),
                "speaker_id": _speaker_hash(speaker),
                "speaker_name": speaker,
                "authenticity": authenticity,
                "n_speakers": 0,
                "has_diarization": 0,
                "dataset": "in_the_wild",
                "label": label,
                "utt_id": audio_path.stem,
            })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    genuine = sum(1 for r in rows if r["authenticity"] == 1)
    spoof = sum(1 for r in rows if r["authenticity"] == 0)
    print(f"In-the-Wild manifest: {len(rows)} entries "
          f"({genuine} bona-fide, {spoof} spoof, {missing} missing audio)")
    print(f"  -> {out_path}")
    return out_path


def main() -> None:
    build_in_the_wild()
    print("\nFor ElevenLabs (all spoof — use inference, not evaluate):")
    print("  python scripts/inference.py \\")
    print("      --model checkpoints/best_forensic_score.pth \\")
    print("      --audio data/raw/Elevenlabs/ \\")
    print("      --out-csv checkpoints/elevenlabs_scores.csv")


if __name__ == "__main__":
    main()
