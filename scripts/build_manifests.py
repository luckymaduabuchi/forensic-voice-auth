from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def collect_audio(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def infer_labels(path: Path) -> tuple[int, int]:
    parts = [x.lower() for x in path.parts]
    spoof_markers = ["spoof", "fake", "clone", "synth", "tts", "vc", "asvspoof"]
    is_spoof = any(marker in "/".join(parts) for marker in spoof_markers)
    authenticity = 0 if is_spoof else 1

    # Speaker fallback: hash parent folder name into stable int bucket.
    parent = path.parent.name
    speaker_id = abs(hash(parent)) % 100000
    return speaker_id, authenticity


def build_rows(audio_paths: list[Path], project_root: Path) -> list[dict]:
    rows = []
    for p in audio_paths:
        speaker_id, authenticity = infer_labels(p)
        rel = p.relative_to(project_root).as_posix()
        rows.append(
            {
                "audio_path": rel,
                "speaker_id": speaker_id,
                "authenticity": authenticity,
                "n_speakers": 1000,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def split_rows(rows: list[dict], seed: int, train_ratio: float, val_ratio: float):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test JSONL manifests.")
    parser.add_argument("--project-root", default=".", help="Repo root.")
    parser.add_argument("--data-root", default="data/raw", help="Raw data root.")
    parser.add_argument("--out-dir", default="data/manifests", help="Manifest output dir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    audio_paths = collect_audio(data_root)
    if not audio_paths:
        raise SystemExit(f"No audio files found under: {data_root}")

    rows = build_rows(audio_paths, project_root=project_root)
    train, val, test = split_rows(
        rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    write_jsonl(out_dir / "train_manifest.jsonl", train)
    write_jsonl(out_dir / "val_manifest.jsonl", val)
    write_jsonl(out_dir / "test_manifest.jsonl", test)

    print(f"wrote {len(train)} train, {len(val)} val, {len(test)} test rows to {out_dir}")


if __name__ == "__main__":
    main()
