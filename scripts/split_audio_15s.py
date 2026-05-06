from __future__ import annotations

"""Split every audio file in a directory into fixed-length chunks.

Default input:
    data/raw/openai

Default output:
    data/raw/openai_15s

Each output file is named:
    <original_stem>_chunk0000.wav
    <original_stem>_chunk0001.wav
    ...

The final chunk is padded with silence by default so every chunk is exactly
15 seconds. Use --no-pad-final to keep the final chunk shorter, or
--drop-final to discard a short final chunk.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    ]
    return sorted(files)


def _safe_rel_stem(path: Path, input_dir: Path) -> str:
    rel = path.relative_to(input_dir).with_suffix("")
    return "__".join(rel.parts)


def _load_audio(path: Path, sample_rate: int | None, mono: bool) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    if sample_rate is not None and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def split_file(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    chunk_seconds: float,
    sample_rate: int | None,
    mono: bool,
    pad_final: bool,
    drop_final: bool,
) -> list[dict[str, object]]:
    waveform, sr = _load_audio(path, sample_rate=sample_rate, mono=mono)
    chunk_samples = int(round(chunk_seconds * sr))
    if chunk_samples <= 0:
        raise ValueError("--chunk-seconds must produce at least one sample")

    total_samples = waveform.size(1)
    if total_samples == 0:
        return []

    stem = _safe_rel_stem(path, input_dir)
    rows: list[dict[str, object]] = []
    chunk_idx = 0

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        is_final_short = chunk.size(1) < chunk_samples

        if is_final_short and drop_final:
            break
        if is_final_short and pad_final:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.size(1)))

        out_path = output_dir / f"{stem}_chunk{chunk_idx:04d}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), chunk, sr)

        rows.append(
            {
                "source_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "audio_path": str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path),
                "chunk_index": chunk_idx,
                "start_seconds": start / sr,
                "end_seconds": end / sr,
                "duration_seconds": chunk.size(1) / sr,
                "source_duration_seconds": total_samples / sr,
                "sample_rate": sr,
                "num_channels": int(chunk.size(0)),
                "padded": bool(is_final_short and pad_final),
            }
        )
        chunk_idx += 1

    return rows


def write_manifest(rows: list[dict[str, object]], output_dir: Path) -> None:
    jsonl_path = output_dir / "manifest.jsonl"
    csv_path = output_dir / "manifest.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    fieldnames = [
        "source_path",
        "audio_path",
        "chunk_index",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "source_duration_seconds",
        "sample_rate",
        "num_channels",
        "padded",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved manifest: {jsonl_path}")
    print(f"saved csv: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split audio files into fixed-length chunks.")
    parser.add_argument("--input-dir", default="data/raw/openai")
    parser.add_argument("--output-dir", default="data/raw/openai_15s")
    parser.add_argument("--chunk-seconds", type=float, default=15.0)
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional resample rate, e.g. 16000.")
    parser.add_argument("--mono", action="store_true", help="Convert chunks to mono.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--no-pad-final", action="store_true", help="Keep the final chunk short instead of padding.")
    parser.add_argument("--drop-final", action="store_true", help="Drop the final chunk if it is shorter than chunk length.")
    args = parser.parse_args()

    input_dir = _project_path(args.input_dir).resolve()
    output_dir = _project_path(args.output_dir).resolve()
    pad_final = not args.no_pad_final

    if args.drop_final and args.no_pad_final:
        print("ERROR: choose either --drop-final or --no-pad-final, not both", file=sys.stderr)
        raise SystemExit(2)
    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    files = _audio_files(input_dir, recursive=args.recursive)
    if not files:
        print(f"ERROR: no audio files found in {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    print(f"input: {input_dir}")
    print(f"output: {output_dir}")
    print(f"files: {len(files)}")
    print(f"chunk_seconds: {args.chunk_seconds}")

    for idx, path in enumerate(files, start=1):
        try:
            rows = split_file(
                path=path,
                input_dir=input_dir,
                output_dir=output_dir,
                chunk_seconds=args.chunk_seconds,
                sample_rate=args.sample_rate,
                mono=args.mono,
                pad_final=pad_final,
                drop_final=args.drop_final,
            )
        except Exception as exc:
            print(f"[{idx}/{len(files)}] FAILED {path}: {exc}", file=sys.stderr)
            continue

        all_rows.extend(rows)
        print(f"[{idx}/{len(files)}] {path.name}: wrote {len(rows)} chunks")

    write_manifest(all_rows, output_dir)
    print(f"done: wrote {len(all_rows)} chunks")


if __name__ == "__main__":
    main()
