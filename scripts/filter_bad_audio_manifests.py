from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _check_row(args: tuple[int, str, str]) -> tuple[int, str, bool, str]:
    idx, line, project_root_str = args
    project_root = Path(project_root_str)
    try:
        item: dict[str, Any] = json.loads(line)
    except Exception as exc:
        return idx, line, False, f"json_error: {exc}"

    audio_path = item.get("audio_path")
    if not audio_path:
        return idx, line, False, "missing_audio_path"

    path = Path(audio_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return idx, line, False, f"missing_file: {path}"

    try:
        import torchaudio

        waveform, _ = torchaudio.load(path)
        if waveform.numel() == 0:
            return idx, line, False, "empty_waveform"
    except Exception as exc:
        return idx, line, False, f"decode_error: {exc}"

    return idx, line, True, ""


def filter_manifest(manifest_path: Path, output_path: Path, bad_path: Path, project_root: Path, workers: int) -> None:
    rows = [(idx, line, str(project_root)) for idx, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()) if line.strip()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.parent.mkdir(parents=True, exist_ok=True)

    good = 0
    bad = 0
    with output_path.open("w", encoding="utf-8") as out_f, bad_path.open("w", encoding="utf-8") as bad_f:
        if workers <= 1:
            iterator = map(_check_row, rows)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(_check_row, rows, chunksize=64)
        try:
            for idx, line, ok, reason in iterator:
                if ok:
                    out_f.write(line + "\n")
                    good += 1
                else:
                    item = {"row": idx, "reason": reason}
                    try:
                        item.update(json.loads(line))
                    except Exception:
                        item["raw_line"] = line
                    bad_f.write(json.dumps(item) + "\n")
                    bad += 1
        finally:
            if workers > 1:
                executor.shutdown(wait=True)

    print(f"{manifest_path.name}: kept={good} removed={bad} -> {output_path}")
    if bad:
        print(f"  bad report: {bad_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove unreadable audio rows from JSONL manifests.")
    parser.add_argument("--project-root", default=".", help="Repository/project root for relative audio paths.")
    parser.add_argument("--out-dir", default="data/manifests/clean", help="Directory for cleaned manifests.")
    parser.add_argument("--report-dir", default="checkpoints/audio_reports", help="Directory for bad-audio JSONL reports.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("manifests", nargs="+", help="Manifest JSONL files to clean.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    report_dir = (project_root / args.report_dir).resolve()

    for manifest in args.manifests:
        manifest_path = (project_root / manifest).resolve()
        output_path = out_dir / f"{manifest_path.stem}_clean.jsonl"
        bad_path = report_dir / f"{manifest_path.stem}_bad_audio.jsonl"
        filter_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            bad_path=bad_path,
            project_root=project_root,
            workers=max(1, int(args.workers)),
        )


if __name__ == "__main__":
    main()
