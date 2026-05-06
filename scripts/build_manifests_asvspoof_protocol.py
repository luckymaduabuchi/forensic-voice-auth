from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000_000


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_csv_arg(value: str) -> set[str]:
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def load_asvspoof_rows(
    project_root: Path,
    asv_root: Path,
    tracks: set[str],
    train_phases: set[str],
    test_phases: set[str],
) -> tuple[list[dict], list[dict]]:
    overlap = train_phases & test_phases
    if overlap:
        raise ValueError(
            "ASVspoof train/test phase overlap would leak test utterances into training: "
            f"{','.join(sorted(overlap))}"
        )

    flac_index: dict[str, Path] = {}
    for flac in asv_root.rglob("*.flac"):
        flac_index[flac.stem] = flac

    # Only load CM (countermeasure) metadata — exclude ASV (speaker verification) protocol files.
    metadata_files = [
        p for p in asv_root.rglob("trial_metadata.txt")
        if "/CM/" in p.as_posix() and "/ASV/" not in p.as_posix()
    ]
    if not metadata_files:
        raise RuntimeError(f"No CM/trial_metadata.txt found under {asv_root}")

    meta_by_utt: dict[str, tuple[str, str, str, str]] = {}
    # utt_id -> (source_speaker, label, track, phase)
    for meta_path in metadata_files:
        track = "DF" if "/DF/" in meta_path.as_posix() else ("LA" if "/LA/" in meta_path.as_posix() else "PA")
        if track not in tracks:
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                source_speaker = parts[0]
                utt_id = parts[1]
                # PA CM metadata has 10+ columns; label position varies by track.
                # Search for the label token rather than assuming a fixed index.
                label = next((p.lower() for p in parts if p.lower() in ("bonafide", "spoof")), None)
                if label is None:
                    continue
                phase = next((p.lower() for p in parts if p.lower() in ("progress", "eval", "hidden")), "unknown")
                meta_by_utt[utt_id] = (source_speaker, label, track, phase)

    def build_row(utt_id: str) -> dict | None:
        meta = meta_by_utt.get(utt_id)
        if meta is None:
            return None
        source_speaker, label, track, phase = meta
        audio_path = flac_index.get(utt_id)
        if audio_path is None:
            return None
        return {
            "audio_path": audio_path.relative_to(project_root).as_posix(),
            "speaker_id": stable_int(source_speaker),
            "authenticity": 1 if label == "bonafide" else 0,
            "n_speakers": 0,
            "has_diarization": 0,
            "dataset": f"asvspoof2021_{track.lower()}",
            "asvspoof_track": track,
            "asvspoof_phase": phase,
            "utt_id": utt_id,
        }

    all_ids = set(meta_by_utt.keys())
    train_val_ids = {uid for uid in all_ids if meta_by_utt[uid][3] in train_phases}
    eval_ids = {uid for uid in all_ids if meta_by_utt[uid][3] in test_phases}

    train_val_rows = [r for uid in sorted(train_val_ids) if (r := build_row(uid)) is not None]
    eval_rows = [r for uid in sorted(eval_ids) if (r := build_row(uid)) is not None]

    missing = sum(1 for uid in all_ids if flac_index.get(uid) is None)
    if missing:
        print(f"warning: {missing} ASVspoof metadata utterances had no matching flac")
    if not train_val_rows:
        raise RuntimeError(
            f"No ASVspoof train/val rows found for tracks={sorted(tracks)} "
            f"and train phases={sorted(train_phases)}"
        )
    if not eval_rows:
        raise RuntimeError(
            f"No ASVspoof test rows found for tracks={sorted(tracks)} "
            f"and test phases={sorted(test_phases)}"
        )
    return train_val_rows, eval_rows


def parse_rttm_speaker_count(rttm_path: Path) -> int:
    speakers = set()
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return max(1, len(speakers))


def parse_rttm_segments(rttm_path: Path) -> list[tuple[float, float, str]]:
    segments = []
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            segments.append((start, start + dur, spk))
    return segments


def load_voxconverse_rows(project_root: Path, vox_root: Path) -> tuple[list[dict], list[dict]]:
    dev_audio = vox_root / "voxconverse-master" / "dev audio"
    dev_labels = vox_root / "voxconverse-master" / "dev labels"
    test_audio = vox_root / "voxconverse-master" / "test audio"
    test_labels = vox_root / "voxconverse-master" / "test labels"

    if not dev_audio.exists():
        raise RuntimeError(f"Missing VoxConverse dev audio dir: {dev_audio}")

    def build_rows(audio_dir: Path, label_dir: Path, split_name: str) -> list[dict]:
        rows: list[dict] = []
        for wav_path in sorted(audio_dir.glob("*.wav")):
            stem = wav_path.stem
            rttm = label_dir / f"{stem}.rttm"
            n_speakers = parse_rttm_speaker_count(rttm) if rttm.exists() else 2
            rttm_rel = rttm.relative_to(project_root).as_posix() if rttm.exists() else None
            if split_name == "dev" and rttm.exists():
                for idx, (start, end, spk) in enumerate(parse_rttm_segments(rttm)):
                    row: dict = {
                        "audio_path": wav_path.relative_to(project_root).as_posix(),
                        "segment_start": start,
                        "segment_end": end,
                        "speaker_id": stable_int(f"{stem}:{spk}"),
                        "speaker_label": spk,
                        "authenticity": 1,
                        "n_speakers": n_speakers,
                        "has_diarization": 1,
                        "dataset": "voxconverse",
                        "split": split_name,
                        "utt_id": f"{stem}_{idx:05d}",
                        "recording_id": stem,
                    }
                    if rttm_rel:
                        row["rttm_path"] = rttm_rel
                    rows.append(row)
            else:
                row = {
                    "audio_path": wav_path.relative_to(project_root).as_posix(),
                    "speaker_id": stable_int(stem),
                    "authenticity": 1,
                    "n_speakers": n_speakers,
                    "has_diarization": 1,
                    "dataset": "voxconverse",
                    "split": split_name,
                    "utt_id": stem,
                }
                if rttm_rel:
                    row["rttm_path"] = rttm_rel
                rows.append(row)
        return rows

    dev_rows = build_rows(dev_audio, dev_labels, "dev")
    test_rows = build_rows(test_audio, test_labels, "test") if test_audio.exists() else []
    return dev_rows, test_rows


def split_asvspoof(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    return shuffled[:cut], shuffled[cut:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifests from ASVspoof protocols + VoxConverse.")
    parser.add_argument("--project-root", default=".", help="Repo root path")
    parser.add_argument("--asvspoof-root", default="data/raw/asvspoof2021", help="ASVspoof root")
    parser.add_argument("--vox-root", default="data/raw/voxceleb2", help="VoxConverse root")
    parser.add_argument("--out-dir", default="data/manifests", help="Output directory for jsonl manifests")
    parser.add_argument(
        "--asvspoof-tracks",
        default="DF,LA",
        help="Comma-separated ASVspoof tracks for authenticity data. Default: DF,LA.",
    )
    parser.add_argument(
        "--asvspoof-train-phases",
        default="progress",
        help="Comma-separated ASVspoof phases used for train/val. Default: progress.",
    )
    parser.add_argument(
        "--asvspoof-test-phases",
        default="eval,hidden",
        help="Comma-separated ASVspoof phases used for locked test. Default: eval,hidden.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    asv_root = (project_root / args.asvspoof_root).resolve()
    vox_root = (project_root / args.vox_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    asv_tracks = parse_csv_arg(args.asvspoof_tracks)
    asv_train_phases = {p.lower() for p in parse_csv_arg(args.asvspoof_train_phases)}
    asv_test_phases = {p.lower() for p in parse_csv_arg(args.asvspoof_test_phases)}
    if not asv_tracks:
        raise ValueError("--asvspoof-tracks must contain at least one track")
    if not asv_train_phases:
        raise ValueError("--asvspoof-train-phases must contain at least one phase")
    if not asv_test_phases:
        raise ValueError("--asvspoof-test-phases must contain at least one phase")

    asv_train_val, asv_eval = load_asvspoof_rows(
        project_root=project_root,
        asv_root=asv_root,
        tracks=asv_tracks,
        train_phases=asv_train_phases,
        test_phases=asv_test_phases,
    )
    vox_dev_rows, vox_test_rows = load_voxconverse_rows(project_root=project_root, vox_root=vox_root)

    asv_train, asv_val = split_asvspoof(asv_train_val, seed=args.seed)

    # Split VoxConverse dev by recording, not by segment, so no recording straddles train/val.
    rng_vox = random.Random(args.seed + 1)
    dev_recording_ids = sorted({r["recording_id"] for r in vox_dev_rows if "recording_id" in r})
    rng_vox.shuffle(dev_recording_ids)
    cut = int(len(dev_recording_ids) * 0.8)
    train_rec_ids = set(dev_recording_ids[:cut])
    val_rec_ids = set(dev_recording_ids[cut:])
    vox_dev_train = [r for r in vox_dev_rows if r.get("recording_id") in train_rec_ids]
    vox_dev_val = [r for r in vox_dev_rows if r.get("recording_id") in val_rec_ids]

    train_rows = vox_dev_train + asv_train
    val_rows = vox_dev_val + asv_val        # VoxConverse val for DER + ASVspoof val for authenticity
    test_rows = vox_test_rows + asv_eval    # locked: VoxConverse test + ASVspoof official eval

    write_jsonl(out_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(out_dir / "val_manifest.jsonl", val_rows)
    write_jsonl(out_dir / "test_manifest.jsonl", test_rows)
    write_jsonl(out_dir / "test_asvspoof_manifest.jsonl", asv_eval)
    write_jsonl(out_dir / "train_asvspoof_manifest.jsonl", asv_train)
    write_jsonl(out_dir / "val_asvspoof_manifest.jsonl", asv_val)

    print(
        f"ASVspoof tracks={','.join(sorted(asv_tracks))} "
        f"train_phases={','.join(sorted(asv_train_phases))} "
        f"test_phases={','.join(sorted(asv_test_phases))}"
    )
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} test_asvspoof={len(asv_eval)}")
    print(f"train_asvspoof={len(asv_train)} val_asvspoof={len(asv_val)}")
    print(f"saved manifests to: {out_dir}")


if __name__ == "__main__":
    main()
