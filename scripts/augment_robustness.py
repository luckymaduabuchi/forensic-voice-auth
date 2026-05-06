from __future__ import annotations

"""Create robustness test sets by applying codec compression and additive noise to audio.

Produces new manifests pointing to augmented audio files, then runs evaluate.py on each.
Requires: sox (for noise), ffmpeg (for codec compression).

Usage:
    python scripts/augment_robustness.py \
        --model checkpoints/best_forensic_score.pth \
        --config configs/training_config.yaml \
        --max-samples 5000

Output JSON: checkpoints/robustness_augmented.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _add_noise(waveform: torch.Tensor, snr_db: float, rng: np.random.Generator) -> torch.Tensor:
    signal_power = (waveform ** 2).mean()
    if signal_power < 1e-10:
        return waveform
    noise = torch.from_numpy(rng.standard_normal(waveform.shape).astype(np.float32))
    noise_power = (noise ** 2).mean()
    scale = (signal_power / (noise_power * 10 ** (snr_db / 10))) ** 0.5
    return (waveform + scale * noise).clamp(-1.0, 1.0)


def _apply_codec(src: Path, dst: Path, codec: str, bitrate: str) -> bool:
    if not _check_tool("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:a", codec, "-b:a", bitrate,
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return False
    # Decode back to wav for evaluation
    wav_dst = dst.with_suffix(".wav")
    cmd2 = ["ffmpeg", "-y", "-i", str(dst), "-ar", "16000", "-ac", "1", str(wav_dst)]
    result2 = subprocess.run(cmd2, capture_output=True)
    return result2.returncode == 0


def _build_augmented_manifest(
    src_manifest: Path,
    aug_audio_dir: Path,
    condition: str,
    max_samples: int,
    snr_db: float | None,
    codec: str | None,
    bitrate: str | None,
    sr: int,
) -> Path:
    aug_audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    items = []
    with src_manifest.open() as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if max_samples and len(items) > max_samples:
        idxs = rng.choice(len(items), size=max_samples, replace=False)
        items = [items[i] for i in idxs]

    manifest_path = aug_audio_dir / f"manifest_{condition}.jsonl"
    written = 0
    with manifest_path.open("w") as out:
        for item in items:
            audio_rel = item.get("audio_path", "")
            src_path = ROOT / audio_rel
            if not src_path.exists():
                continue
            stem = Path(audio_rel).stem
            aug_path = aug_audio_dir / f"{stem}_{condition}.wav"

            if snr_db is not None:
                # Noise augmentation — done in-memory
                try:
                    wav, orig_sr = torchaudio.load(src_path)
                    if orig_sr != sr:
                        wav = torchaudio.functional.resample(wav, orig_sr, sr)
                    wav = wav.mean(0) if wav.dim() == 2 else wav
                    wav = _add_noise(wav, snr_db=snr_db, rng=rng)
                    torchaudio.save(str(aug_path), wav.unsqueeze(0), sr)
                except Exception:
                    continue
            elif codec is not None:
                tmp_enc = aug_path.with_suffix(".mp3" if codec == "libmp3lame" else ".ogg")
                if not _apply_codec(src_path, tmp_enc, codec, bitrate or "32k"):
                    continue
                wav_back = tmp_enc.with_suffix(".wav")
                if not wav_back.exists():
                    continue
                aug_path = wav_back
                try:
                    tmp_enc.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                continue

            row = {**item, "audio_path": str(aug_path.relative_to(ROOT))}
            out.write(json.dumps(row) + "\n")
            written += 1

    print(f"  [{condition}] wrote {written} entries → {manifest_path.name}")
    return manifest_path


def _run_eval(model: str, config: str, manifest: str, benchmark: str, max_auth: int, threshold: float) -> dict:
    import copy
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["test_asvspoof_path"] = manifest
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(cfg, tf)
        tmp_cfg = tf.name
    try:
        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--model", model,
            "--config", tmp_cfg,
            "--benchmark", benchmark,
            "--max-auth-samples", str(max_auth),
            "--max-der-files", "0",
            "--decision-threshold", str(threshold),
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        result_path = ROOT / "checkpoints" / f"eval_{benchmark}.json"
        return json.loads(result_path.read_text())
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation via codec and noise augmentation.")
    parser.add_argument("--model", default="checkpoints/best_forensic_score.pth")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--out-json", default="checkpoints/robustness_augmented.json")
    parser.add_argument("--aug-dir", default="data/augmented")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    src_manifest = ROOT / cfg["data"]["test_asvspoof_path"]
    aug_dir = ROOT / args.aug_dir
    sr = cfg["data"].get("sample_rate", 16000)

    has_ffmpeg = _check_tool("ffmpeg")
    if not has_ffmpeg:
        print("WARNING: ffmpeg not found — codec conditions will be skipped")

    conditions = [
        # (name, snr_db, codec, bitrate, benchmark_slug)
        ("noise_10db",  10.0,  None,           None,    "robust_noise_10db"),
        ("noise_20db",  20.0,  None,           None,    "robust_noise_20db"),
        ("codec_mp3_32k", None, "libmp3lame",  "32k",   "robust_mp3_32k"),
        ("codec_opus_16k", None, "libopus",    "16k",   "robust_opus_16k"),
    ]

    results = {"model": args.model, "conditions": {}}
    model_path = str((ROOT / args.model).resolve())

    for name, snr_db, codec, bitrate, benchmark in conditions:
        if codec is not None and not has_ffmpeg:
            print(f"  [{name}] skipped (ffmpeg not available)")
            continue
        print(f"\nBuilding augmented set: {name}")
        manifest = _build_augmented_manifest(
            src_manifest=src_manifest,
            aug_audio_dir=aug_dir / name,
            condition=name,
            max_samples=args.max_samples,
            snr_db=snr_db,
            codec=codec,
            bitrate=bitrate,
            sr=sr,
        )
        if manifest.stat().st_size == 0:
            print(f"  [{name}] empty manifest — skipping eval")
            continue
        print(f"  [{name}] running evaluation...")
        res = _run_eval(
            model=model_path,
            config=str(ROOT / args.config),
            manifest=str(manifest),
            benchmark=benchmark,
            max_auth=args.max_samples,
            threshold=args.decision_threshold,
        )
        auth = res.get("authenticity", {})
        results["conditions"][name] = {
            "f1": auth.get("f1"),
            "precision": auth.get("precision"),
            "recall": auth.get("recall"),
            "eer": auth.get("eer"),
            "min_dcf": auth.get("min_dcf"),
            "roc_auc": auth.get("roc_auc"),
            "num_samples": auth.get("num_samples"),
        }
        print(f"  [{name}] F1={auth.get('f1', 'nan'):.4f}  EER={auth.get('eer', 'nan')}")

    out = (ROOT / args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")

    # Print summary table
    print(f"\n{'Condition':<20} {'F1':>6} {'EER':>7} {'minDCF':>8} {'ROC-AUC':>8}")
    print("-" * 55)
    for cond, r in results["conditions"].items():
        f1 = f"{r['f1']:.4f}" if r.get("f1") is not None else "  —   "
        eer = f"{r['eer']:.4f}" if r.get("eer") is not None else "  —   "
        mdc = f"{r['min_dcf']:.4f}" if r.get("min_dcf") is not None else "  —   "
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "  —   "
        print(f"{cond:<20} {f1:>6} {eer:>7} {mdc:>8} {auc:>8}")


if __name__ == "__main__":
    main()
