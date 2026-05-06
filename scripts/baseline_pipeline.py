from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import torch
import yaml
from pyannote.metrics.diarization import DiarizationErrorRate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_authenticity, load_model, parse_rttm

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:  # pragma: no cover
    PyannotePipeline = None


warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=r".*torchaudio\._backend\.set_audio_backend has been deprecated.*")
warnings.filterwarnings("ignore", message=r".*'uem' was approximated by the union of 'reference' and 'hypothesis' extents.*")


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_der_with_pyannote_pipeline(
    test_manifest_path: Path,
    project_root: Path,
    vox_root: str,
    diarization_model: str,
    max_files: int | None = None,
    hf_token: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if max_files is not None and max_files <= 0:
        return {"der": float("nan"), "num_files": 0, "note": "DER eval skipped (max_files<=0)"}

    if PyannotePipeline is None:
        return {"der": float("nan"), "num_files": 0, "error": "pyannote.audio Pipeline unavailable"}

    try:
        print(f"loading diarization pipeline: {diarization_model}", flush=True)
        pipeline = PyannotePipeline.from_pretrained(diarization_model, use_auth_token=hf_token)
    except Exception as exc:
        return {"der": float("nan"), "num_files": 0, "error": f"failed to load diarization pipeline: {exc}"}

    if device is not None and device.type == "cuda" and hasattr(pipeline, "to"):
        try:
            pipeline.to(device)
            print(f"diarization pipeline device: {device}", flush=True)
        except Exception as exc:
            print(f"warning: failed to move diarization pipeline to {device}: {exc}", flush=True)

    if not test_manifest_path.exists():
        return {"der": float("nan"), "num_files": 0, "error": f"missing manifest: {test_manifest_path}"}

    items = []
    with test_manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # Skip non-diarization rows (ASVspoof) up front — avoids iterating 640K rows.
    items = [item for item in items if int(item.get("has_diarization", 0)) == 1]
    target_files = len(items) if max_files is None else min(max_files, len(items))
    print(f"pyannote DER: evaluating up to {target_files} files from {len(items)} candidates", flush=True)

    der_metric = DiarizationErrorRate()
    count = 0
    started = time.time()
    for item in items:
        audio_rel = item.get("audio_path")
        if not audio_rel:
            continue
        audio_path = project_root / audio_rel
        utt_id = item.get("utt_id", Path(audio_rel).stem)
        rttm_path = project_root / vox_root / "voxconverse-master" / "test labels" / f"{utt_id}.rttm"
        if not audio_path.exists() or not rttm_path.exists():
            continue

        ref = parse_rttm(rttm_path)
        try:
            hyp = pipeline(str(audio_path))
        except Exception as exc:
            print(f"warning: pyannote pipeline failed on {audio_path.name}: {exc}")
            continue
        if len(hyp.labels()) == 0:
            continue
        der_metric(ref, hyp)
        count += 1
        if count == 1 or count % 10 == 0 or (max_files is not None and count >= max_files):
            elapsed_min = (time.time() - started) / 60.0
            print(f"pyannote DER progress: {count}/{target_files} files ({elapsed_min:.1f} min)", flush=True)
        if max_files is not None and count >= max_files:
            break

    if count == 0:
        return {"der": float("nan"), "num_files": 0, "error": "no evaluable files"}
    return {"der": float(abs(der_metric)), "num_files": count}


def save_outputs(results: dict[str, Any], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    flat = []
    for section, values in results.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if k in {"threshold_sweep", "threshold_sweep_test_diagnostic"}:
                    continue
                flat.append({"section": section, "metric": k, "value": v})
    auth_results = results.get("authenticity", {})
    sweep = auth_results.get(
        "threshold_sweep_test_diagnostic",
        auth_results.get("threshold_sweep", []),
    )
    for row in sweep:
        flat.append(
            {
                "section": "authenticity_threshold_sweep_test_diagnostic",
                "metric": f"thr_{row['threshold']:.2f}",
                "value": json.dumps(
                    {
                        "accuracy": row["accuracy"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                    }
                ),
            }
        )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline A: diarization pipeline + separate anti-spoof detector.")
    parser.add_argument("--model", required=True, help="Checkpoint for anti-spoof model.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--max-der-files", type=int, default=200)
    parser.add_argument("--max-auth-samples", type=int, default=100000)
    parser.add_argument("--decision-threshold", type=float, default=0.8)
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--hf-token", default=None, help="Optional HF token for gated models.")
    parser.add_argument("--out-json", default="checkpoints/baseline_pipeline.json")
    parser.add_argument("--out-csv", default="checkpoints/baseline_pipeline.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # Separate anti-spoof model evaluation.
    print("starting anti-spoof evaluation", flush=True)
    model = load_model(args.model, cfg["model"], device=device)
    auth_metrics = evaluate_authenticity(
        model=model,
        manifest_path=cfg["data"].get("test_asvspoof_path", cfg["data"].get("test_path", cfg["data"]["val_path"])),
        project_root=project_root,
        batch_size=cfg["training"].get("batch_size", 8),
        device=device,
        max_samples=args.max_auth_samples,
        decision_threshold=args.decision_threshold,
        segment_seconds=cfg["data"].get("segment_seconds", 2.0),
        raw_scores_path=project_root / "checkpoints" / "raw_scores_baseline_pipeline.csv",
    )
    print(f"finished anti-spoof evaluation: {auth_metrics.get('num_samples', 0)} samples", flush=True)

    # Separate diarization model evaluation.
    print("starting pyannote diarization evaluation", flush=True)
    der_metrics = evaluate_der_with_pyannote_pipeline(
        test_manifest_path=project_root / "data/manifests/test_manifest.jsonl",
        project_root=project_root,
        vox_root="data/raw/voxceleb2",
        diarization_model=args.diarization_model,
        max_files=args.max_der_files,
        hf_token=args.hf_token,
        device=device,
    )
    print(f"finished pyannote diarization evaluation: {der_metrics}", flush=True)

    results = {
        "baseline": "pipeline_diarization_plus_separate_spoof",
        "model": args.model,
        "diarization_model": args.diarization_model,
        "authenticity": auth_metrics,
        "diarization": der_metrics,
    }
    out_json = (ROOT / args.out_json).resolve()
    out_csv = (ROOT / args.out_csv).resolve()
    save_outputs(results, out_json=out_json, out_csv=out_csv)

    print(json.dumps(results, indent=2))
    print(f"saved json: {out_json}")
    print(f"saved csv: {out_csv}")


if __name__ == "__main__":
    main()
