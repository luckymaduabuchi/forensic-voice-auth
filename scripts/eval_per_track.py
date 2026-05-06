from __future__ import annotations

"""Evaluate authentication performance broken down by ASVspoof track (DF / LA).

Writes per-track clean manifests on first run, then calls evaluate.py on each.

Usage:
    python scripts/eval_per_track.py \
        --model checkpoints/best_forensic_score.pth \
        --config configs/training_config.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _filter_manifest(src: Path, dst: Path, track: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("asvspoof_track", "").upper() == track.upper():
                f_out.write(line)
                count += 1
    return count


def _run_eval(model: str, config: str, manifest: str, benchmark: str, decision_threshold: float, max_auth: int) -> dict:
    import tempfile, yaml
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
            "--decision-threshold", str(decision_threshold),
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        result_path = ROOT / "checkpoints" / f"eval_{benchmark}.json"
        return json.loads(result_path.read_text())
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-track (DF/LA) authentication evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-auth-samples",
        type=int,
        default=1000000,
        help="Large default evaluates the full clean DF/LA track splits.",
    )
    parser.add_argument("--out-json", default="checkpoints/eval_per_track.json")
    args = parser.parse_args()

    import yaml
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    src_manifest = ROOT / cfg["data"]["test_asvspoof_path"]

    results = {}
    for track in ("DF", "LA"):
        dst = ROOT / "data/manifests/clean" / f"test_{track.lower()}_manifest_clean.jsonl"
        n = _filter_manifest(src_manifest, dst, track)
        print(f"  {track}: {n} rows -> {dst.name}")

        res = _run_eval(
            model=str(ROOT / args.model),
            config=str(ROOT / args.config),
            manifest=str(dst.relative_to(ROOT)),
            benchmark=f"track_{track.lower()}",
            decision_threshold=args.decision_threshold,
            max_auth=args.max_auth_samples,
        )
        auth = res.get("authenticity", {})
        results[track] = {
            "f1": auth.get("f1"),
            "precision": auth.get("precision"),
            "recall": auth.get("recall"),
            "accuracy": auth.get("accuracy"),
            "eer": auth.get("eer"),
            "min_dcf": auth.get("min_dcf"),
            "roc_auc": auth.get("roc_auc"),
            "pr_auc": auth.get("pr_auc"),
            "num_genuine": auth.get("num_genuine"),
            "num_spoof": auth.get("num_spoof"),
        }

        # Backward-compatible fallback for older eval JSONs.
        if results[track]["eer"] is None and auth.get("threshold_sweep_test_diagnostic") and auth.get("num_genuine"):
            from scripts.compute_eer import _eer_mindcf
            eer, min_dcf = _eer_mindcf(auth["threshold_sweep_test_diagnostic"], auth["num_genuine"], auth["num_spoof"])
            results[track]["eer"] = round(eer, 4)
            results[track]["min_dcf"] = round(min_dcf, 4)

    # Print table
    print(f"\n{'Track':<8} {'F1':>6} {'Prec':>7} {'Recall':>7} {'EER':>7} {'ROC-AUC':>8} {'Genuine':>8} {'Spoof':>8}")
    print("-" * 75)
    for track, r in results.items():
        print(
            f"{track:<8} {r.get('f1',0):.4f} {r.get('precision',0):.4f}  "
            f"{r.get('recall',0):.4f} {r.get('eer',float('nan')):.4f}  "
            f"{r.get('roc_auc',float('nan')):.4f} {r.get('num_genuine','?'):>8} {r.get('num_spoof','?'):>8}"
        )

    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
