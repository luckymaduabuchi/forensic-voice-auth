from __future__ import annotations

"""Run joint training with multiple seeds and aggregate results for stability reporting.

Usage:
    python scripts/train_multiseed.py \
        --config configs/training_config.yaml \
        --seeds 42 123 456 \
        --gpu 0
"""

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training for stability analysis.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override per seed.")
    parser.add_argument("--max-auth-samples", type=int, default=100000)
    parser.add_argument("--max-der-files", type=int, default=200)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--out-json", default="checkpoints/multiseed_results.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import os, tempfile
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}

    with open(ROOT / args.config) as f:
        base_cfg = yaml.safe_load(f)

    all_results = []

    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        cfg = deepcopy(base_cfg)
        cfg["training"]["seed"] = seed
        if args.epochs is not None:
            cfg["training"]["num_epochs"] = args.epochs
        ckpt_dir = f"checkpoints/seed_{seed}"
        cfg["training"]["checkpoint_dir"] = ckpt_dir

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(cfg, tf)
            tmp_cfg = tf.name

        try:
            # Train
            train_cmd = [sys.executable, "scripts/train.py", "--config", tmp_cfg]
            if not args.dry_run:
                subprocess.run(train_cmd, check=True, cwd=str(ROOT), env=env)

            # Evaluate best checkpoint
            best_ckpt = ROOT / ckpt_dir / "best_forensic_score.pth"
            if not best_ckpt.exists():
                best_ckpt = ROOT / ckpt_dir / "best_f1.pth"

            eval_cmd = [
                sys.executable, "scripts/evaluate.py",
                "--model", str(best_ckpt),
                "--config", tmp_cfg,
                "--benchmark", "asvspoof_voxconverse",
                "--max-auth-samples", str(args.max_auth_samples),
                "--max-der-files", str(args.max_der_files),
                "--decision-threshold", str(args.decision_threshold),
            ]
            if not args.dry_run:
                subprocess.run(eval_cmd, check=True, cwd=str(ROOT), env=env)
                res_path = ROOT / "checkpoints" / "eval_asvspoof_voxconverse.json"
                res = json.loads(res_path.read_text())
                auth = res["authenticity"]
                diar = res.get("diarization", {})

                eer = auth.get("eer")
                min_dcf = auth.get("min_dcf")
                if eer is None and auth.get("threshold_sweep_test_diagnostic"):
                    sys.path.insert(0, str(ROOT))
                    from scripts.compute_eer import _eer_mindcf
                    eer, min_dcf = _eer_mindcf(
                        auth["threshold_sweep_test_diagnostic"],
                        auth["num_genuine"],
                        auth["num_spoof"],
                    )

                row = {
                    "seed": seed,
                    "f1": auth.get("f1"),
                    "precision": auth.get("precision"),
                    "recall": auth.get("recall"),
                    "der": diar.get("der"),
                    "forensic_score": 0.5 * auth.get("f1", 0) + 0.5 * (1.0 - diar.get("der", 1.0)),
                    "eer": round(float(eer), 4) if eer is not None and eer == eer else None,
                    "min_dcf": round(float(min_dcf), 4) if min_dcf is not None and min_dcf == min_dcf else None,
                    "roc_auc": auth.get("roc_auc"),
                    "pr_auc": auth.get("pr_auc"),
                }
                all_results.append(row)
                print(f"seed={seed}  f1={row['f1']:.4f}  der={row['der']:.4f}  "
                      f"forensic={row['forensic_score']:.4f}  eer={row['eer']}")
            else:
                print(f"[dry-run] would train seed={seed}, ckpt={ckpt_dir}")

        finally:
            Path(tmp_cfg).unlink(missing_ok=True)

    if args.dry_run or not all_results:
        print("\nDry run complete — no results to aggregate.")
        return

    # Aggregate
    metrics = ["f1", "precision", "recall", "der", "forensic_score", "eer", "min_dcf", "roc_auc", "pr_auc"]
    summary = {"seeds": args.seeds, "runs": all_results, "aggregate": {}}
    for m in metrics:
        vals = [r[m] for r in all_results if r.get(m) is not None]
        if vals:
            summary["aggregate"][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*50}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*50}")
    print(f"{'Metric':<16} {'Mean':>8} {'±Std':>8}")
    print("-" * 34)
    for m, v in summary["aggregate"].items():
        print(f"{m:<16} {v['mean']:>8.4f} {v['std']:>8.4f}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
