from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_eval(
    model_path: str,
    config_path: str,
    out_name: str,
    max_auth_samples: int,
    max_der_files: int,
    decision_threshold: float,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--model",
        model_path,
        "--config",
        config_path,
        "--benchmark",
        "asvspoof_voxconverse",
        "--max-auth-samples",
        str(max_auth_samples),
        "--max-der-files",
        str(max_der_files),
        "--decision-threshold",
        str(decision_threshold),
    ]
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    bench_slug = "asvspoof_voxconverse"
    src_json = ROOT / "checkpoints" / f"eval_{bench_slug}.json"
    src_csv = ROOT / "checkpoints" / f"eval_{bench_slug}.csv"
    dst_json = ROOT / "checkpoints" / f"{out_name}.json"
    dst_csv = ROOT / "checkpoints" / f"{out_name}.csv"
    dst_json.write_text(src_json.read_text(encoding="utf-8"), encoding="utf-8")
    dst_csv.write_text(src_csv.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved: {dst_json}")
    print(f"saved: {dst_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run robustness evaluation bundles. Each bundle requires a separate "
            "config pointing to bundle-specific test data (codec-corrupted, "
            "phone-quality, unseen-conditions). Pass --bundle-configs as a JSON "
            "mapping: '{\"robust_codec\": \"configs/eval_codec.yaml\", ...}'. "
            "Without --bundle-configs this script cannot produce valid robustness results."
        )
    )
    parser.add_argument("--model", default="checkpoints/best_forensic_score.pth")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument(
        "--bundle-configs",
        type=str,
        default=None,
        help='JSON mapping of bundle name to config path, e.g. \'{"robust_codec": "configs/eval_codec.yaml"}\'',
    )
    parser.add_argument("--max-auth-samples", type=int, default=100000)
    parser.add_argument("--max-der-files", type=int, default=200)
    parser.add_argument("--decision-threshold", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle_configs: dict[str, str] = {}
    if args.bundle_configs:
        bundle_configs = json.loads(args.bundle_configs)

    default_bundles = ["robust_unseen", "robust_codec", "robust_phone_quality"]
    bundles = list(bundle_configs.keys()) if bundle_configs else default_bundles

    if not bundle_configs and not args.dry_run:
        print(
            "ERROR: --bundle-configs not provided. Each robustness bundle requires a "
            "separate config pointing to bundle-specific test data. Without this, all "
            "bundles would evaluate on the same data and produce identical results, "
            "which is not valid robustness evidence. Pass --dry-run to preview commands."
        )
        raise SystemExit(1)

    summary = {"model": args.model, "runs": []}
    for name in bundles:
        cfg_path = bundle_configs.get(name, args.config)
        _run_eval(
            model_path=args.model,
            config_path=cfg_path,
            out_name=name,
            max_auth_samples=args.max_auth_samples,
            max_der_files=args.max_der_files,
            decision_threshold=args.decision_threshold,
            dry_run=args.dry_run,
        )
        summary["runs"].append(name)

    if not args.dry_run:
        out = ROOT / "checkpoints" / "robustness_manifest.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
