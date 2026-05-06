from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _default_experiments() -> list[dict[str, Any]]:
    return [
        {
            "name": "full_joint_resnet34",
            "overrides": {
                "model.backbone_name": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "training.loss_weights.contrastive": 1.0,
                "training.loss_weights.authenticity": 1.0,
                "training.loss_weights.diarization": 0.3,
            },
        },
        {
            "name": "no_contrastive",
            "overrides": {
                "training.loss_weights.contrastive": 0.0,
                "training.loss_weights.authenticity": 1.0,
                "training.loss_weights.diarization": 0.3,
            },
        },
        {
            "name": "no_diarization_aux",
            "overrides": {
                "training.loss_weights.contrastive": 1.0,
                "training.loss_weights.authenticity": 1.0,
                "training.loss_weights.diarization": 0.0,
            },
        },
        {
            "name": "contrastive_only",
            "overrides": {
                "training.loss_weights.contrastive": 1.0,
                "training.loss_weights.authenticity": 0.0,
                "training.loss_weights.diarization": 0.0,
            },
        },
        {
            "name": "backbone_pyannote_embedding",
            "overrides": {
                "model.backbone_name": "pyannote/embedding",
                "training.loss_weights.contrastive": 1.0,
                "training.loss_weights.authenticity": 1.0,
                "training.loss_weights.diarization": 0.3,
            },
        },
        {
            "name": "auth_only",
            "overrides": {
                "training.loss_weights.contrastive": 0.0,
                "training.loss_weights.authenticity": 1.0,
                "training.loss_weights.diarization": 0.0,
            },
        },
        # --- Frozen vs unfrozen backbone ---
        {
            "name": "frozen_backbone",
            "overrides": {
                "model.freeze_backbone": True,
            },
        },
        # --- Loss weight sensitivity (one weight varied, others at default) ---
        # defaults: contrastive=1.0, authenticity=1.0, diarization=0.3
        {
            "name": "lw_contrastive_0.5",
            "overrides": {"training.loss_weights.contrastive": 0.5},
        },
        {
            "name": "lw_contrastive_2.0",
            "overrides": {"training.loss_weights.contrastive": 2.0},
        },
        {
            "name": "lw_auth_0.5",
            "overrides": {"training.loss_weights.authenticity": 0.5},
        },
        {
            "name": "lw_auth_2.0",
            "overrides": {"training.loss_weights.authenticity": 2.0},
        },
        {
            "name": "lw_diar_0.1",
            "overrides": {"training.loss_weights.diarization": 0.1},
        },
        {
            "name": "lw_diar_1.0",
            "overrides": {"training.loss_weights.diarization": 1.0},
        },
        # --- Window size ablations ---
        {
            "name": "window_1s",
            "overrides": {"data.segment_seconds": 1.0},
        },
        {
            "name": "window_3s",
            "overrides": {"data.segment_seconds": 3.0},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study train/eval matrix.")
    parser.add_argument("--base-config", default="configs/training_config.yaml")
    parser.add_argument("--out-csv", default="checkpoints/ablation_results.csv")
    parser.add_argument("--max-auth-samples", type=int, default=20000)
    parser.add_argument("--max-der-files", type=int, default=50)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per ablation run (default 20 for speed)")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment names to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config_path = (ROOT / args.base_config).resolve()
    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    experiments = _default_experiments()
    if args.experiments:
        aliases = {"backbone_wespeaker_ecapa": "backbone_pyannote_embedding"}
        wanted = {aliases.get(name, name) for name in args.experiments}
        known = {exp["name"] for exp in experiments}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown experiments: {unknown}. Known: {sorted(known)}")
        experiments = [exp for exp in experiments if exp["name"] in wanted]
    rows: list[dict[str, Any]] = []

    for exp in experiments:
        name = exp["name"]
        cfg = deepcopy(base_cfg)
        cfg["training"]["num_epochs"] = args.epochs
        checkpoint_dir = ROOT / "checkpoints" / f"ablation_{name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _set_nested(cfg, "training.checkpoint_dir", str(checkpoint_dir.relative_to(ROOT)))

        for k, v in exp["overrides"].items():
            _set_nested(cfg, k, v)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(cfg, tf, sort_keys=False)
            cfg_path = Path(tf.name)

        try:
            train_cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path)]
            _run(train_cmd, dry_run=args.dry_run)

            best_ckpt = checkpoint_dir / "best_forensic_score.pth"
            if not best_ckpt.exists():
                best_ckpt = checkpoint_dir / "best_f1.pth"

            eval_cmd = [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                str(cfg_path),
                "--model",
                str(best_ckpt),
                "--benchmark",
                "asvspoof_voxconverse",
                "--max-auth-samples",
                str(args.max_auth_samples),
                "--max-der-files",
                str(args.max_der_files),
                "--decision-threshold",
                str(args.decision_threshold),
            ]
            _run(eval_cmd, dry_run=args.dry_run)

            if args.dry_run:
                rows.append({"experiment": name, "status": "planned"})
                continue

            out_json = ROOT / "checkpoints" / "eval_asvspoof_voxconverse.json"
            res = _load_json(out_json)
            exp_json = checkpoint_dir / "eval_result.json"
            exp_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

            auth = res.get("authenticity", {})
            diar = res.get("diarization", {})

            eer = auth.get("eer")
            min_dcf = auth.get("min_dcf")
            if eer is None and auth.get("threshold_sweep_test_diagnostic") and auth.get("num_genuine"):
                sys.path.insert(0, str(ROOT))
                from scripts.compute_eer import _eer_mindcf
                eer, min_dcf = _eer_mindcf(auth["threshold_sweep_test_diagnostic"], auth["num_genuine"], auth["num_spoof"])

            rows.append(
                {
                    "experiment": name,
                    "f1": auth.get("f1"),
                    "precision": auth.get("precision"),
                    "recall": auth.get("recall"),
                    "eer": round(float(eer), 4) if eer is not None and eer == eer else None,
                    "min_dcf": round(float(min_dcf), 4) if min_dcf is not None and min_dcf == min_dcf else None,
                    "roc_auc": auth.get("roc_auc"),
                    "pr_auc": auth.get("pr_auc"),
                    "der": diar.get("der"),
                    "forensic_score": (
                        0.5 * (auth.get("f1") or 0) + 0.5 * (1.0 - (diar.get("der") or 1.0))
                        if diar.get("der") is not None else None
                    ),
                }
            )
        finally:
            cfg_path.unlink(missing_ok=True)

    out_csv = (ROOT / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "f1",
        "precision",
        "recall",
        "eer",
        "min_dcf",
        "roc_auc",
        "pr_auc",
        "der",
        "forensic_score",
        "status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"saved ablation summary: {out_csv}")


if __name__ == "__main__":
    main()
