from __future__ import annotations

"""Find the optimal decision threshold on the validation set, then re-evaluate
on the test set with that threshold and compare against the fixed 0.5 baseline.

This is the correct way to report threshold-dependent metrics (F1, precision,
recall, accuracy): select the threshold on val, lock it, report on test.

Usage:
    python scripts/calibrate_threshold.py \
        --model checkpoints/best_forensic_score.pth \
        --config configs/training_config.yaml \
        --out-json checkpoints/threshold_calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ForensicAudioDataset
from src.evaluation.metrics import compute_authenticity_metrics
from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig


def _collect_scores(
    model: ForensicVoice,
    manifest: str,
    device: torch.device,
    batch_size: int,
    segment_seconds: float,
    project_root: Path,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ForensicAudioDataset(
        manifest_path=manifest,
        split="test",
        sample_rate=16000,
        segment_seconds=segment_seconds,
        project_root=str(project_root),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["audio"].to(device), n_speakers=None)
            probs = torch.sigmoid(out["authenticity_logits"]).cpu().view(-1)
            labels = batch["authenticity"].cpu().view(-1)
            all_probs.append(probs)
            all_labels.append(labels)
            seen += labels.numel()
            if max_samples and seen >= max_samples:
                break
    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_labels).numpy().astype(int),
    )


def _find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Sweep thresholds, return the one with highest F1 on this set."""
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.linspace(0.05, 0.95, 181):  # 0.005 step — fine-grained
        m = compute_authenticity_metrics(
            torch.from_numpy(probs), torch.from_numpy(labels), threshold=float(t)
        )
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}
    return best


def _eval_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    m = compute_authenticity_metrics(
        torch.from_numpy(probs), torch.from_numpy(labels), threshold=threshold
    )
    return {**m, "threshold": threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-calibrated threshold selection.")
    parser.add_argument("--model", default="checkpoints/best_forensic_score.pth")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--max-val-samples", type=int, default=50000)
    parser.add_argument("--max-test-samples", type=int, default=100000)
    parser.add_argument("--out-json", default="checkpoints/threshold_calibration.json")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = cfg["training"].get("batch_size", 64)
    seg = cfg["data"].get("segment_seconds", 2.0)

    model = ForensicVoice(ForensicVoiceConfig(**cfg["model"])).to(device)
    ckpt = torch.load(ROOT / args.model, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    val_manifest = str(ROOT / cfg["data"]["val_asvspoof_path"])
    test_manifest = str(ROOT / cfg["data"]["test_asvspoof_path"])

    print("Collecting val scores...")
    val_probs, val_labels = _collect_scores(
        model, val_manifest, device, bs, seg, ROOT, args.max_val_samples
    )
    print(f"  val: {len(val_labels)} samples  "
          f"({int((val_labels==1).sum())} genuine, {int((val_labels==0).sum())} spoof)")

    print("Finding optimal threshold on val set...")
    best_val = _find_best_threshold(val_probs, val_labels)
    calibrated_threshold = best_val["threshold"]
    print(f"  val-optimal threshold: {calibrated_threshold:.3f}  "
          f"(val F1={best_val['f1']:.4f})")

    print("\nCollecting test scores...")
    test_probs, test_labels = _collect_scores(
        model, test_manifest, device, bs, seg, ROOT, args.max_test_samples
    )
    print(f"  test: {len(test_labels)} samples  "
          f"({int((test_labels==1).sum())} genuine, {int((test_labels==0).sum())} spoof)")

    # Evaluate at threshold=0.5 and at val-calibrated threshold
    test_at_0_5 = _eval_at_threshold(test_probs, test_labels, 0.5)
    test_at_cal = _eval_at_threshold(test_probs, test_labels, calibrated_threshold)

    print(f"\n{'Threshold':<22} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Acc':>7}")
    print("-" * 52)
    for label, result in [("fixed 0.5", test_at_0_5), (f"val-cal ({calibrated_threshold:.3f})", test_at_cal)]:
        print(f"{label:<22} {result['f1']:>7.4f} {result['precision']:>7.4f} "
              f"{result['recall']:>7.4f} {result['accuracy']:>7.4f}")

    results = {
        "model": args.model,
        "val_manifest": val_manifest,
        "test_manifest": test_manifest,
        "val_calibration": {
            "n_samples": len(val_labels),
            "optimal_threshold": calibrated_threshold,
            "val_f1_at_optimal": round(best_val["f1"], 4),
            "val_precision_at_optimal": round(best_val["precision"], 4),
            "val_recall_at_optimal": round(best_val["recall"], 4),
        },
        "test_at_threshold_0.5": {k: round(float(v), 4) if isinstance(v, float) else v
                                   for k, v in test_at_0_5.items()},
        "test_at_val_calibrated": {k: round(float(v), 4) if isinstance(v, float) else v
                                    for k, v in test_at_cal.items()},
        "delta_f1": round(test_at_cal["f1"] - test_at_0_5["f1"], 4),
    }

    out = (ROOT / args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDelta F1 (calibrated - fixed): {results['delta_f1']:+.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
