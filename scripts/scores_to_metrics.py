from __future__ import annotations

"""Recompute all metrics from a raw scores CSV saved by evaluate.py.

The CSV has columns: utt_id, label, score, track

This script computes exact EER, ROC-AUC, PR-AUC, minDCF, per-track breakdowns,
and bootstrap 95% CI — all from the saved scores without re-running the model.

Usage:
    python scripts/scores_to_metrics.py \
        checkpoints/raw_scores_asvspoof_voxconverse.csv \
        --out-json checkpoints/scores_metrics.json \
        --n-bootstrap 2000
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]


def _has_two_classes(labels: np.ndarray) -> bool:
    return bool(np.any(labels == 0) and np.any(labels == 1))


def _interpolated_threshold(thresholds: np.ndarray, i: int, alpha: float | None = None) -> float:
    if alpha is not None and i + 1 < len(thresholds):
        left = thresholds[i]
        right = thresholds[i + 1]
        if np.isfinite(left) and np.isfinite(right):
            return float(left + alpha * (right - left))
        if np.isfinite(right):
            return float(right)
        if np.isfinite(left):
            return float(left)

    threshold = thresholds[i]
    if np.isfinite(threshold):
        return float(threshold)

    finite = thresholds[np.isfinite(thresholds)]
    return float(finite[0]) if finite.size else float("nan")


def _auc_from_curve(x: np.ndarray, y: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def _eer_from_scores(labels: np.ndarray, scores: np.ndarray, p_target: float = 0.05) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Linear interpolation at FNR=FPR crossing
    diff = fnr - fpr
    crossings = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if crossings.size:
        i = int(crossings[0])
        denom = diff[i] - diff[i + 1]
        alpha = diff[i] / denom if denom != 0 else 0.0
        eer = float(fnr[i] + alpha * (fnr[i + 1] - fnr[i]))
        eer_thr = _interpolated_threshold(thresholds, i, alpha)
    else:
        i = int(np.argmin(np.abs(diff)))
        eer = float((fnr[i] + fpr[i]) / 2.0)
        eer_thr = _interpolated_threshold(thresholds, i)

    norm = min(p_target, 1.0 - p_target)
    min_dcf = float(np.min((p_target * fnr + (1.0 - p_target) * fpr) / norm))

    return {
        "eer": round(eer, 6),
        "eer_threshold": round(eer_thr, 6),
        "min_dcf": round(min_dcf, 6),
        "roc_auc": round(_auc_from_curve(fpr, tpr), 6),
        "pr_auc": round(float(average_precision_score(labels, scores)), 6),
        "num_genuine": int((labels == 1).sum()),
        "num_spoof": int((labels == 0).sum()),
        "method": "score_exact",
    }


def _bootstrap_summary(vals: list[float], n_bootstrap: int) -> dict:
    arr = np.array(vals, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_95_lo": float("nan"),
            "ci_95_hi": float("nan"),
            "n_bootstrap": n_bootstrap,
        }
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "ci_95_lo": round(float(np.percentile(arr, 2.5)), 6),
        "ci_95_hi": round(float(np.percentile(arr, 97.5)), 6),
        "n_bootstrap": n_bootstrap,
    }


def _bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = {"eer": [], "roc_auc": [], "pr_auc": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        lb, sc = labels[idx], scores[idx]
        if not _has_two_classes(lb):
            continue
        try:
            exact = _eer_from_scores(lb, sc)
            vals["eer"].append(exact["eer"])
            vals["roc_auc"].append(exact["roc_auc"])
            vals["pr_auc"].append(float(average_precision_score(lb, sc)))
        except Exception:
            pass
    return {
        metric: _bootstrap_summary(metric_vals, n_bootstrap)
        for metric, metric_vals in vals.items()
    }


def compute_metrics_for_subset(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if not _has_two_classes(labels) or len(labels) < 10:
        return {"error": "insufficient samples or only one class"}
    base = _eer_from_scores(labels, scores)
    return {
        **base,
        "bootstrap": _bootstrap_metrics(labels, scores, n_bootstrap, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics from raw scores CSV.")
    parser.add_argument("scores_csv", help="CSV with columns: utt_id, label, score, track")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="checkpoints/scores_metrics.json")
    args = parser.parse_args()

    rows = []
    with open(args.scores_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    tracks = np.array([r["track"] for r in rows])

    print(f"Loaded {len(rows)} samples: {int((labels==1).sum())} genuine, {int((labels==0).sum())} spoof")

    results: dict = {}

    # Overall metrics
    print("\nOverall:")
    overall = compute_metrics_for_subset(labels, scores, args.n_bootstrap, args.seed)
    results["overall"] = overall
    print(f"  EER:     {overall['eer']:.4f}  95%CI [{overall['bootstrap']['eer']['ci_95_lo']:.4f}, {overall['bootstrap']['eer']['ci_95_hi']:.4f}]")
    print(f"  ROC-AUC: {overall['roc_auc']:.4f}  95%CI [{overall['bootstrap']['roc_auc']['ci_95_lo']:.4f}, {overall['bootstrap']['roc_auc']['ci_95_hi']:.4f}]")
    print(f"  PR-AUC:  {overall['pr_auc']:.4f}  95%CI [{overall['bootstrap']['pr_auc']['ci_95_lo']:.4f}, {overall['bootstrap']['pr_auc']['ci_95_hi']:.4f}]")
    print(f"  minDCF:  {overall['min_dcf']:.4f}")

    # Per-track breakdown
    unique_tracks = [t for t in sorted(set(tracks.tolist())) if t]
    if unique_tracks:
        results["per_track"] = {}
        print(f"\nPer-track breakdown:")
        for track in unique_tracks:
            mask = tracks == track
            if mask.sum() < 10:
                continue
            m = compute_metrics_for_subset(labels[mask], scores[mask], args.n_bootstrap // 2, args.seed)
            results["per_track"][track] = m
            print(f"  {track:<6}  EER={m['eer']:.4f}  AUC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  "
                  f"genuine={m['num_genuine']}  spoof={m['num_spoof']}")

    out = (ROOT / args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
