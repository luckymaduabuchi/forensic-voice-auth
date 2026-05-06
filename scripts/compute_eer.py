from __future__ import annotations

"""Compute EER and minDCF from ForensicVoice eval JSON files.

Usage:
    python scripts/compute_eer.py checkpoints/eval_joint_test.json \
                                  checkpoints/eval_baseline_spoof.json

New eval JSONs contain exact score-based EER/minDCF. Older JSONs only contain a
coarse test threshold sweep; for those files this script prints an approximate
diagnostic value and marks the method as ``coarse_sweep``.

Standard CM parameters here: p_target=0.05, C_miss=1, C_fa=1. This is a
normalized CM DCF, not ASVspoof tandem t-DCF.
"""

import argparse
import json
from pathlib import Path


def _eer_mindcf(
    sweep: list[dict],
    num_genuine: int,
    num_spoof: int,
    p_target: float = 0.05,
) -> tuple[float, float]:
    """Derive EER and normalised minDCF from a precision/recall threshold sweep."""
    fnrs, fprs = [], []
    for entry in sweep:
        recall = float(entry["recall"])
        precision = float(entry.get("precision") or 1e-12)
        fnr = 1.0 - recall
        tp = recall * num_genuine
        fp = tp * (1.0 - precision) / precision if precision > 1e-12 else float(num_spoof)
        fpr = min(fp / num_spoof, 1.0) if num_spoof > 0 else 0.0
        fnrs.append(fnr)
        fprs.append(fpr)

    # EER: linear interpolation at the FNR=FPR crossing
    eer = float("nan")
    for i in range(len(fnrs) - 1):
        d0 = fnrs[i] - fprs[i]
        d1 = fnrs[i + 1] - fprs[i + 1]
        if d0 * d1 <= 0:
            alpha = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
            eer = fnrs[i] + alpha * (fnrs[i + 1] - fnrs[i])
            break
    if eer != eer:  # no crossing found — take midpoint of closest pair
        closest = min(zip(fnrs, fprs), key=lambda p: abs(p[0] - p[1]))
        eer = sum(closest) / 2.0

    # minDCF (normalised by min prior cost)
    norm = min(p_target, 1.0 - p_target)
    min_dcf = min(
        (p_target * fnr + (1.0 - p_target) * fpr) / norm
        for fnr, fpr in zip(fnrs, fprs)
    )

    return eer, min_dcf


def compute_from_file(path: Path, p_target: float = 0.05) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    auth = data.get("authenticity", {})
    if auth.get("eer") is not None and auth.get("min_dcf") is not None:
        return {
            "model": data.get("model", str(path)),
            "f1": auth.get("f1"),
            "precision": auth.get("precision"),
            "recall": auth.get("recall"),
            "eer": round(float(auth["eer"]), 4),
            "min_dcf": round(float(auth["min_dcf"]), 4),
            "roc_auc": auth.get("roc_auc"),
            "pr_auc": auth.get("pr_auc"),
            "der": data.get("diarization", {}).get("der"),
            "method": "score_exact",
        }

    sweep = auth.get("threshold_sweep_test_diagnostic", auth.get("threshold_sweep", []))
    n_genuine = int(auth.get("num_genuine", 0))
    n_spoof = int(auth.get("num_spoof", 0))

    if not sweep or n_genuine == 0 or n_spoof == 0:
        return {"model": data.get("model", str(path)), "eer": float("nan"), "min_dcf": float("nan"), "method": "unavailable"}

    eer, min_dcf = _eer_mindcf(sweep, n_genuine, n_spoof, p_target=p_target)
    return {
        "model": data.get("model", str(path)),
        "f1": auth.get("f1"),
        "precision": auth.get("precision"),
        "recall": auth.get("recall"),
        "eer": round(eer, 4),
        "min_dcf": round(min_dcf, 4),
        "roc_auc": None,
        "pr_auc": None,
        "der": data.get("diarization", {}).get("der"),
        "method": "coarse_sweep",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EER and minDCF from eval JSON files.")
    parser.add_argument("jsons", nargs="+", help="Eval JSON file(s)")
    parser.add_argument("--p-target", type=float, default=0.05)
    args = parser.parse_args()

    rows = [compute_from_file(Path(p), p_target=args.p_target) for p in args.jsons]

    # Print table
    hdr = f"{'Model':<45} {'F1':>6} {'EER':>7} {'minDCF':>8} {'ROC-AUC':>8} {'DER':>7} {'Method':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        name = Path(r["model"]).stem if r["model"] else "?"
        f1 = f"{r['f1']:.4f}" if r["f1"] is not None else "  —   "
        eer = f"{r['eer']:.4f}" if r["eer"] == r["eer"] else "  —   "
        mdc = f"{r['min_dcf']:.4f}" if r["min_dcf"] == r["min_dcf"] else "  —   "
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "  —   "
        der = f"{r['der']:.4f}" if r.get("der") is not None and r["der"] == r["der"] else "  —   "
        print(f"{name:<45} {f1:>6} {eer:>7} {mdc:>8} {auc:>8} {der:>7} {r['method']:>12}")


if __name__ == "__main__":
    main()
