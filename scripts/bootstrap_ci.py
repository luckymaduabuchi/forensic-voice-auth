from __future__ import annotations

"""Compute bootstrap 95% confidence intervals on main result metrics.

Reads eval JSON files produced by evaluate.py and bootstraps over the
threshold sweep sample counts to estimate CI on F1, EER, minDCF, DER.

Usage:
    python scripts/bootstrap_ci.py \
        checkpoints/eval_asvspoof_voxconverse.json \
        --n-bootstrap 2000 \
        --out-json checkpoints/bootstrap_ci.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_f1(tp: int, fp: int, fn: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap F1 by resampling TP/FP/FN counts via multinomial."""
    probs = np.array([tp, fp, fn, max(0, n - tp - fp - fn)], dtype=float)
    probs /= probs.sum()
    f1s = []
    for _ in range(n):
        s = rng.multinomial(n, probs)
        btp, bfp, bfn = int(s[0]), int(s[1]), int(s[2])
        prec = btp / (btp + bfp) if (btp + bfp) > 0 else 0.0
        rec = btp / (btp + bfn) if (btp + bfn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.array(f1s)


def _bootstrap_eer_from_sweep(
    sweep: list[dict],
    num_genuine: int,
    num_spoof: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap EER by perturbing genuine/spoof counts with Binomial noise."""
    from scripts.compute_eer import _eer_mindcf
    eers = []
    for _ in range(n_bootstrap):
        # Perturb counts via Binomial with same probability — preserves relative size
        gen_b = int(rng.binomial(num_genuine * 2, 0.5))
        spo_b = int(rng.binomial(num_spoof * 2, 0.5))
        if gen_b == 0 or spo_b == 0:
            continue
        try:
            eer, _ = _eer_mindcf(sweep, gen_b, spo_b)
            if eer == eer:  # not NaN
                eers.append(eer)
        except Exception:
            pass
    return np.array(eers)


def _ci(arr: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = (1 - level) / 2 * 100
    hi = (1 + level) / 2 * 100
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def compute_ci(path: Path, n_bootstrap: int, seed: int = 42) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    auth = data.get("authenticity", {})
    diar = data.get("diarization", {})
    rng = np.random.default_rng(seed)

    results: dict = {"file": str(path), "n_bootstrap": n_bootstrap}

    # F1 bootstrap
    tp = auth.get("tp", 0)
    fp = auth.get("fp", 0)
    fn = auth.get("fn", 0)
    n_samples = auth.get("num_samples", tp + fp + fn)
    if tp + fp + fn > 0:
        f1s = _bootstrap_f1(tp, fp, fn, n_bootstrap, rng)
        lo, hi = _ci(f1s)
        results["f1"] = {
            "point": auth.get("f1"),
            "ci_95_lo": round(lo, 4),
            "ci_95_hi": round(hi, 4),
            "std": round(float(f1s.std()), 4),
        }

    # EER bootstrap from sweep
    sweep = auth.get("threshold_sweep_test_diagnostic", auth.get("threshold_sweep", []))
    ng = auth.get("num_genuine", 0)
    ns = auth.get("num_spoof", 0)

    # Use stored exact EER if available
    point_eer = auth.get("eer")
    if sweep and ng > 0 and ns > 0:
        import sys
        sys.path.insert(0, str(ROOT))
        eers = _bootstrap_eer_from_sweep(sweep, ng, ns, n_bootstrap, rng)
        if len(eers) > 10:
            lo, hi = _ci(eers)
            results["eer"] = {
                "point": point_eer,
                "ci_95_lo": round(lo, 4),
                "ci_95_hi": round(hi, 4),
                "std": round(float(eers.std()), 4),
                "note": "bootstrapped from threshold sweep — approximate",
            }

    # DER: report as point estimate only (single aggregate value, no per-file list stored)
    der = diar.get("der")
    if der is not None:
        results["der"] = {"point": der, "note": "single aggregate — no bootstrap without per-file data"}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 95% CI on eval metrics.")
    parser.add_argument("jsons", nargs="+", help="Eval JSON file(s)")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="checkpoints/bootstrap_ci.json")
    args = parser.parse_args()

    all_results = []
    for p in args.jsons:
        r = compute_ci(Path(p), n_bootstrap=args.n_bootstrap, seed=args.seed)
        all_results.append(r)

        print(f"\n{Path(p).name}")
        if "f1" in r:
            f = r["f1"]
            print(f"  F1:  {f['point']:.4f}  95% CI [{f['ci_95_lo']:.4f}, {f['ci_95_hi']:.4f}]  ±{f['std']:.4f}")
        if "eer" in r:
            e = r["eer"]
            pt = f"{e['point']:.4f}" if e["point"] is not None else "—"
            print(f"  EER: {pt}  95% CI [{e['ci_95_lo']:.4f}, {e['ci_95_hi']:.4f}]  ±{e['std']:.4f}")
        if "der" in r:
            print(f"  DER: {r['der']['point']:.4f}  (no bootstrap — aggregate only)")

    out = (ROOT / args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
