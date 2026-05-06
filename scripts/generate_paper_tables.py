from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_row(name: str, res: dict[str, Any]) -> dict[str, Any]:
    auth = res.get("authenticity", {})
    diar = res.get("diarization", {})
    return {
        "method": name,
        "benchmark": res.get("benchmark", "asvspoof_voxconverse"),
        "accuracy": auth.get("accuracy"),
        "precision": auth.get("precision"),
        "recall": auth.get("recall"),
        "f1": auth.get("f1"),
        "der": diar.get("der"),
        "num_samples": auth.get("num_samples"),
        "num_der_files": diar.get("num_files"),
        "decision_threshold": auth.get("decision_threshold"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NeurIPS-ready tables from checkpoint eval artifacts.")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="paper/tables")
    args = parser.parse_args()

    ckpt_dir = (ROOT / args.checkpoints_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ckpt_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for p in files:
        if p.name.startswith("eval_") or p.name.startswith("baseline_") or p.name.startswith("robust_"):
            try:
                res = _load_json(p)
            except Exception:
                continue
            rows.append(_extract_row(p.stem, res))

    main_csv = out_dir / "main_results.csv"
    fieldnames = [
        "method",
        "benchmark",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "der",
        "num_samples",
        "num_der_files",
        "decision_threshold",
    ]
    with main_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "main_results.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Method | Benchmark | Acc | Prec | Rec | F1 | DER |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['method']} | {r['benchmark']} | "
                f"{_fmt(r['accuracy'])} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | "
                f"{_fmt(r['f1'])} | {_fmt(r['der'])} |\n"
            )

    print(f"saved: {main_csv}")
    print(f"saved: {md_path}")


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "NA"


if __name__ == "__main__":
    main()
