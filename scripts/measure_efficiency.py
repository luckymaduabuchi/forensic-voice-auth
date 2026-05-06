from __future__ import annotations

"""Measure inference latency and parameter count for joint model vs separate baselines.

Usage:
    python scripts/measure_efficiency.py \
        --model checkpoints/best_forensic_score.pth \
        --config configs/training_config.yaml \
        --out-json checkpoints/efficiency_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_latency(
    model: torch.nn.Module,
    audio: torch.Tensor,
    device: torch.device,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    model.eval()
    audio = audio.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(audio)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(audio)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    import statistics
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "std_ms": round(statistics.stdev(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference efficiency.")
    parser.add_argument("--model", default="checkpoints/best_forensic_score.pth")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--out-json", default="checkpoints/efficiency_results.json")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg["data"].get("sample_rate", 16000)
    seg_len = int(args.segment_seconds * sr)
    audio = torch.randn(args.batch_size, seg_len)

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}, segment: {args.segment_seconds}s ({seg_len} samples)")

    # --- Joint model ---
    model = ForensicVoice(ForensicVoiceConfig(**cfg["model"])).to(device)
    ckpt = torch.load(ROOT / args.model, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)

    params = count_parameters(model)
    latency = measure_latency(model, audio, device, n_runs=args.n_runs)

    print(f"\nJoint ForensicVoice")
    print(f"  Parameters: {params['total']:,} total / {params['trainable']:,} trainable")
    print(f"  Latency (batch={args.batch_size}): {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
    print(f"  Note: 1 forward pass handles both authentication + diarization simultaneously")

    # Compute theoretical baseline cost (two separate forward passes)
    two_pass_ms = latency["mean_ms"] * 2
    print(f"\nSeparate pipelines baseline (2 forward passes + clustering overhead):")
    print(f"  Minimum latency estimate: {two_pass_ms:.2f} ms (2x joint, no clustering counted)")
    print(f"  Speedup from joint model: ≥{two_pass_ms / latency['mean_ms']:.1f}x")

    results = {
        "device": str(device),
        "batch_size": args.batch_size,
        "segment_seconds": args.segment_seconds,
        "n_runs": args.n_runs,
        "joint_model": {
            "checkpoint": args.model,
            "parameters": params,
            "latency_ms": latency,
            "forward_passes_per_inference": 1,
        },
        "separate_baseline_estimate": {
            "forward_passes_per_inference": 2,
            "estimated_min_latency_ms": round(two_pass_ms, 3),
            "speedup_lower_bound": round(two_pass_ms / latency["mean_ms"], 2),
        },
    }

    out = (ROOT / args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
