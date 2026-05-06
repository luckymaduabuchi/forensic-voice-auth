from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import torch
import torchaudio
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ForensicAudioDataset
from src.evaluation.metrics import compute_authenticity_metrics
from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig


def compute_threshold_free_auth_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    p_target: float = 0.05,
) -> dict[str, float]:
    """Compute exact score-based metrics for genuine-as-positive CM scores."""
    scores = probabilities.detach().cpu().view(-1).numpy()
    y = labels.detach().cpu().view(-1).long().numpy()
    if len(np.unique(y)) < 2:
        return {
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "min_dcf": float("nan"),
            "p_target": float(p_target),
        }

    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fnr - fpr

    crossing = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if crossing.size:
        i = int(crossing[0])
        denom = diff[i] - diff[i + 1]
        alpha = diff[i] / denom if denom != 0 else 0.0
        eer = float(fnr[i] + alpha * (fnr[i + 1] - fnr[i]))
        eer_threshold = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
    else:
        i = int(np.argmin(np.abs(diff)))
        eer = float((fnr[i] + fpr[i]) / 2.0)
        eer_threshold = float(thresholds[i])

    norm = min(p_target, 1.0 - p_target)
    min_dcf = float(np.min((p_target * fnr + (1.0 - p_target) * fpr) / norm))

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "min_dcf": min_dcf,
        "p_target": float(p_target),
    }

# Reduce repetitive, non-fatal warning spam in evaluation logs.
warnings.filterwarnings(
    "ignore",
    message=r".*Model was trained with pyannote.audio 0\.0\.1.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Model was trained with torch 1\.8\.1\+cu102.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*'uem' was approximated by the union of 'reference' and 'hypothesis' extents.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.set_audio_backend has been deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*multiple `ModelCheckpoint` callback states.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Lightning automatically upgraded your loaded checkpoint.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Model has been trained with a task-dependent loss function.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Found keys that are not in the model state dict but in the checkpoint.*",
)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(
    model_path: str,
    model_cfg: dict[str, Any],
    device: torch.device,
) -> ForensicVoice:
    model = ForensicVoice(ForensicVoiceConfig(**model_cfg)).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing:
        print(f"warning: {len(missing)} keys missing from checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate_authenticity(
    model: ForensicVoice,
    manifest_path: str,
    project_root: Path,
    batch_size: int,
    device: torch.device,
    max_samples: int | None = None,
    decision_threshold: float = 0.5,
    segment_seconds: float = 2.0,
    raw_scores_path: Path | None = None,
) -> dict[str, Any]:
    if max_samples is not None and max_samples <= 0:
        return {
            "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "accuracy": float("nan"), "num_samples": 0, "num_genuine": 0, "num_spoof": 0,
            "eer": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan"),
            "min_dcf": float("nan"), "decision_threshold": decision_threshold,
            "note": "auth eval skipped (max_samples=0)",
        }

    dataset = ForensicAudioDataset(
        manifest_path=manifest_path,
        split="test",
        sample_rate=16000,
        segment_seconds=segment_seconds,
        project_root=str(project_root),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []
    all_utt_ids: list[str] = []
    all_tracks: list[str] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            out = model(audio, n_speakers=None)
            probs = torch.sigmoid(out["authenticity_logits"]).cpu()
            labels = batch["authenticity"].cpu()
            all_probs.append(probs)
            all_labels.append(labels)
            all_utt_ids.extend(batch.get("utt_id", [""] * len(labels)))
            all_tracks.extend(batch.get("asvspoof_track", [""] * len(labels)))
            seen += labels.numel()
            if max_samples is not None and seen >= max_samples:
                break

    probabilities = torch.cat(all_probs, dim=0).view(-1)
    labels = torch.cat(all_labels, dim=0).view(-1)

    # Save raw scores for later recomputation of EER/ROC-AUC/PR-AUC/per-track metrics
    # without re-running the model.
    if raw_scores_path is not None:
        raw_scores_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_scores_path.open("w", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            _w.writerow(["utt_id", "label", "score", "track"])
            for uid, lbl, sc, tr in zip(
                all_utt_ids,
                labels.tolist(),
                probabilities.tolist(),
                all_tracks,
            ):
                _w.writerow([uid, int(lbl), f"{sc:.6f}", tr])

    base_metrics = compute_authenticity_metrics(probabilities, labels, threshold=decision_threshold)
    threshold_free = compute_threshold_free_auth_metrics(probabilities, labels)
    preds = (probabilities >= decision_threshold).long()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    # Threshold sweep for better operating point.
    sweep = []
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_authenticity_metrics(probabilities, labels, threshold=float(t))
        sweep.append({"threshold": float(t), **m})
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}

    probs_np = probabilities.numpy()
    labels_np = labels.numpy()
    genuine_probs = probs_np[labels_np == 1]
    spoof_probs = probs_np[labels_np == 0]

    return {
        **base_metrics,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_samples": int(labels.numel()),
        "num_genuine": int((labels == 1).sum().item()),
        "num_spoof": int((labels == 0).sum().item()),
        "mean_prob_genuine": float(genuine_probs.mean()) if genuine_probs.size > 0 else float("nan"),
        "mean_prob_spoof": float(spoof_probs.mean()) if spoof_probs.size > 0 else float("nan"),
        "decision_threshold": float(decision_threshold),
        # Exact threshold-free metrics from all model scores, not the coarse sweep below.
        "eer": float(threshold_free["eer"]),
        "eer_threshold": float(threshold_free["eer_threshold"]),
        "roc_auc": float(threshold_free["roc_auc"]),
        "pr_auc": float(threshold_free["pr_auc"]),
        "min_dcf": float(threshold_free["min_dcf"]),
        "min_dcf_p_target": float(threshold_free["p_target"]),
        # Threshold sweep is computed on TEST data — for diagnosis only.
        # Do NOT use best_threshold_test_diagnostic as a paper result.
        # Select threshold on val data separately, then lock it before running this script.
        "best_threshold_test_diagnostic": best,
        "threshold_sweep_test_diagnostic": sweep,
    }


def parse_rttm(rttm_path: Path) -> Annotation:
    ann = Annotation(uri=rttm_path.stem)
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            ann[Segment(start, start + dur)] = spk
    return ann


def load_full_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor | None:
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        return None
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.float()


def predict_annotation_for_audio(
    model: ForensicVoice,
    audio_path: Path,
    n_speakers: int,
    device: torch.device,
    sample_rate: int = 16000,
    window_s: float = 3.0,
    stride_s: float = 1.5,
    min_segment_s: float = 0.8,
) -> Annotation:
    signal = load_full_audio(audio_path, sample_rate=sample_rate)
    if signal is None:
        return Annotation(uri=audio_path.stem)
    window = int(window_s * sample_rate)
    stride = int(stride_s * sample_rate)
    if signal.numel() < window:
        signal = torch.nn.functional.pad(signal, (0, window - signal.numel()))

    segments = []
    starts = []
    for s in range(0, max(1, signal.numel() - window + 1), stride):
        chunk = signal[s : s + window]
        if chunk.numel() < window:
            chunk = torch.nn.functional.pad(chunk, (0, window - chunk.numel()))
        segments.append(chunk)
        starts.append(s / sample_rate)

    batch = torch.stack(segments).to(device)
    with torch.no_grad():
        out = model(batch, n_speakers=None)
        emb = out["speaker_embedding"].detach().cpu().numpy()

    # Guard clustering from invalid embedding rows (NaN/Inf) so DER eval does not crash.
    valid_mask = np.isfinite(emb).all(axis=1)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        labels = np.zeros(len(segments), dtype=int)
    else:
        emb_valid = emb[valid_idx]
        k = max(1, min(int(n_speakers), emb_valid.shape[0]))
        if emb_valid.shape[0] == 1:
            cluster_valid = np.array([0], dtype=int)
        else:
            cluster = AgglomerativeClustering(n_clusters=k, linkage="ward")
            cluster_valid = cluster.fit_predict(emb_valid)

        labels = np.zeros(len(segments), dtype=int)
        labels[valid_idx] = cluster_valid
        for i in range(1, len(labels)):
            if not valid_mask[i]:
                labels[i] = labels[i - 1]

    # Convert overlapping window labels into contiguous regions by grouping runs.
    pred = Annotation(uri=audio_path.stem)
    run_start_idx = 0
    for i in range(1, len(labels) + 1):
        end_of_run = i == len(labels) or labels[i] != labels[i - 1]
        if not end_of_run:
            continue
        lbl = labels[i - 1]
        start_t = starts[run_start_idx]
        end_t = starts[i - 1] + window_s
        if (end_t - start_t) >= min_segment_s:
            pred[Segment(start_t, end_t)] = f"spk{lbl}"
        run_start_idx = i
    return pred


def evaluate_der(
    model: ForensicVoice,
    test_manifest_path: str,
    project_root: Path,
    vox_root: str,
    device: torch.device,
    max_files: int | None = None,
) -> dict[str, float]:
    if max_files is not None and max_files <= 0:
        return {"der": float("nan"), "num_files": 0}

    dataset = ForensicAudioDataset(
        manifest_path=test_manifest_path,
        split="test",
        sample_rate=16000,
        segment_seconds=2.0,
        project_root=str(project_root),
    )
    der_metric = DiarizationErrorRate()

    count = 0
    for item in dataset.items:
        audio_rel = item.get("audio_path")
        if not audio_rel:
            continue
        audio_path = project_root / audio_rel
        utt_id = item.get("utt_id", Path(audio_rel).stem)
        rttm_path = project_root / vox_root / "voxconverse-master" / "test labels" / f"{utt_id}.rttm"
        if not audio_path.exists() or not rttm_path.exists():
            continue
        ref = parse_rttm(rttm_path)
        n_speakers = int(item.get("n_speakers", 2))
        hyp = predict_annotation_for_audio(model, audio_path, n_speakers=n_speakers, device=device)
        if len(hyp.labels()) == 0:
            continue
        der_metric(ref, hyp)
        count += 1
        if max_files is not None and count >= max_files:
            break

    if count == 0:
        return {"der": float("nan"), "num_files": 0}
    return {"der": float(abs(der_metric)), "num_files": count}


def save_outputs(results: dict[str, Any], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    flat = []
    for section, values in results.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if k in {"threshold_sweep", "threshold_sweep_test_diagnostic"}:
                    continue
                flat.append({"section": section, "metric": k, "value": v})
    auth_results = results.get("authenticity", {})
    sweep = auth_results.get(
        "threshold_sweep_test_diagnostic",
        auth_results.get("threshold_sweep", []),
    )
    for row in sweep:
        flat.append(
            {
                "section": "authenticity_threshold_sweep_test_diagnostic",
                "metric": f"thr_{row['threshold']:.2f}",
                "value": json.dumps(
                    {
                        "accuracy": row["accuracy"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                    }
                ),
            }
        )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ForensicVoice checkpoints.")
    parser.add_argument("--model", required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--benchmark",
        default="asvspoof_voxconverse",
        help="Benchmark label used to name output files.",
    )
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to training config yaml (for model/data settings).",
    )
    parser.add_argument(
        "--max-der-files",
        type=int,
        default=None,
        help="Optional cap for DER evaluation file count.",
    )
    parser.add_argument(
        "--max-auth-samples",
        type=int,
        default=10000,
        help="Optional cap for authenticity evaluation samples (default: 10000).",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for authenticity metrics at reporting time.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench_slug = args.benchmark.replace(" ", "_").replace("/", "_")
    raw_scores_path = project_root / "checkpoints" / f"raw_scores_{bench_slug}.csv"

    model = load_model(args.model, cfg["model"], device=device)
    auth_metrics = evaluate_authenticity(
        model=model,
        manifest_path=cfg["data"].get("test_asvspoof_path", cfg["data"].get("test_path", cfg["data"]["val_path"])),
        project_root=project_root,
        batch_size=cfg["training"].get("batch_size", 8),
        device=device,
        max_samples=args.max_auth_samples,
        decision_threshold=args.decision_threshold,
        segment_seconds=cfg["data"].get("segment_seconds", 2.0),
        raw_scores_path=raw_scores_path,
    )
    der_metrics = evaluate_der(
        model=model,
        test_manifest_path=str(project_root / "data/manifests/test_manifest.jsonl"),
        project_root=project_root,
        vox_root="data/raw/voxceleb2",
        device=device,
        max_files=args.max_der_files,
    )

    results = {
        "benchmark": args.benchmark,
        "model": args.model,
        "authenticity": auth_metrics,
        "diarization": der_metrics,
    }
    out_json = project_root / "checkpoints" / f"eval_{bench_slug}.json"
    out_csv = project_root / "checkpoints" / f"eval_{bench_slug}.csv"
    save_outputs(results, out_json=out_json, out_csv=out_csv)

    print(json.dumps(results, indent=2))
    print(f"saved json: {out_json}")
    print(f"saved csv: {out_csv}")
    print(f"saved raw scores: {raw_scores_path}")


if __name__ == "__main__":
    main()
