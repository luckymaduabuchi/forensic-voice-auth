from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import warnings

import numpy as np
import torch
import torchaudio
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.manifold import TSNE
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from src.data.loaders import get_dataloaders
from src.evaluation.metrics import compute_authenticity_metrics
from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig
from src.models.losses import JointForensicLoss

# Keep training logs readable by suppressing repetitive, non-fatal warnings.
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

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None


def _parse_rttm(rttm_path: Path) -> Annotation:
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


def _load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor | None:
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


def _predict_annotation(
    model: ForensicVoice,
    audio_path: Path,
    n_speakers: int,
    device: torch.device,
    sample_rate: int = 16000,
    window_s: float = 3.0,
    stride_s: float = 1.5,
    min_segment_s: float = 0.8,
) -> Annotation:
    signal = _load_audio(audio_path, sample_rate=sample_rate)
    pred = Annotation(uri=audio_path.stem)
    if signal is None:
        return pred

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

    # Guard against invalid embedding rows so DER eval never crashes training.
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
            cluster_valid = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(emb_valid)

        labels = np.zeros(len(segments), dtype=int)
        labels[valid_idx] = cluster_valid
        # If some rows were invalid, copy nearest previous valid label (or 0 at start).
        for i in range(1, len(labels)):
            if not valid_mask[i]:
                labels[i] = labels[i - 1]

    run_start = 0
    for i in range(1, len(labels) + 1):
        end_run = i == len(labels) or labels[i] != labels[i - 1]
        if not end_run:
            continue
        lbl = labels[i - 1]
        st = starts[run_start]
        et = starts[i - 1] + window_s
        if (et - st) >= min_segment_s:
            pred[Segment(st, et)] = f"spk{lbl}"
        run_start = i
    return pred


def _compute_epoch_der(
    model: ForensicVoice, project_root: Path, device: torch.device, max_files: int = 10
) -> tuple[float, list[float]]:
    """Returns (aggregate_DER, per_recording_DER_list)."""
    test_manifest = project_root / "data/manifests/val_manifest.jsonl"
    if not test_manifest.exists():
        return float("nan"), []

    import json

    items = []
    with test_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    seen_audio: set[str] = set()
    unique_items = []
    for item in items:
        key = item.get("audio_path", "")
        if key and key not in seen_audio:
            seen_audio.add(key)
            unique_items.append(item)
    items = unique_items

    agg_metric = DiarizationErrorRate()
    per_file_ders: list[float] = []
    count = 0
    for item in items:
        audio_rel = item.get("audio_path")
        if not audio_rel:
            continue
        audio_path = project_root / audio_rel
        rttm_rel = item.get("rttm_path")
        if not rttm_rel:
            continue
        rttm = project_root / rttm_rel
        if not audio_path.exists() or not rttm.exists():
            continue
        ref = _parse_rttm(rttm)
        hyp = _predict_annotation(model, audio_path, int(item.get("n_speakers", 2)), device=device)
        if len(hyp.labels()) == 0:
            continue
        file_metric = DiarizationErrorRate()
        file_metric(ref, hyp)
        agg_metric(ref, hyp)
        per_file_ders.append(float(abs(file_metric)))
        count += 1
        if count >= max_files:
            break

    if count == 0:
        return float("nan"), []
    return float(abs(agg_metric)), per_file_ders


def _save_training_artifacts(
    logs_dir: Path,
    history: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    embed_speaker_ids: np.ndarray | None = None,
    embed_auth_labels: np.ndarray | None = None,
    per_file_ders: list[float] | None = None,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "training_history.csv"
    json_path = logs_dir / "training_history.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_contrastive",
                "train_auth",
                "train_diar",
                "val_loss",
                "val_contrastive",
                "val_auth",
                "val_diar",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_der",
                "forensic_score",
                "lr",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if plt is None:
        return

    epochs = [h["epoch"] for h in history]

    # --- 1. Total loss curve ---
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(logs_dir / "loss_curve.png", dpi=180)
    plt.close()

    # --- 2. Per-component loss (contrastive / authenticity / diarization) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, label) in zip(
        axes,
        [("contrastive", "Contrastive"), ("auth", "Authenticity"), ("diar", "Diarization")],
    ):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="val")
        ax.set_title(f"{label} Loss")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(logs_dir / "loss_components.png", dpi=180)
    plt.close(fig)

    # --- 3. Val metrics over epochs (F1 / precision / recall / forensic score) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label in [
        ("val_f1", "F1"),
        ("val_precision", "Precision"),
        ("val_recall", "Recall"),
        ("forensic_score", "Forensic Score"),
    ]:
        ax.plot(epochs, [h[key] for h in history], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics over Epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(logs_dir / "val_metrics.png", dpi=180)
    plt.close(fig)

    # --- 4. DER over epochs ---
    der_vals = [h["val_der"] for h in history]
    valid_der = [(e, d) for e, d in zip(epochs, der_vals) if d == d]  # drop NaN
    if valid_der:
        ep_der, der_v = zip(*valid_der)
        plt.figure(figsize=(8, 4))
        plt.plot(ep_der, der_v, color="firebrick", marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("DER")
        plt.title("Validation Diarization Error Rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(logs_dir / "der_curve.png", dpi=180)
        plt.close()

    # --- 5. Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spoof", "genuine"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Validation Confusion Matrix (Latest Epoch)")
    fig.tight_layout()
    fig.savefig(logs_dir / "confusion_matrix_latest.png", dpi=180)
    plt.close(fig)

    # --- 6-8: need raw probabilities ---
    if y_probs is not None and len(np.unique(y_true)) == 2:
        # 6. ROC curve + AUC
        fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
        auc = roc_auc_score(y_true, y_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Validation)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(logs_dir / "roc_curve.png", dpi=180)
        plt.close()

        # 7. Precision-Recall curve + AP
        prec, rec, _ = precision_recall_curve(y_true, y_probs, pos_label=1)
        ap = average_precision_score(y_true, y_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Validation)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(logs_dir / "pr_curve.png", dpi=180)
        plt.close()

        # 8. Score distribution (genuine vs spoof)
        genuine_scores = y_probs[y_true == 1]
        spoof_scores = y_probs[y_true == 0]
        plt.figure(figsize=(7, 4))
        plt.hist(spoof_scores, bins=60, alpha=0.6, color="crimson", label=f"Spoof (n={len(spoof_scores)})", density=True)
        plt.hist(genuine_scores, bins=60, alpha=0.6, color="steelblue", label=f"Genuine (n={len(genuine_scores)})", density=True)
        plt.xlabel("Authenticity Score")
        plt.ylabel("Density")
        plt.title("Score Distribution (Validation)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(logs_dir / "score_distribution.png", dpi=180)
        plt.close()

    # --- 9. t-SNE of speaker embeddings ---
    if (
        embeddings is not None
        and embed_auth_labels is not None
        and embed_speaker_ids is not None
        and len(embeddings) >= 10
    ):
        try:
            n = min(2000, len(embeddings))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(embeddings), size=n, replace=False)
            emb_sub = embeddings[idx]
            auth_sub = embed_auth_labels[idx]
            spk_sub = embed_speaker_ids[idx]

            valid = np.isfinite(emb_sub).all(axis=1)
            emb_sub, auth_sub, spk_sub = emb_sub[valid], auth_sub[valid], spk_sub[valid]

            if len(emb_sub) >= 10:
                # Cap at 20 speakers so tab20 colormap is unambiguous
                unique_spk = sorted(set(spk_sub.tolist()))[:20]
                keep = np.isin(spk_sub, unique_spk)
                emb_sub, auth_sub, spk_sub = emb_sub[keep], auth_sub[keep], spk_sub[keep]

                perp = min(30, len(emb_sub) - 1)
                coords = TSNE(n_components=2, perplexity=perp, random_state=42, n_jobs=1).fit_transform(emb_sub)
                cmap = plt.cm.get_cmap("tab20", len(unique_spk))
                spk_to_color = {s: cmap(i) for i, s in enumerate(unique_spk)}

                fig, ax = plt.subplots(figsize=(8, 7))
                for is_genuine, marker, label in [(1, "o", "Genuine"), (0, "x", "Spoof")]:
                    mask = auth_sub == is_genuine
                    if mask.sum() == 0:
                        continue
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        c=[spk_to_color[s] for s in spk_sub[mask].tolist()],
                        marker=marker,
                        s=20,
                        alpha=0.7,
                        label=label,
                    )
                ax.set_title("t-SNE of Speaker Embeddings\n(color=speaker, shape=genuine○/spoof✕)")
                ax.legend(loc="upper right")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(logs_dir / "tsne_embeddings.png", dpi=180)
                plt.close(fig)
        except Exception:
            pass  # t-SNE is best-effort; don't crash training

    # --- 10. Calibration curve (reliability diagram) ---
    if y_probs is not None and len(np.unique(y_true)) == 2:
        try:
            n_bins = 10
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_centers, frac_pos = [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (y_probs >= lo) & (y_probs < hi)
                if mask.sum() > 0:
                    bin_centers.append((lo + hi) / 2)
                    frac_pos.append(y_true[mask].mean())
            if bin_centers:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
                ax.plot(bin_centers, frac_pos, "o-", color="steelblue", lw=2, label="Model")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives (genuine)")
                ax.set_title("Calibration Curve (Reliability Diagram)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(logs_dir / "calibration_curve.png", dpi=180)
                plt.close(fig)
        except Exception:
            pass

    # --- 11. Embedding cosine similarity matrix ---
    if (
        embeddings is not None
        and embed_speaker_ids is not None
        and len(embeddings) >= 10
    ):
        try:
            valid = np.isfinite(embeddings).all(axis=1)
            emb_v = embeddings[valid]
            spk_v = embed_speaker_ids[valid]

            unique_spk = sorted(set(spk_v.tolist()))
            # Sample up to 20 speakers with at least 2 embeddings
            selected = [s for s in unique_spk if (spk_v == s).sum() >= 2][:20]
            if len(selected) >= 4:
                mean_embs = []
                for s in selected:
                    m = emb_v[spk_v == s].mean(axis=0)
                    norm = np.linalg.norm(m)
                    mean_embs.append(m / norm if norm > 1e-8 else m)
                mean_embs = np.stack(mean_embs)
                sim_matrix = mean_embs @ mean_embs.T

                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
                ax.set_title(f"Embedding Cosine Similarity\n({len(selected)} speakers, mean embedding)")
                ax.set_xlabel("Speaker index")
                ax.set_ylabel("Speaker index")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(logs_dir / "cosine_similarity_matrix.png", dpi=180)
                plt.close(fig)
        except Exception:
            pass

    # --- 12. Per-recording DER histogram ---
    if per_file_ders and len(per_file_ders) >= 3:
        try:
            plt.figure(figsize=(7, 4))
            plt.hist(per_file_ders, bins=min(20, len(per_file_ders)), color="firebrick", edgecolor="white", alpha=0.85)
            plt.axvline(float(np.mean(per_file_ders)), color="k", linestyle="--", label=f"Mean={np.mean(per_file_ders):.3f}")
            plt.xlabel("DER per recording")
            plt.ylabel("Count")
            plt.title("Per-Recording DER Distribution (Validation)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(logs_dir / "der_histogram.png", dpi=180)
            plt.close()
        except Exception:
            pass


def run_training(config: dict) -> None:
    seed = int(config["training"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP causes gradient overflow with unfrozen backbone, blocking all learning.
    amp_enabled = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    model_cfg = ForensicVoiceConfig(**config["model"])
    model = ForensicVoice(model_cfg).to(device)
    train_loader, val_loader = get_dataloaders(config)

    # Keep diarization class space fixed for the whole run.
    # Re-creating the head per batch can destabilize/erase learning.
    fixed_n_speakers = None
    speaker_id_to_idx: dict[int, int] = {}
    try:
        speaker_ids = [
            int(item.get("speaker_id", 0))
            for item in getattr(train_loader.dataset, "items", [])
            if item is not None and int(item.get("has_diarization", 1 if int(item.get("n_speakers", 0)) > 0 else 0)) == 1
        ]
        unique_ids = sorted(set(speaker_ids))
        if len(unique_ids) > 0:
            speaker_id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
            fixed_n_speakers = max(2, len(unique_ids))
    except Exception:
        fixed_n_speakers = None
    if fixed_n_speakers is None:
        fixed_n_speakers = 2
    model.set_n_speakers(fixed_n_speakers)

    loss_fn = JointForensicLoss(
        w_contrastive=config["training"]["loss_weights"]["contrastive"],
        w_authenticity=config["training"]["loss_weights"]["authenticity"],
        w_diarization=config["training"]["loss_weights"]["diarization"],
        temperature=config["training"]["temperature"],
    )
    head_lr = config["training"]["learning_rate"] * 10  # 0.01: heads need fast adaptation
    backbone_lr = config["training"]["learning_rate"] * 0.1  # 0.0001: gentle backbone fine-tune
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": head_lr},
        ],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["decay_steps"],
        gamma=config["training"].get("decay_gamma", 0.1),
    )

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_forensic_score = -1e9
    project_root = Path(config["data"].get("project_root", ".")).resolve()
    der_eval_files = int(config["training"].get("der_eval_files", 10))
    decision_threshold = float(config["training"].get("decision_threshold", 0.5))
    logs_dir = checkpoint_dir / "logs"
    history: list[dict] = []

    for epoch in range(config["training"]["num_epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['num_epochs']} ===")
        model.train()
        epoch_loss = 0.0
        epoch_contrastive = 0.0
        epoch_auth_loss = 0.0
        epoch_diar_loss = 0.0
        valid_train_steps = 0
        skipped_train_steps = 0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch+1}"):
            audio = batch["audio"].to(device, non_blocking=True)
            auth_labels = batch["authenticity"].to(device, non_blocking=True)
            diar_mask = batch.get("has_diarization")
            if diar_mask is None:
                diar_mask = torch.ones_like(auth_labels, dtype=torch.bool)
            diar_mask = diar_mask.to(device, non_blocking=True).bool()
            diar_labels_raw = batch["speaker_id"]
            diar_labels_cpu = torch.tensor(
                [
                    speaker_id_to_idx.get(int(x), 0)
                    for x in diar_labels_raw.view(-1).tolist()
                ],
                dtype=torch.long,
            )
            diar_labels = diar_labels_cpu.to(device, non_blocking=True).clamp(min=0, max=fixed_n_speakers - 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(audio, n_speakers=fixed_n_speakers)
                loss_dict = loss_fn(
                    embeddings=outputs["speaker_embedding"],
                    auth_logits=outputs["authenticity_logits"],
                    diar_logits=outputs["diarization_logits"],
                    auth_labels=auth_labels,
                    diar_labels=diar_labels,
                    diar_mask=diar_mask,
                )
                total_loss = loss_dict["total_loss"]
            if not torch.isfinite(total_loss):
                skipped_train_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(total_loss.detach().cpu().item())
            epoch_contrastive += loss_dict.get("contrastive_loss", 0.0)
            epoch_auth_loss += loss_dict.get("authenticity_loss", 0.0)
            epoch_diar_loss += loss_dict.get("diarization_loss", 0.0)
            valid_train_steps += 1

        model.eval()
        all_probs = []
        all_labels = []
        all_embeddings = []
        all_speaker_ids = []
        val_loss_total = 0.0
        val_contrastive = 0.0
        val_auth_loss = 0.0
        val_diar_loss = 0.0
        valid_val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device, non_blocking=True)
                auth_labels = batch["authenticity"].to(device, non_blocking=True)
                diar_mask = batch.get("has_diarization")
                if diar_mask is None:
                    diar_mask = torch.ones_like(auth_labels, dtype=torch.bool)
                diar_mask = diar_mask.to(device, non_blocking=True).bool()
                diar_labels_raw = batch["speaker_id"]
                diar_labels_cpu = torch.tensor(
                    [
                        speaker_id_to_idx.get(int(x), 0)
                        for x in diar_labels_raw.view(-1).tolist()
                    ],
                    dtype=torch.long,
                )
                diar_labels = diar_labels_cpu.to(device, non_blocking=True).clamp(min=0, max=fixed_n_speakers - 1)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(audio, n_speakers=fixed_n_speakers)
                    probs = torch.sigmoid(outputs["authenticity_logits"])
                    loss_dict = loss_fn(
                        embeddings=outputs["speaker_embedding"],
                        auth_logits=outputs["authenticity_logits"],
                        diar_logits=outputs["diarization_logits"],
                        auth_labels=auth_labels,
                        diar_labels=diar_labels,
                        diar_mask=diar_mask,
                    )
                if not torch.isfinite(loss_dict["total_loss"]):
                    continue
                all_probs.append(probs.cpu())
                all_labels.append(auth_labels.cpu())
                all_embeddings.append(outputs["speaker_embedding"].detach().cpu())
                all_speaker_ids.append(diar_labels_raw.cpu())
                val_loss_total += float(loss_dict["total_loss"].detach().cpu().item())
                val_contrastive += loss_dict.get("contrastive_loss", 0.0)
                val_auth_loss += loss_dict.get("authenticity_loss", 0.0)
                val_diar_loss += loss_dict.get("diarization_loss", 0.0)
                valid_val_steps += 1

        if len(all_probs) == 0:
            probs = torch.zeros(1)
            labels = torch.zeros(1, dtype=torch.long)
            val_embeddings_np = None
            val_spk_np = None
        else:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            val_embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
            val_spk_np = torch.cat(all_speaker_ids, dim=0).numpy().astype(int)
        metrics = compute_authenticity_metrics(probs, labels, threshold=decision_threshold)
        der, per_file_ders = _compute_epoch_der(model, project_root=project_root, device=device, max_files=der_eval_files)
        if der != der:  # NaN check
            forensic_score = metrics["f1"]
        else:
            forensic_score = 0.5 * metrics["f1"] + 0.5 * (1.0 - der)

        avg_loss = epoch_loss / max(1, valid_train_steps)
        avg_contrastive = epoch_contrastive / max(1, valid_train_steps)
        avg_auth = epoch_auth_loss / max(1, valid_train_steps)
        avg_diar = epoch_diar_loss / max(1, valid_train_steps)
        avg_val_loss = val_loss_total / max(1, valid_val_steps)
        y_true = labels.numpy().astype(int)
        y_probs_np = probs.numpy().reshape(-1)
        y_pred = (y_probs_np >= decision_threshold).astype(int)
        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_contrastive": avg_contrastive,
                "train_auth": avg_auth,
                "train_diar": avg_diar,
                "val_loss": avg_val_loss,
                "val_contrastive": val_contrastive / max(1, valid_val_steps),
                "val_auth": val_auth_loss / max(1, valid_val_steps),
                "val_diar": val_diar_loss / max(1, valid_val_steps),
                "val_accuracy": float(metrics["accuracy"]),
                "val_precision": float(metrics["precision"]),
                "val_recall": float(metrics["recall"]),
                "val_f1": float(metrics["f1"]),
                "val_der": float(der),
                "forensic_score": float(forensic_score),
                "lr": lr,
            }
        )
        _save_training_artifacts(
            logs_dir=logs_dir,
            history=history,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs_np,
            embeddings=val_embeddings_np,
            embed_speaker_ids=val_spk_np,
            embed_auth_labels=y_true,
            per_file_ders=per_file_ders,
        )

        print(
            f"epoch={epoch+1} train_loss={avg_loss:.4f} "
            f"[con={avg_contrastive:.4f} auth={avg_auth:.4f} diar={avg_diar:.4f}] "
            f"val_loss={avg_val_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} "
            f"val_f1={metrics['f1']:.4f} val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} val_der={der:.4f} "
            f"forensic_score={forensic_score:.4f} "
            f"skipped_train_steps={skipped_train_steps}"
        )

        torch.save(
            {"epoch": epoch + 1, "model_state": model.state_dict(), "metrics": metrics},
            checkpoint_dir / "latest.pth",
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {"epoch": epoch + 1, "model_state": model.state_dict(), "metrics": metrics},
                checkpoint_dir / "best_f1.pth",
            )
        if forensic_score > best_forensic_score:
            best_forensic_score = forensic_score
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                    "val_der": der,
                    "forensic_score": forensic_score,
                },
                checkpoint_dir / "best_forensic_score.pth",
            )
        scheduler.step()
