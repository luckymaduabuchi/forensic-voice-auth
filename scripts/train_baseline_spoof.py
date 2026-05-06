from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import get_dataloaders
from src.evaluation.metrics import compute_authenticity_metrics
from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _save_training_artifacts(
    logs_dir: Path,
    history: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "mean_prob_genuine",
                "mean_prob_spoof",
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
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Spoof Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(logs_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline Spoof Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(logs_dir / "accuracy_curve.png", dpi=180)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spoof", "genuine"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Baseline Spoof Confusion Matrix (Latest Epoch)")
    fig.tight_layout()
    fig.savefig(logs_dir / "confusion_matrix_latest.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Baseline A spoof-only model.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/baseline_spoof")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.num_epochs")
    parser.add_argument("--decision-threshold", type=float, default=None, help="Override training.decision_threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["num_epochs"] = int(args.epochs)
    if args.decision_threshold is not None:
        cfg["training"]["decision_threshold"] = float(args.decision_threshold)
    decision_threshold = float(cfg["training"].get("decision_threshold", 0.5))
    # AMP causes gradient overflow when fine-tuning the unfrozen backbone,
    # causing the GradScaler to skip most updates and produce zero learning.
    cfg["training"]["amp"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    model = ForensicVoice(ForensicVoiceConfig(**cfg["model"])).to(device)
    baseline_cfg = {**cfg, "data": {**cfg["data"],
        "train_path": cfg["data"]["train_asvspoof_path"],
        "val_path": cfg["data"]["val_asvspoof_path"],
    }}
    train_loader, val_loader = get_dataloaders(baseline_cfg, balance_by_diarization=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    head_lr = cfg["training"]["learning_rate"] * 10  # 0.01: head needs to adapt fast
    backbone_lr = cfg["training"]["learning_rate"] * 0.1  # 0.0001: gentle backbone fine-tune
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": head_lr},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg["training"]["decay_steps"],
        gamma=cfg["training"].get("decay_gamma", 0.1),
    )

    ckpt_dir = (ROOT / args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ckpt_dir / "logs"
    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"\n=== Baseline Spoof Epoch {epoch + 1}/{cfg['training']['num_epochs']} ===")
        model.train()
        train_loss = 0.0
        valid_steps = 0

        for batch in tqdm(train_loader, desc=f"baseline spoof train epoch {epoch+1}"):
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["authenticity"].float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(audio, n_speakers=None)
                logits = out["authenticity_logits"]
                loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                continue
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.detach().cpu().item())
            valid_steps += 1

        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device, non_blocking=True)
                labels = batch["authenticity"].float().unsqueeze(1).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(audio, n_speakers=None)
                    logits = out["authenticity_logits"]
                    loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    continue
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                probs = torch.sigmoid(logits).cpu()
                all_probs.append(probs)
                all_labels.append(labels.cpu())
                val_loss += float(loss.detach().cpu().item())
                val_steps += 1

        if len(all_probs) == 0:
            probs = torch.zeros(1)
            labels = torch.zeros(1, dtype=torch.long)
        else:
            probs = torch.cat(all_probs, dim=0).view(-1)
            labels = torch.cat(all_labels, dim=0).view(-1).long()

        metrics = compute_authenticity_metrics(probs, labels, threshold=decision_threshold)
        mean_prob_genuine = float(probs[labels == 1].mean().item()) if int((labels == 1).sum().item()) > 0 else float("nan")
        mean_prob_spoof = float(probs[labels == 0].mean().item()) if int((labels == 0).sum().item()) > 0 else float("nan")
        avg_train_loss = train_loss / max(1, valid_steps)
        avg_val_loss = val_loss / max(1, val_steps)
        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": float(metrics["accuracy"]),
                "val_precision": float(metrics["precision"]),
                "val_recall": float(metrics["recall"]),
                "val_f1": float(metrics["f1"]),
                "mean_prob_genuine": mean_prob_genuine,
                "mean_prob_spoof": mean_prob_spoof,
                "lr": lr,
            }
        )
        y_true = labels.numpy().astype(int)
        y_pred = (probs.numpy().reshape(-1) >= decision_threshold).astype(int)
        _save_training_artifacts(logs_dir=logs_dir, history=history, y_true=y_true, y_pred=y_pred)

        print(
            f"baseline_spoof epoch={epoch+1} "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} val_f1={metrics['f1']:.4f} "
            f"mean_prob_genuine={mean_prob_genuine:.4f} mean_prob_spoof={mean_prob_spoof:.4f}"
        )

        latest = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "metrics": metrics,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        torch.save(latest, ckpt_dir / "latest.pth")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(latest, ckpt_dir / "best_f1.pth")

        scheduler.step()

    print(f"\nSaved baseline spoof checkpoints to: {ckpt_dir}")


if __name__ == "__main__":
    main()
