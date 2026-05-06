from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ForensicAudioDataset
from src.models.forensic_voice import ForensicVoice, ForensicVoiceConfig
from src.models.losses import SupConLoss
from src.training.trainer import _compute_epoch_der


def _write_vox_manifest(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with src_path.open() as f:
        for line in f:
            if line.strip() and json.loads(line).get("has_diarization", 0):
                rows.append(line)
    with dst_path.open("w") as f:
        f.writelines(rows)
    print(f"wrote {len(rows)} VoxConverse rows -> {dst_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diarization-only baseline.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/baseline_diarization")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs

    # VoxConverse-only manifest derived from clean joint train manifest.
    vox_train_path = ROOT / "data/manifests/clean/train_vox_manifest.jsonl"
    if not vox_train_path.exists():
        _write_vox_manifest(ROOT / cfg["data"]["train_path"], vox_train_path)

    project_root = Path(cfg["data"].get("project_root", ".")).resolve()
    sample_rate = cfg["data"].get("sample_rate", 16000)
    segment_seconds = cfg["data"].get("segment_seconds", 2.0)

    train_dataset = ForensicAudioDataset(
        manifest_path=str(vox_train_path),
        split="train",
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        project_root=str(project_root),
    )

    # Fixed speaker vocabulary for the diarization CE head.
    speaker_ids = [int(item["speaker_id"]) for item in train_dataset.items]
    unique_ids = sorted(set(speaker_ids))
    speaker_id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
    n_speakers = max(2, len(unique_ids))
    print(f"VoxConverse train: {len(train_dataset)} segments, {n_speakers} unique speakers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model = ForensicVoice(ForensicVoiceConfig(**cfg["model"])).to(device)
    model.set_n_speakers(n_speakers)

    contrastive_loss = SupConLoss(temperature=cfg["training"]["temperature"])
    ce_loss = nn.CrossEntropyLoss()
    w_con = cfg["training"]["loss_weights"]["contrastive"]
    w_diar = cfg["training"]["loss_weights"]["diarization"]

    head_lr = cfg["training"]["learning_rate"]
    backbone_lr = head_lr * 0.1
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

    nw = cfg["training"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=cfg["training"].get("pin_memory", False),
        persistent_workers=cfg["training"].get("persistent_workers", False) and nw > 0,
        prefetch_factor=cfg["training"].get("prefetch_factor", 2) if nw > 0 else None,
    )

    ckpt_dir = (ROOT / args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_der = float("inf")
    history: list[dict] = []
    der_eval_files = int(cfg["training"].get("der_eval_files", 50))

    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"\n=== Baseline Diarization Epoch {epoch+1}/{cfg['training']['num_epochs']} ===")
        model.train()
        epoch_loss = 0.0
        epoch_con = 0.0
        epoch_diar = 0.0
        valid_steps = 0

        for batch in tqdm(train_loader, desc=f"baseline diar epoch {epoch+1}"):
            audio = batch["audio"].to(device, non_blocking=True)
            diar_labels = torch.tensor(
                [speaker_id_to_idx.get(int(x), 0) for x in batch["speaker_id"].view(-1).tolist()],
                dtype=torch.long,
            ).to(device).clamp(0, n_speakers - 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(audio, n_speakers=n_speakers)
                emb = out["speaker_embedding"]
                diar_logits = out["diarization_logits"]

                l_con = contrastive_loss(emb, diar_labels)
                l_diar = ce_loss(diar_logits, diar_labels)
                loss = w_con * l_con + w_diar * l_diar
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            epoch_con += float(l_con.detach().cpu())
            epoch_diar += float(l_diar.detach().cpu())
            valid_steps += 1

        avg_loss = epoch_loss / max(1, valid_steps)
        avg_con = epoch_con / max(1, valid_steps)
        avg_diar = epoch_diar / max(1, valid_steps)
        der, _ = _compute_epoch_der(model, project_root=project_root, device=device, max_files=der_eval_files)
        lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"epoch={epoch+1} train_loss={avg_loss:.4f} "
            f"[con={avg_con:.4f} diar={avg_diar:.4f}] "
            f"val_der={der:.4f} lr={lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_contrastive": avg_con,
            "train_diarization": avg_diar,
            "val_der": float(der) if der == der else None,
            "lr": lr,
        })

        state = {"epoch": epoch + 1, "model_state": model.state_dict(), "val_der": float(der) if der == der else None}
        torch.save(state, ckpt_dir / "latest.pth")
        if der == der and der < best_der:
            best_der = der
            torch.save(state, ckpt_dir / "best_der.pth")
            print(f"  saved best_der.pth  (der={best_der:.4f})")

        scheduler.step()

    (ckpt_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"\nSaved baseline diarization checkpoints to: {ckpt_dir}")


if __name__ == "__main__":
    main()
