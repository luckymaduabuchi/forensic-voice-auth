from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import ForensicAudioDataset


def get_dataloaders(config: dict, balance_by_diarization: bool = True) -> tuple[DataLoader, DataLoader]:
    train_dataset = ForensicAudioDataset(
        manifest_path=config["data"]["train_path"],
        split="train",
        sample_rate=config["data"].get("sample_rate", 16000),
        segment_seconds=config["data"].get("segment_seconds", 2.0),
        project_root=config["data"].get("project_root"),
    )
    val_dataset = ForensicAudioDataset(
        manifest_path=config["data"]["val_path"],
        split="val",
        sample_rate=config["data"].get("sample_rate", 16000),
        segment_seconds=config["data"].get("segment_seconds", 2.0),
        project_root=config["data"].get("project_root"),
    )

    use_balanced_sampler = config["training"].get("balanced_sampler", True)
    train_sampler = None
    train_shuffle = True
    if use_balanced_sampler and len(train_dataset.items) > 0:
        if balance_by_diarization:
            # Joint training: first balance authenticity labels, then split each
            # authenticity label's probability mass across its diarization groups.
            # This prevents genuine-only diarization rows from making batches
            # genuine-heavy while still keeping diarization examples visible.
            group_counts: dict[tuple[int, int], int] = {}
            auth_groups: dict[int, set[int]] = {}
            for item in train_dataset.items:
                has_diarization = int(item.get("has_diarization", 0))
                authenticity = int(item.get("authenticity", 1))
                key = (has_diarization, authenticity)
                group_counts[key] = group_counts.get(key, 0) + 1
                auth_groups.setdefault(authenticity, set()).add(has_diarization)
            auth_mass = 1.0 / max(1, len(auth_groups))
            weights = []
            for item in train_dataset.items:
                has_diarization = int(item.get("has_diarization", 0))
                authenticity = int(item.get("authenticity", 1))
                key = (has_diarization, authenticity)
                group_mass = auth_mass / max(1, len(auth_groups[authenticity]))
                weights.append(group_mass / group_counts[key])
        else:
            # Authenticity-only training: balance genuine/spoof only.
            labels = [int(item.get("authenticity", 1)) for item in train_dataset.items]
            label_counts: dict[int, int] = {}
            for y in labels:
                label_counts[y] = label_counts.get(y, 0) + 1
            weights = [1.0 / label_counts[y] for y in labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=config["training"].get("pin_memory", False),
        persistent_workers=config["training"].get("persistent_workers", False)
        and config["training"].get("num_workers", 0) > 0,
        prefetch_factor=config["training"].get("prefetch_factor", 2)
        if config["training"].get("num_workers", 0) > 0
        else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=config["training"].get("pin_memory", False),
        persistent_workers=config["training"].get("persistent_workers", False)
        and config["training"].get("num_workers", 0) > 0,
        prefetch_factor=config["training"].get("prefetch_factor", 2)
        if config["training"].get("num_workers", 0) > 0
        else None,
    )
    return train_loader, val_loader
