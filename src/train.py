"""Training entry-point for AnyLaneNET.

Usage::

    python src/train.py --config configs/default.yaml

Override individual settings via ``key=value`` after the config path (Hydra-style
overrides are **not** used here; plain argparse is used for simplicity).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from anylane.data import LaneDataset, get_train_transforms, get_val_transforms
from anylane.models import LaneNet
from anylane.utils import compute_iou

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    images_dir = Path(data_cfg["images_dir"])
    masks_dir = Path(data_cfg["masks_dir"])
    img_h, img_w = data_cfg["img_height"], data_cfg["img_width"]

    full_ds = LaneDataset(images_dir, masks_dir, transform=get_train_transforms(img_h, img_w))

    val_size = max(1, int(len(full_ds) * data_cfg.get("val_split", 0.15)))
    train_size = len(full_ds) - val_size
    # Determine train/val split indices with a fixed seed for reproducibility.
    _generator = torch.Generator().manual_seed(42)
    train_ds, _unused_val_split = random_split(
        full_ds, [train_size, val_size], generator=_generator
    )

    # Rebuild val split with validation transforms (no augmentation), using the same seed.
    val_ds = LaneDataset(images_dir, masks_dir, transform=get_val_transforms(img_h, img_w))
    _generator_val = torch.Generator().manual_seed(42)
    _, val_ds_indices = random_split(
        val_ds, [train_size, val_size], generator=_generator_val
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds_indices,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader


def _save_checkpoint(model: nn.Module, path: Path, epoch: int, best_iou: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "best_iou": best_iou},
        path,
    )
    logger.info("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader = _build_dataloaders(cfg)

    model = LaneNet(
        num_classes=cfg["model"].get("num_classes", 1),
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    train_cfg = cfg["training"]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    writer = SummaryWriter(log_dir=cfg.get("log_dir", "logs"))
    best_iou = 0.0
    checkpoint_path = Path(cfg.get("checkpoint_dir", "checkpoints")) / "best.pth"

    for epoch in range(1, train_cfg["epochs"] + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']} [train]"):
            images = images.to(device, non_blocking=True)
            masks = masks.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']} [val]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                val_iou += compute_iou(logits, masks)

        val_iou /= len(val_loader)
        writer.add_scalar("IoU/val", val_iou, epoch)
        logger.info("Epoch %d | loss=%.4f | val_iou=%.4f", epoch, train_loss, val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            _save_checkpoint(model, checkpoint_path, epoch, best_iou)

    writer.close()
    logger.info("Training complete. Best IoU: %.4f", best_iou)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnyLaneNET")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)
