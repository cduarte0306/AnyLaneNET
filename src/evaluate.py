"""Evaluation / inference script for AnyLaneNET.

Usage::

    python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from anylane.data import LaneDataset, get_val_transforms
from anylane.models import LaneNet
from anylane.utils import compute_accuracy, compute_iou

logger = logging.getLogger(__name__)


def evaluate(cfg: dict, checkpoint: str) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    img_h, img_w = data_cfg["img_height"], data_cfg["img_width"]

    dataset = LaneDataset(
        images_dir=data_cfg["images_dir"],
        masks_dir=data_cfg["masks_dir"],
        transform=get_val_transforms(img_h, img_w),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    model = LaneNet(
        num_classes=cfg["model"].get("num_classes", 1),
        pretrained=False,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint, ckpt.get("epoch", -1))

    total_iou = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            total_iou += compute_iou(logits, masks)
            total_acc += compute_accuracy(logits, masks)

    n = len(loader)
    results = {"iou": total_iou / n, "accuracy": total_acc / n}
    logger.info("Results → IoU: %.4f | Accuracy: %.4f", results["iou"], results["accuracy"])
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AnyLaneNET")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate(config, args.checkpoint)
