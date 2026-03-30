"""Albumentations transform pipelines for training and validation."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(height: int = 360, width: int = 640) -> A.Compose:
    """Return augmentation pipeline used during training.

    Augmentations applied:
    * Random horizontal flip
    * Random brightness / contrast
    * Gaussian blur (occasional)
    * Normalisation (ImageNet statistics)
    * Resize to (height, width)
    * Convert to ``torch.Tensor`` via ``ToTensorV2``
    """
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(height: int = 360, width: int = 640) -> A.Compose:
    """Return deterministic pipeline used during validation / inference."""
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
