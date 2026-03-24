"""Lane segmentation dataset loader.

Expected directory layout::

    data/
      raw/
        images/
          *.png  (or *.jpg)
        masks/
          *.png  (binary, 0 = background, 255 = lane)

Image and mask filenames must match (e.g. ``frame_0001.png`` ↔ ``frame_0001.png``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LaneDataset(Dataset):
    """PyTorch dataset for lane segmentation.

    Args:
        images_dir:  Path to directory containing RGB images.
        masks_dir:   Path to directory containing binary mask images.
        transform:   Albumentations ``Compose`` applied to (image, mask) pairs.
        img_suffix:  File extension used to list images (default: ``.png``).
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Optional[Callable] = None,
        img_suffix: str = ".png",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() == img_suffix
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No '{img_suffix}' images found in {self.images_dir}"
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise OSError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)  # binarise

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors if not already done by albumentations ToTensorV2
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return image, mask
