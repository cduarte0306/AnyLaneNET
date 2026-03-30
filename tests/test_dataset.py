"""Tests for the LaneDataset loader."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from anylane.data import LaneDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path: Path, n: int = 4, h: int = 64, w: int = 64) -> tuple[Path, Path]:
    """Create *n* dummy image/mask pairs and return (images_dir, masks_dir)."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for i in range(n):
        name = f"frame_{i:04d}.png"
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
        cv2.imwrite(str(images_dir / name), img)
        cv2.imwrite(str(masks_dir / name), mask)

    return images_dir, masks_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLaneDataset:
    def test_len(self, tmp_path: Path) -> None:
        images_dir, masks_dir = _make_dataset(tmp_path, n=6)
        ds = LaneDataset(images_dir, masks_dir)
        assert len(ds) == 6

    def test_item_shapes(self, tmp_path: Path) -> None:
        images_dir, masks_dir = _make_dataset(tmp_path, n=2, h=64, w=64)
        ds = LaneDataset(images_dir, masks_dir)
        img, mask = ds[0]
        assert img.shape == (3, 64, 64), f"Unexpected image shape: {img.shape}"
        assert mask.shape == (64, 64), f"Unexpected mask shape: {mask.shape}"

    def test_mask_binary(self, tmp_path: Path) -> None:
        """Mask values should be 0 or 1 after binarisation."""
        images_dir, masks_dir = _make_dataset(tmp_path, n=3)
        ds = LaneDataset(images_dir, masks_dir)
        for i in range(len(ds)):
            _, mask = ds[i]
            unique = set(mask.numpy().flatten().tolist())
            assert unique <= {0, 1}, f"Non-binary mask values: {unique}"

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            LaneDataset(images_dir, masks_dir)

    def test_with_transforms(self, tmp_path: Path) -> None:
        from anylane.data import get_val_transforms

        images_dir, masks_dir = _make_dataset(tmp_path, n=2, h=128, w=128)
        ds = LaneDataset(images_dir, masks_dir, transform=get_val_transforms(64, 64))
        img, mask = ds[0]
        assert img.shape == (3, 64, 64)
        assert mask.shape == (64, 64)
