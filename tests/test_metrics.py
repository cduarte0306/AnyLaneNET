"""Tests for segmentation metrics."""

from __future__ import annotations

import torch
import pytest

from anylane.utils import compute_iou, compute_accuracy


def _make_batch(val: float, shape=(2, 1, 64, 64)) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, targets) where prediction perfectly matches target."""
    targets = torch.zeros(*shape[:-2], shape[-2], shape[-1]).long()
    # logit > 0  → sigmoid > 0.5 → predicted positive
    logits = torch.full(shape, val)
    return logits, targets


class TestComputeIoU:
    def test_perfect_all_negative(self) -> None:
        """All pixels background, model predicts all background → IoU ≈ 1."""
        logits = torch.full((2, 1, 32, 32), -5.0)   # sigmoid ≈ 0 → negative
        targets = torch.zeros(2, 32, 32).long()
        iou = compute_iou(logits, targets)
        assert pytest.approx(iou, abs=1e-3) == 1.0

    def test_perfect_all_positive(self) -> None:
        """All pixels lane, model predicts all lane → IoU ≈ 1."""
        logits = torch.full((2, 1, 32, 32), 5.0)
        targets = torch.ones(2, 32, 32).long()
        iou = compute_iou(logits, targets)
        assert pytest.approx(iou, abs=1e-3) == 1.0

    def test_completely_wrong(self) -> None:
        """Model predicts all positive but target is all negative → low IoU."""
        logits = torch.full((1, 1, 32, 32), 5.0)
        targets = torch.zeros(1, 32, 32).long()
        iou = compute_iou(logits, targets)
        assert iou < 0.1

    def test_output_is_float(self) -> None:
        logits = torch.zeros(2, 1, 16, 16)
        targets = torch.zeros(2, 16, 16).long()
        assert isinstance(compute_iou(logits, targets), float)

    def test_3d_input(self) -> None:
        """Accepts (N, H, W) shaped predictions."""
        logits = torch.full((2, 32, 32), -5.0)
        targets = torch.zeros(2, 32, 32).long()
        iou = compute_iou(logits, targets)
        assert 0.0 <= iou <= 1.0


class TestComputeAccuracy:
    def test_perfect_prediction(self) -> None:
        logits = torch.full((2, 1, 32, 32), -5.0)
        targets = torch.zeros(2, 32, 32).long()
        acc = compute_accuracy(logits, targets)
        assert pytest.approx(acc, abs=1e-4) == 1.0

    def test_output_range(self) -> None:
        logits = torch.randn(4, 1, 32, 32)
        targets = torch.randint(0, 2, (4, 32, 32))
        acc = compute_accuracy(logits, targets)
        assert 0.0 <= acc <= 1.0
