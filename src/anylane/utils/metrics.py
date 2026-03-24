"""Evaluation metrics for lane segmentation."""

from __future__ import annotations

import torch


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute binary Intersection-over-Union (IoU / Jaccard index).

    Args:
        preds:     Model output logits or probabilities, shape ``(N, 1, H, W)``
                   or ``(N, H, W)``.
        targets:   Ground-truth binary masks (0/1), same spatial shape as *preds*.
        threshold: Binarisation threshold applied to *preds*.
        eps:       Small constant to prevent division by zero.

    Returns:
        Mean IoU across the batch as a Python float.
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)

    preds_bin = (torch.sigmoid(preds) >= threshold).long()
    targets = targets.long()

    intersection = (preds_bin & targets).float().sum(dim=(1, 2))
    union = (preds_bin | targets).float().sum(dim=(1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute pixel-wise classification accuracy.

    Args:
        preds:     Model output logits or probabilities, shape ``(N, 1, H, W)``
                   or ``(N, H, W)``.
        targets:   Ground-truth binary masks (0/1).
        threshold: Binarisation threshold applied to *preds*.

    Returns:
        Accuracy as a Python float in ``[0, 1]``.
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)

    preds_bin = (torch.sigmoid(preds) >= threshold).long()
    correct = (preds_bin == targets.long()).float()
    return correct.mean().item()
