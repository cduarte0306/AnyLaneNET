"""Visualization utilities for lane detection results."""

from __future__ import annotations

import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary segmentation mask on top of an RGB image.

    Args:
        image:  RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
        mask:   Binary mask, shape ``(H, W)``, values 0 or 1.
        color:  RGB colour used to highlight lane pixels.
        alpha:  Blending weight for the mask overlay (0 = invisible, 1 = opaque).

    Returns:
        Blended RGB image as ``numpy.ndarray`` with dtype ``uint8``.
    """
    overlay = image.copy()
    lane_pixels = mask.astype(bool)
    overlay[lane_pixels] = (
        np.array(color, dtype=np.uint8) * alpha
        + overlay[lane_pixels] * (1 - alpha)
    ).astype(np.uint8)
    return overlay
