"""Tests for the LaneNet model."""

from __future__ import annotations

import torch
import pytest

from anylane.models import LaneNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> LaneNet:
    """Return a LaneNet instance (no pretrained weights, for speed)."""
    return LaneNet(num_classes=1, pretrained=False).eval()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape(model: LaneNet) -> None:
    """Output spatial size should match input spatial size."""
    batch, c, h, w = 2, 3, 360, 640
    x = torch.zeros(batch, c, h, w)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch, 1, h, w), f"Unexpected output shape: {out.shape}"


def test_output_shape_small(model: LaneNet) -> None:
    """Model should handle smaller inputs (multiples of 32)."""
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 224, 224)


def test_forward_no_nan(model: LaneNet) -> None:
    """Forward pass should produce no NaN values."""
    x = torch.randn(2, 3, 360, 640)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN detected in model output"


def test_multi_class() -> None:
    """Model should support more than one output class."""
    m = LaneNet(num_classes=3, pretrained=False).eval()
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 3, 224, 224)


def test_num_classes_attribute() -> None:
    """num_classes attribute should reflect the value passed to the constructor."""
    m = LaneNet(num_classes=2, pretrained=False)
    assert m.num_classes == 2
