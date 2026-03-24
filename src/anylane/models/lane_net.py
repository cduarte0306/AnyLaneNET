"""LaneNet – encoder-decoder CNN for binary lane segmentation.

Architecture:
  Encoder: ResNet-18 backbone (ImageNet pre-trained, optional).
  Decoder: progressive upsampling with skip connections.
  Output:  single-channel logit map (sigmoid → lane probability).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    """3×3 Conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    """Upsample × 2, optional skip-connection concat, then ConvBnRelu.

    Upsampling is done to the *exact* spatial size of the skip tensor when one
    is provided, so no off-by-one errors occur on odd input dimensions.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _ConvBnRelu(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


# ---------------------------------------------------------------------------
# LaneNet
# ---------------------------------------------------------------------------

class LaneNet(nn.Module):
    """Encoder-decoder network for lane segmentation.

    Args:
        num_classes: Number of output classes (default: 1 for binary segmentation).
        pretrained:  Whether to initialise the encoder with ImageNet weights.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ---- Encoder (ResNet-18) ----
        backbone = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.pool = backbone.maxpool                                              # /4
        self.enc1 = backbone.layer1   # 64  ch, /4
        self.enc2 = backbone.layer2   # 128 ch, /8
        self.enc3 = backbone.layer3   # 256 ch, /16
        self.enc4 = backbone.layer4   # 512 ch, /32

        # ---- Bottleneck ----
        self.bottleneck = _ConvBnRelu(512, 512)

        # ---- Decoder ----
        self.dec4 = _DecoderBlock(512, 256, 256)
        self.dec3 = _DecoderBlock(256, 128, 128)
        self.dec2 = _DecoderBlock(128, 64,  64)
        self.dec1 = _DecoderBlock(64,  64,  32)
        self.dec0 = _DecoderBlock(32,  0,   16)

        # ---- Head ----
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.enc0(x)          # H/2
        s1 = self.enc1(self.pool(s0))  # H/4
        s2 = self.enc2(s1)         # H/8
        s3 = self.enc3(s2)         # H/16
        s4 = self.enc4(s3)         # H/32

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder
        d4 = self.dec4(b,  s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        d0 = self.dec0(d1)

        return self.head(d0)
