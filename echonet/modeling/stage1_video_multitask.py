"""Corrected Stage 1 video EF + sparse segmentation architecture.

The EF path is a standard R(2+1)D-18 video encoder followed by spatiotemporal
(global T/H/W) pooling and one scalar regression head.  It therefore produces
one EF prediction from a multi-frame clip and never needs ED/ES locations.

The segmentation path reuses the *same* R(2+1)D encoder on the expert-labeled
ED/ES frames (represented as one-frame clips) and decodes a binary LV mask.
ED/ES frames/indices are supplied only to ``forward_segmentation`` during
training/evaluation.  ``forward_ef`` accepts video alone.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from torchvision.models.video import R2Plus1D_18_Weights
except ImportError:  # pragma: no cover - compatibility with older torchvision
    R2Plus1D_18_Weights = None


def _build_pretrained_r2plus1d18(pretrained: bool):
    if R2Plus1D_18_Weights is not None:
        return torchvision.models.video.r2plus1d_18(
            weights=(R2Plus1D_18_Weights.DEFAULT if pretrained else None)
        )
    return torchvision.models.video.r2plus1d_18(pretrained=pretrained)


class _SegmentationDecoder(nn.Module):
    """Lightweight FPN-style decoder from R(2+1)D spatial feature maps."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.lateral1 = nn.Conv2d(64, width, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, width, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, width, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, width, kernel_size=1)
        self.refine3 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, 1, kernel_size=1),
        )

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        f1 = features["layer1"].squeeze(2)
        f2 = features["layer2"].squeeze(2)
        f3 = features["layer3"].squeeze(2)
        f4 = features["layer4"].squeeze(2)

        p4 = self.lateral4(f4)
        p3 = self.refine3(self.lateral3(f3) + self._up_to(p4, f3))
        p2 = self.refine2(self.lateral2(f2) + self._up_to(p3, f2))
        p1 = self.refine1(self.lateral1(f1) + self._up_to(p2, f1))
        logits = self.out(p1)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


class Stage1VideoMultitaskModel(nn.Module):
    """Shared R(2+1)D-18 encoder with EF and LV-segmentation heads.

    All instances construct the same encoder and both heads.  B1 simply calls
    ``forward_ef``; B2 calls ``forward_segmentation``; B3 calls both.  This
    keeps encoder/head initialization and the EF path directly matched across
    the controlled comparison.
    """

    def __init__(
        self,
        pretrained: bool = True,
        segmentation_decoder_width: int = 128,
        ef_bias_fraction: float = 0.556,
    ) -> None:
        super().__init__()
        base = _build_pretrained_r2plus1d18(pretrained)

        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Explicit temporal/spatial aggregation before a single video-level EF.
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.ef_head = nn.Linear(512, 1)
        with torch.no_grad():
            self.ef_head.bias.fill_(float(ef_bias_fraction))

        self.segmentation_decoder = _SegmentationDecoder(
            width=segmentation_decoder_width
        )

    def _encode_final(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _encode_pyramid(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}

    def forward_ef(self, video: torch.Tensor) -> torch.Tensor:
        """Predict one EF value from video only.

        Parameters
        ----------
        video : Tensor[B, C, T, H, W]
            T must be > 2.  No ED/ES indices, masks, or labeled frames are
            accepted by this method.
        """
        if video.ndim != 5:
            raise ValueError(
                f"Expected video shape [B, C, T, H, W], got {tuple(video.shape)}"
            )
        if video.shape[2] <= 2:
            raise ValueError(
                f"Corrected EF requires T > 2 frames; received T={video.shape[2]}"
            )
        features = self._encode_final(video)
        video_features = self.temporal_pool(features).flatten(1)
        return self.ef_head(video_features).squeeze(1)

    def forward_segmentation(self, frames: torch.Tensor) -> torch.Tensor:
        """Segment labeled ED/ES frames using the encoder shared with EF."""
        if frames.ndim != 4:
            raise ValueError(
                f"Expected frame shape [N, C, H, W], got {tuple(frames.shape)}"
            )
        output_size = (int(frames.shape[-2]), int(frames.shape[-1]))
        one_frame_clips = frames.unsqueeze(2)
        pyramid = self._encode_pyramid(one_frame_clips)
        # For T=1 input, every pyramid tensor has temporal size 1.
        for name, value in pyramid.items():
            if value.shape[2] != 1:
                raise RuntimeError(
                    f"Unexpected temporal dimension in {name}: {value.shape[2]}"
                )
        return self.segmentation_decoder(pyramid, output_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Default forward is intentionally EF-only/video-only."""
        return self.forward_ef(video)
