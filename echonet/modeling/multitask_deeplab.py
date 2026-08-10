"""Naive multi-task DeepLabV3 model for Stage 1.

The model shares a ResNet-50/DeepLabV3 backbone between:
1. binary LV segmentation of a labeled ED/ES frame; and
2. continuous EF regression from the shared backbone features.

One forward pass accepts individual frames. The training script evaluates both
ED and ES frames for each video, averages the two frame-level EF predictions,
and uses that single video-level EF prediction for the regression loss/metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from torchvision.models import ResNet50_Weights


class MultitaskDeepLabV3(nn.Module):
    """Shared DeepLabV3-ResNet50 with segmentation and EF regression heads."""

    def __init__(
        self,
        pretrained: bool = True,
        regression_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None

        # aux_loss=True is required when loading torchvision's pretrained
        # segmentation weights. The auxiliary head is discarded afterwards.
        self.base_model = deeplabv3_resnet50(
            weights=weights,
            weights_backbone=(None if not pretrained else ResNet50_Weights.IMAGENET1K_V1),
            aux_loss=True if pretrained else False,
        )
        self.base_model.aux_classifier = None

        # DeepLabV3 classifier has 256 channels immediately before the final
        # prediction convolution. Replace the final layer with one LV logit map.
        final_layer = self.base_model.classifier[-1]
        if not isinstance(final_layer, nn.Conv2d):
            raise TypeError("Expected DeepLabV3 classifier[-1] to be Conv2d")
        self.base_model.classifier[-1] = nn.Conv2d(
            final_layer.in_channels,
            1,
            kernel_size=final_layer.kernel_size,
            stride=final_layer.stride,
            padding=final_layer.padding,
            dilation=final_layer.dilation,
            bias=True,
        )

        # ResNet-50 backbone output has 2048 channels.
        self.ef_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ef_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, regression_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(regression_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, C, H, W], got {tuple(x.shape)}"
            )

        input_size = x.shape[-2:]
        features = self.base_model.backbone(x)
        shared_features = features["out"]

        segmentation_logits = self.base_model.classifier(shared_features)
        segmentation_logits = F.interpolate(
            segmentation_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        ef_prediction = self.ef_head(self.ef_pool(shared_features)).squeeze(1)

        return {
            "segmentation": segmentation_logits,
            "ef": ef_prediction,
        }
