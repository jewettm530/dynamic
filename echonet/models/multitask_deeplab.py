#Add to 'models' folder

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class MultitaskDeepLabV3(nn.Module):
    """
    Multitask DeepLabV3 model.

    Outputs:
    - segmentation logits: [B, 1, H, W]
    - classification logits: [B, num_classes]

    The segmentation branch predicts the LV mask.
    The classification branch uses pooled CNN features.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        self.base_model = deeplabv3_resnet50(weights=weights)

        # Replace segmentation classifier with binary mask output
        self.base_model.classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=1
        )

        # Classification head from backbone features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.base_model.backbone(x)

        seg_logits = self.base_model.classifier(features["out"])

        # Resize segmentation output back to input image size
        seg_logits = torch.nn.functional.interpolate(
            seg_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        pooled = self.avgpool(features["out"])
        pooled = torch.flatten(pooled, 1)
        class_logits = self.classifier(pooled)

        return {
            "segmentation": seg_logits,
            "classification": class_logits
        }