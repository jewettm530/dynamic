import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class MultitaskDeepLabV3(nn.Module):
    """
    DeepLabV3 multitask model with a shared ResNet-50 backbone.

    Branches:
        segmentation:
            Predicts one binary LV mask per input frame.

        classification:
            Predicts a binary EF category from shared backbone features.

    Output dictionary:
        {
            "segmentation": [B, 1, H, W],
            "classification": [B, num_classes],
        }
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        classification_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2 when using CrossEntropyLoss."
            )

        weights = (
            DeepLabV3_ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.base_model = deeplabv3_resnet50(
            weights=weights,
            aux_loss=True,
        )

        # The pretrained weights require the auxiliary classifier to be created,
        # but the multitask model does not use it.
        self.base_model.aux_classifier = None

        # DeepLabV3's classifier produces 256 channels immediately before
        # its final class-prediction convolution.
        self.base_model.classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=1,
        )

        self.classification_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet-50's final backbone feature map has 2048 channels.
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, classification_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classification_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, C, H, W], got {tuple(x.shape)}"
            )

        input_size = x.shape[-2:]

        features = self.base_model.backbone(x)
        shared_features = features["out"]

        segmentation_logits = self.base_model.classifier(
            shared_features
        )

        segmentation_logits = F.interpolate(
            segmentation_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        pooled_features = self.classification_pool(
            shared_features
        )

        classification_logits = self.classification_head(
            pooled_features
        )

        return {
            "segmentation": segmentation_logits,
            "classification": classification_logits,
        }