"""Controlled single-task ablations of the Stage 1 multi-task model."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3

class EFOnlyDeepLabV3(nn.Module):
    """Same W2 backbone + EF head, with the segmentation branch removed."""
    def __init__(self, pretrained=True, regression_hidden_dim=256, dropout=0.3):
        super().__init__()
        reference = MultitaskDeepLabV3(
            pretrained=pretrained,
            regression_hidden_dim=regression_hidden_dim,
            dropout=dropout,
        )
        # Construct the full reference first so same seed => identical relevant init.
        self.backbone = reference.base_model.backbone
        self.ef_pool = reference.ef_pool
        self.ef_head = reference.ef_head
        del reference

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        features = self.backbone(x)["out"]
        return self.ef_head(self.ef_pool(features)).squeeze(1)

class SegmentationOnlyDeepLabV3(nn.Module):
    """Same W2 backbone + segmentation classifier, with the EF branch removed."""
    def __init__(self, pretrained=True, regression_hidden_dim=256, dropout=0.3):
        super().__init__()
        reference = MultitaskDeepLabV3(
            pretrained=pretrained,
            regression_hidden_dim=regression_hidden_dim,
            dropout=dropout,
        )
        self.backbone = reference.base_model.backbone
        self.classifier = reference.base_model.classifier
        del reference

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        input_size = x.shape[-2:]
        features = self.backbone(x)["out"]
        logits = self.classifier(features)
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
