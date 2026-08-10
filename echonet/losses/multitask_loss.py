"""Stage 1 multi-task loss.

Stage 1 fixes the EF target scale to native EF percentage points (0-100) and
uses MSE for EF regression. The segmentation baseline uses BCEWithLogitsLoss;
therefore the naive MTL baseline uses the same segmentation loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultitaskLoss(nn.Module):
    """Weighted EF-regression + LV-segmentation loss."""

    def __init__(
        self,
        ef_weight: float,
        seg_weight: float,
    ) -> None:
        super().__init__()

        if ef_weight < 0 or seg_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if ef_weight + seg_weight <= 0:
            raise ValueError("At least one loss weight must be positive")

        self.ef_weight = float(ef_weight)
        self.seg_weight = float(seg_weight)
        self.ef_criterion = nn.MSELoss(reduction="mean")
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        segmentation_targets: torch.Tensor,
        ef_predictions: torch.Tensor,
        ef_targets: torch.Tensor,
    ):
        seg_loss = self.seg_criterion(
            segmentation_logits,
            segmentation_targets,
        )
        ef_loss = self.ef_criterion(
            ef_predictions.reshape(-1),
            ef_targets.reshape(-1),
        )

        weighted_ef = self.ef_weight * ef_loss
        weighted_seg = self.seg_weight * seg_loss
        total_loss = weighted_ef + weighted_seg

        components = {
            "raw_ef_loss": float(ef_loss.detach().cpu().item()),
            "raw_seg_loss": float(seg_loss.detach().cpu().item()),
            "weighted_ef_loss": float(weighted_ef.detach().cpu().item()),
            "weighted_seg_loss": float(weighted_seg.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }
        return total_loss, components
