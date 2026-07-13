import torch
import torch.nn as nn

from echonet.losses.segmentation_losses import (
    BCEDiceLoss,
    DiceLoss,
)


class MultitaskLoss(nn.Module):
    def __init__(
        self,
        segmentation_loss: str = "bce_dice",
        seg_weight: float = 1.0,
        class_weight: float = 0.3,
    ):
        super().__init__()

        if segmentation_loss == "bce":
            self.segmentation_criterion = nn.BCEWithLogitsLoss()
        elif segmentation_loss == "dice":
            self.segmentation_criterion = DiceLoss()
        elif segmentation_loss == "bce_dice":
            self.segmentation_criterion = BCEDiceLoss()
        else:
            raise ValueError(
                f"Unsupported segmentation loss: {segmentation_loss}"
            )

        self.classification_criterion = nn.CrossEntropyLoss()

        self.seg_weight = seg_weight
        self.class_weight = class_weight

    def forward(self, outputs, masks, labels):
        segmentation_loss = self.segmentation_criterion(
            outputs["segmentation"],
            masks,
        )

        classification_loss = self.classification_criterion(
            outputs["classification"],
            labels,
        )

        total_loss = (
            self.seg_weight * segmentation_loss
            + self.class_weight * classification_loss
        )

        return total_loss, {
            "total_loss": total_loss.detach().item(),
            "segmentation_loss": segmentation_loss.detach().item(),
            "classification_loss": classification_loss.detach().item(),
        }