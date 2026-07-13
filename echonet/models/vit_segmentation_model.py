import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTSegmentationModel(nn.Module):
    """
    ViT encoder + simple segmentation decoder.

    Output format matches torchvision segmentation models:
        {"out": tensor of shape (B, 1, H, W)}
    """

    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        num_classes=1,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
            img_size=112,
        )

        feature_channels = self.encoder.feature_info.channels()[-1]

        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        feats = self.encoder(x)[-1]

        # Some ViT feature outputs may be NHWC instead of NCHW
        if feats.ndim == 4 and feats.shape[1] < feats.shape[-1]:
            feats = feats.permute(0, 3, 1, 2).contiguous()

        out = self.decoder(feats)
        out = F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {"out": out}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
