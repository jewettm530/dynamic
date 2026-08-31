"""Model definitions used by EchoNet experiments."""

from .multitask_deeplab import MultitaskDeepLabV3
from .stage1_video_multitask import Stage1VideoMultitaskModel

try:  # ViT is optional for the corrected Stage 1 path.
    from .vit_segmentation_model import ViTSegmentationModel
except ModuleNotFoundError:  # pragma: no cover
    ViTSegmentationModel = None

__all__ = [
    "MultitaskDeepLabV3",
    "Stage1VideoMultitaskModel",
    "ViTSegmentationModel",
]
