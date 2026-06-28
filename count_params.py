import torch
import torchvision

try:
    import timm
    timm_available = True
except ImportError:
    timm_available = False
    print("Install timm for ViT comparison: pip install timm")

try:
    from echonet.utils.vit_segmentation_model import ViTSegmentationModel
    vit_segmentation_available = True
except ImportError:
    vit_segmentation_available = False
    print("Could not import ViTSegmentationModel from echonet.utils.vit_segmentation_model")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def print_model_params(name, model):
    total = count_total_params(model)
    trainable = count_trainable_params(model)

    print(f"{name:35s}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print()


print("\n==================================================")
print("EchoNet-Dynamic Segmentation Model Parameter Counts")
print("==================================================")

# --------------------------------------------------
# CNN segmentation baseline: DeepLabV3-ResNet50
# --------------------------------------------------

cnn = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None,
    weights_backbone=None,
    aux_loss=False
)

# Match EchoNet binary LV segmentation output
cnn.classifier[-1] = torch.nn.Conv2d(
    cnn.classifier[-1].in_channels,
    1,
    kernel_size=cnn.classifier[-1].kernel_size
)

print_model_params("CNN: DeepLabV3-ResNet50", cnn)


# --------------------------------------------------
# Standard ViT image model only
# This is useful for simple model complexity comparison.
# --------------------------------------------------

if timm_available:
    vit_base = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        img_size=112,
        dynamic_img_size=True,
    )

    print_model_params("Standard ViT-Base image model", vit_base)


# --------------------------------------------------
# ViT segmentation model
# This should match the model you actually train.
# --------------------------------------------------

if vit_segmentation_available:
    vit_seg = ViTSegmentationModel(
        model_name="vit_base_patch16_224",
        pretrained=False,
        num_classes=1,
    )

    print_model_params("ViT segmentation model", vit_seg)


print("Notes:")
print("- CNN count uses DeepLabV3-ResNet50 with a 1-channel LV segmentation head.")
print("- Standard ViT count is for model-complexity comparison only.")
print("- ViT segmentation count is the model that should be trained for the CNN vs ViT experiment.")
print("- Pretrained=True vs pretrained=False usually does not change parameter count.")
print("- Input size is set to 112x112 to match the EchoNet CNN segmentation setup.")