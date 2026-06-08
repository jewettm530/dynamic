import torchvision
try:
    import timm
    timm_available = True
except ImportError:
    timm_available = False
    print("Install timm for ViT comparison: pip install timm")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n=== EchoNet-Dynamic CNN Models ===")
for name in ["r2plus1d_18", "r3d_18", "mc3_18"]:
    model = torchvision.models.video.__dict__[name](pretrained=False)
    print(f"{name:12s}: {count_params(model):,} trainable parameters")

if timm_available:
    print("\n=== Vision Transformer (ViT) for comparison ===")
    vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    print(f"ViT-Base (image): {count_params(vit):,} trainable parameters")

# Optional: video transformer
# videomae = timm.create_model("videomae_base_patch16_224", pretrained=False)
# print(f"VideoMAE-Base   : {count_params(videomae):,} trainable parameters")
