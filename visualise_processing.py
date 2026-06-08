import torch
import numpy as np
import matplotlib.pyplot as plt
from echonet.datasets import Echo, loadvideo
from echonet.utils import get_model
import os

# Set paths
data_dir = os.environ.get("ECHONET_DATA_DIR", "/data/jewettm/dynamic/datasets")
model_path = "output/video/final_model/best.pt"   # or use a pretrained model

# Load a random video from test set
dataset = Echo(split="test", length=32, period=2)
video, target = dataset[5]   # pick an index
print(f"Video shape: {video.shape}")   # (3,32,112,112)

# Load EF model
model = get_model("r2plus1d_18", pretrained=True, device="cuda")  # or load checkpoint
# If you have a trained checkpoint:
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Predict EF
input_tensor = torch.from_numpy(video).unsqueeze(0).cuda()
with torch.no_grad():
    ef_pred = model(input_tensor).item()
print(f"Predicted EF: {ef_pred:.1f}%")

# Visualise 8 frames
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    frame = video[:, i, :, :].transpose(1, 2, 0)   # (H,W,3)
    ax.imshow(frame)
    ax.set_title(f"Frame {i}")
    ax.axis("off")
plt.suptitle(f"Predicted EF = {ef_pred:.1f}%")
plt.savefig("video_frames.png")
print("Saved video_frames.png")