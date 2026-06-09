import sys
sys.path.insert(0, '/data/jewettm/dynamic')

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from echonet.datasets.echo import Echo, loadvideo  # <-- correct import

device = torch.device("cpu")

# Load a random video from test set
dataset = Echo(split="test", length=32, period=2)
video, target = dataset[42]   # pick any index
print(f"Video shape: {video.shape}")

# Load trained EF model (best checkpoint)
checkpoint = torch.load("output/video/final_model/best.pt", map_location=device)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model = torchvision.models.video.r2plus1d_18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# Predict EF
input_tensor = torch.from_numpy(video).unsqueeze(0).to(device)
with torch.no_grad():
    ef_pred = model(input_tensor).item()
print(f"Predicted EF: {ef_pred:.1f}%")

# Visualise 8 frames
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    frame = video[:, i, :, :].transpose(1, 2, 0)
    frame = np.clip(frame, 0, 1)
    ax.imshow(frame)
    ax.set_title(f"Frame {i}")
    ax.axis("off")
plt.suptitle(f"Predicted EF = {ef_pred:.1f}%")
plt.savefig("video_frames.png", dpi=150)
print("Saved video_frames.png")