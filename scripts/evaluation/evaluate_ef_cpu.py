import sys
sys.path.insert(0, '/data/jewettm/dynamic')

import torch
import torchvision
import numpy as np
import echonet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import time

device = torch.device("cpu")

# Load checkpoint
checkpoint = torch.load("output/video/final_model/best.pt", map_location=device)

# Remove 'module.' prefix from state_dict keys
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

# Create model
model = torchvision.models.video.r2plus1d_18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Load the stripped state dict with strict=False
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

print("Model loaded successfully")

# Dataset parameters
mean = np.array([0.0, 0.0, 0.0])
std  = np.array([1.0, 1.0, 1.0])
kwargs = {
    "target_type": "EF",
    "mean": mean,
    "std": std,
    "length": 32,
    "period": 2
}

test_dataset = echonet.datasets.Echo(split="test", **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=False
)

print("Running test evaluation on CPU (may take a few minutes)...")
start = time.time()
loss, yhat, y = echonet.utils.video.run_epoch(model, test_loader, False, None, device)
print(f"Time: {time.time()-start:.1f}s")

r2 = r2_score(y, yhat)
mae = mean_absolute_error(y, yhat)
rmse = math.sqrt(mean_squared_error(y, yhat))

print(f"\n===== Test results (no test‑time augmentation) =====")
print(f"R²:  {r2:.3f}")
print(f"MAE: {mae:.2f}%")
print(f"RMSE:{rmse:.2f}%")
