# add to 'utils' folder

import os
import torch
from PIL import Image


def save_binary_mask(logits, save_path, threshold=0.5):
    """
    Saves a prediction as a clean grayscale binary mask.

    Background = black, 0
    Object = white, 255
    """

    from pathlib import Path

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    mask = pred.squeeze().detach().cpu().numpy()
    mask = (mask * 255).astype("uint8")

    image = Image.fromarray(mask, mode="L")
    image.save(str(save_path))