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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    mask = pred.squeeze().detach().cpu().numpy()
    mask = (mask * 255).astype("uint8")

    image = Image.fromarray(mask, mode="L")
    image.save(save_path)