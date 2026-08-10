"""Metrics required by the EchoNet-Dynamic Stage 1 baseline plan."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


EPS = 1e-8


def regression_metrics(targets, predictions) -> Dict[str, float]:
    """MAE/RMSE in EF percentage points, R^2, and Pearson r."""
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    yhat = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if y.size != yhat.size or y.size == 0:
        raise ValueError("targets and predictions must be non-empty and equal length")

    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat) if y.size >= 2 else float("nan")

    if y.size >= 2 and np.std(y) > 0 and np.std(yhat) > 0:
        r, _ = pearsonr(y, yhat)
    else:
        r = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(r),
    }


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    intersection = np.logical_and(pred, true).sum()
    denominator = pred.sum() + true.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * intersection) / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def hd95_pixels(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Symmetric 95th percentile Hausdorff distance in pixels.

    Empty-mask policy:
      * both empty -> 0
      * exactly one empty -> image diagonal (finite worst-case penalty)
    """
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    if pred.shape != true.shape:
        raise ValueError("pred_mask and true_mask must have the same shape")

    if not pred.any() and not true.any():
        return 0.0
    if not pred.any() or not true.any():
        h, w = pred.shape[-2:]
        return float(np.hypot(h - 1, w - 1))

    pred_surface = _surface(pred)
    true_surface = _surface(true)

    # Distance to nearest surface pixel. distance_transform_edt measures the
    # distance from each non-zero pixel to the nearest zero, so invert surface.
    dist_to_true = ndimage.distance_transform_edt(~true_surface)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface)

    d_pred_true = dist_to_true[pred_surface]
    d_true_pred = dist_to_pred[true_surface]
    distances = np.concatenate([d_pred_true, d_true_pred])
    return float(np.percentile(distances, 95))


def segmentation_frame_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-frame Dice and HD95 arrays."""
    probs = np.asarray(probabilities)
    truth = np.asarray(targets)

    if probs.ndim == 4 and probs.shape[1] == 1:
        probs = probs[:, 0]
    if truth.ndim == 4 and truth.shape[1] == 1:
        truth = truth[:, 0]
    if probs.shape != truth.shape:
        raise ValueError(f"Shape mismatch: probabilities={probs.shape}, targets={truth.shape}")

    pred = probs >= threshold
    truth = truth >= 0.5

    dice = np.asarray(
        [dice_score(p, t) for p, t in zip(pred, truth)], dtype=np.float64
    )
    hd95 = np.asarray(
        [hd95_pixels(p, t) for p, t in zip(pred, truth)], dtype=np.float64
    )
    return dice, hd95


def summarize_segmentation(
    ed_dice: Iterable[float],
    es_dice: Iterable[float],
    ed_hd95: Iterable[float],
    es_hd95: Iterable[float],
) -> Dict[str, float]:
    ed_dice = np.asarray(list(ed_dice), dtype=np.float64)
    es_dice = np.asarray(list(es_dice), dtype=np.float64)
    ed_hd95 = np.asarray(list(ed_hd95), dtype=np.float64)
    es_hd95 = np.asarray(list(es_hd95), dtype=np.float64)

    if min(ed_dice.size, es_dice.size, ed_hd95.size, es_hd95.size) == 0:
        raise ValueError("Segmentation metric arrays cannot be empty")

    dice_ed = float(np.mean(ed_dice))
    dice_es = float(np.mean(es_dice))
    hd95_ed = float(np.mean(ed_hd95))
    hd95_es = float(np.mean(es_hd95))

    return {
        "dice_ed": dice_ed,
        "dice_es": dice_es,
        "mean_dice": float((dice_ed + dice_es) / 2.0),
        "hd95_ed": hd95_ed,
        "hd95_es": hd95_es,
        "mean_hd95": float((hd95_ed + hd95_es) / 2.0),
    }
