"""Shared binary classification and segmentation evaluation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


EPSILON = 1e-8


def safe_roc_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Return ROC-AUC or NaN when only one target class is present."""

    targets = np.asarray(targets).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)

    if targets.size == 0 or np.unique(targets).size < 2:
        return float("nan")

    return float(
        roc_auc_score(
            targets,
            probabilities,
        )
    )


def metrics_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> Dict[str, float]:
    """Calculate binary metrics from a global confusion matrix."""

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    f1 = (
        2.0 * precision * recall
        / max(precision + recall, EPSILON)
    )

    dice = (
        2.0 * tp
        / max(2 * tp + fp + fn, 1)
    )

    iou = (
        tp
        / max(tp + fp + fn, 1)
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "dice": float(dice),
        "iou": float(iou),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class BinaryMetricAccumulator:
    """
    Accumulate global binary metrics over an epoch.

    Confusion counts use every prediction. ROC-AUC uses a reproducible
    probability sample to avoid retaining every segmentation pixel.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_auc_samples: int = 1_000_000,
        seed: int = 0,
    ):
        self.threshold = threshold
        self.max_auc_samples = max_auc_samples
        self.rng = np.random.default_rng(seed)

        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

        self.auc_targets = []
        self.auc_probabilities = []
        self.auc_sample_count = 0

    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add one batch of probabilities and binary targets."""

        probabilities = (
            probabilities.detach()
            .float()
            .reshape(-1)
            .cpu()
        )

        targets = (
            targets.detach()
            .float()
            .reshape(-1)
            .cpu()
        )

        target_binary = targets >= 0.5
        prediction_binary = probabilities >= self.threshold

        self.tp += int(
            torch.logical_and(
                prediction_binary,
                target_binary,
            ).sum().item()
        )

        self.tn += int(
            torch.logical_and(
                ~prediction_binary,
                ~target_binary,
            ).sum().item()
        )

        self.fp += int(
            torch.logical_and(
                prediction_binary,
                ~target_binary,
            ).sum().item()
        )

        self.fn += int(
            torch.logical_and(
                ~prediction_binary,
                target_binary,
            ).sum().item()
        )

        remaining = (
            self.max_auc_samples
            - self.auc_sample_count
        )

        if remaining <= 0:
            return

        probability_array = probabilities.numpy()
        target_array = target_binary.numpy().astype(np.uint8)

        sample_size = min(
            remaining,
            probability_array.size,
        )

        if sample_size < probability_array.size:
            indices = self.rng.choice(
                probability_array.size,
                size=sample_size,
                replace=False,
            )

            probability_array = probability_array[indices]
            target_array = target_array[indices]

        self.auc_probabilities.append(probability_array)
        self.auc_targets.append(target_array)
        self.auc_sample_count += sample_size

    def compute(self) -> Dict[str, float]:
        """Return all accumulated binary metrics."""

        metrics = metrics_from_counts(
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            tp=self.tp,
        )

        if self.auc_targets:
            targets = np.concatenate(self.auc_targets)
            probabilities = np.concatenate(
                self.auc_probabilities
            )

            metrics["roc_auc"] = safe_roc_auc(
                targets,
                probabilities,
            )
        else:
            metrics["roc_auc"] = float("nan")

        return metrics