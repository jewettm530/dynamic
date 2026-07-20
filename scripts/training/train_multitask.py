"""
train_multitask.py

Train a multitask DeepLabV3 model on EchoNet-Dynamic.

Tasks
-----
1. Segment the left ventricle on:
   - Large/end-diastolic frames
   - Small/end-systolic frames
2. Classify reduced ejection fraction:
   - 0 = EF >= EF_THRESHOLD
   - 1 = EF < EF_THRESHOLD

Outputs
-------
output/comparison/multitask_25_epochs/
    checkpoint.pt
    best.pt
    training_history.csv
    validation_metrics.csv

Metric aggregation
------------------
Segmentation metrics are calculated from global pixel-level confusion counts
for the entire epoch. Metrics are reported for:
    - overall: large and small frames combined
    - large: end-diastolic frames
    - small: end-systolic frames

Classification metrics are patient-level. The positive-class probabilities
from each patient's large and small frames are averaged before calculating
accuracy, precision, recall, F1, specificity, ROC-AUC, and confusion counts.

This script requires:
    echonet/utils/evaluation_metrics.py

That module must provide:
    BinaryMetricAccumulator
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from echonet.datasets.echo import Echo
from echonet.losses.multitask_loss import MultitaskLoss
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.paths import (
    DATA_DIR,
    FILE_LIST_PATH,
    VIDEOS_DIR,
    VOLUME_TRACINGS_PATH,
)
from echonet.utils.evaluation_metrics import BinaryMetricAccumulator


MODEL_NAME = "multitask_deeplabv3_resnet50"

NUM_CLASSES = 2
EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_WORKERS = 4
EF_THRESHOLD = 40.0

SEGMENTATION_THRESHOLD = 0.5
SEGMENTATION_AUC_SAMPLES = 1_000_000
FRAME_AUC_SAMPLES = 500_000
CLASSIFICATION_AUC_SAMPLES = 100_000

OUTPUT_DIR = Path("output/comparison/multitask_25_epochs")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
VALIDATION_METRICS_PATH = OUTPUT_DIR / "validation_metrics.csv"


HISTORY_COLUMNS = [
    "model",
    "epoch",
    "phase",
    "frame_type",
    "total_loss",
    "segmentation_loss",
    "classification_loss",
    "dice",
    "iou",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "elapsed_seconds",
    "number_of_patients",
    "number_of_frames",
    "peak_gpu_memory_allocated",
    "peak_gpu_memory_reserved",
    "batch_size",
]


class EchoMultitaskDataset(Dataset):
    """
    Return large/small frames, masks, EF class, and continuous EF.

    Each item has:
        large_image: [3, H, W]
        small_image: [3, H, W]
        large_mask:  [1, H, W]
        small_mask:  [1, H, W]
        label:       scalar class index
        ef:          scalar EF value
    """

    def __init__(
        self,
        root: str,
        split: str,
        ef_threshold: float = EF_THRESHOLD,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        self.ef_threshold = ef_threshold

        self.dataset = Echo(
            root=root,
            split=split,
            target_type=[
                "LargeFrame",
                "SmallFrame",
                "LargeTrace",
                "SmallTrace",
                "EF",
            ],
            mean=mean,
            std=std,
            length=16,
            period=2,
            clips=1,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _prepare_frame(frame: Any) -> torch.Tensor:
        image = torch.as_tensor(
            frame,
            dtype=torch.float32,
        )

        if image.ndim != 3:
            raise ValueError(
                "Expected a frame with shape [C, H, W], "
                f"but received {tuple(image.shape)}."
            )

        if image.shape[0] != 3:
            raise ValueError(
                "Expected a three-channel frame with shape [3, H, W], "
                f"but received {tuple(image.shape)}."
            )

        return image

    @staticmethod
    def _prepare_mask(mask: Any) -> torch.Tensor:
        mask_tensor = torch.as_tensor(
            mask,
            dtype=torch.float32,
        )

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        if mask_tensor.ndim != 3:
            raise ValueError(
                "Expected a mask with shape [1, H, W], "
                f"but received {tuple(mask_tensor.shape)}."
            )

        if mask_tensor.shape[0] != 1:
            raise ValueError(
                "Expected a one-channel mask with shape [1, H, W], "
                f"but received {tuple(mask_tensor.shape)}."
            )

        return (mask_tensor > 0.5).float()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        _, targets = self.dataset[index]

        (
            large_frame,
            small_frame,
            large_mask,
            small_mask,
            ef,
        ) = targets

        ef_tensor = torch.as_tensor(
            ef,
            dtype=torch.float32,
        )

        label = torch.tensor(
            int(float(ef_tensor) < self.ef_threshold),
            dtype=torch.long,
        )

        return {
            "large_image": self._prepare_frame(large_frame),
            "small_image": self._prepare_frame(small_frame),
            "large_mask": self._prepare_mask(large_mask),
            "small_mask": self._prepare_mask(small_mask),
            "label": label,
            "ef": ef_tensor,
        }


def _as_float(value: Any) -> float:
    """Convert a scalar tensor or numeric value to float."""
    if torch.is_tensor(value):
        return float(value.detach().item())

    return float(value)


def _find_loss_component(
    loss_dict: Mapping[str, Any],
    possible_names: Sequence[str],
) -> float:
    """
    Retrieve a loss component while tolerating minor key-name differences.

    Returns NaN when none of the possible names is present.
    """
    for name in possible_names:
        if name in loss_dict:
            return _as_float(loss_dict[name])

    return float("nan")


def _mean_or_nan(total: float, count: int) -> float:
    if count == 0:
        return float("nan")

    return total / count


def run_multitask_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Run one training or validation epoch.

    Passing optimizer=None performs evaluation without gradient updates.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_segmentation_loss = 0.0
    total_classification_loss = 0.0

    segmentation_loss_batches = 0
    classification_loss_batches = 0
    number_of_batches = 0
    number_of_patients = 0

    overall_segmentation = BinaryMetricAccumulator(
        threshold=SEGMENTATION_THRESHOLD,
        max_auc_samples=SEGMENTATION_AUC_SAMPLES,
        seed=0,
    )

    large_segmentation = BinaryMetricAccumulator(
        threshold=SEGMENTATION_THRESHOLD,
        max_auc_samples=FRAME_AUC_SAMPLES,
        seed=1,
    )

    small_segmentation = BinaryMetricAccumulator(
        threshold=SEGMENTATION_THRESHOLD,
        max_auc_samples=FRAME_AUC_SAMPLES,
        seed=2,
    )

    patient_classification = BinaryMetricAccumulator(
        threshold=0.5,
        max_auc_samples=CLASSIFICATION_AUC_SAMPLES,
        seed=3,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.time()

    with torch.set_grad_enabled(is_training):
        for batch in dataloader:
            large_images = batch["large_image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            small_images = batch["small_image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            large_masks = batch["large_mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            small_masks = batch["small_mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            current_batch_size = labels.shape[0]

            # One forward pass processes both cardiac phases.
            images = torch.cat(
                [large_images, small_images],
                dim=0,
            )

            masks = torch.cat(
                [large_masks, small_masks],
                dim=0,
            )

            # Each frame inherits its patient's EF class during optimization.
            repeated_labels = torch.cat(
                [labels, labels],
                dim=0,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            if "segmentation" not in outputs:
                raise KeyError(
                    "The model output must contain a 'segmentation' tensor."
                )

            if "classification" not in outputs:
                raise KeyError(
                    "The model output must contain a 'classification' tensor."
                )

            loss, loss_dict = criterion(
                outputs,
                masks,
                repeated_labels,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            segmentation_logits = outputs["segmentation"]

            if segmentation_logits.ndim == 3:
                segmentation_logits = segmentation_logits.unsqueeze(1)

            if segmentation_logits.shape != masks.shape:
                raise ValueError(
                    "Segmentation output and mask shapes do not match: "
                    f"{tuple(segmentation_logits.shape)} versus "
                    f"{tuple(masks.shape)}."
                )

            segmentation_probabilities = torch.sigmoid(
                segmentation_logits
            )

            large_probabilities = segmentation_probabilities[
                :current_batch_size
            ]

            small_probabilities = segmentation_probabilities[
                current_batch_size:
            ]

            overall_segmentation.update(
                probabilities=segmentation_probabilities,
                targets=masks,
            )

            large_segmentation.update(
                probabilities=large_probabilities,
                targets=large_masks,
            )

            small_segmentation.update(
                probabilities=small_probabilities,
                targets=small_masks,
            )

            classification_logits = outputs["classification"]

            if (
                classification_logits.ndim != 2
                or classification_logits.shape[1] != NUM_CLASSES
            ):
                raise ValueError(
                    "Expected classification logits with shape [2B, 2], "
                    f"but received {tuple(classification_logits.shape)}."
                )

            frame_positive_probabilities = torch.softmax(
                classification_logits,
                dim=1,
            )[:, 1]

            large_class_probabilities = (
                frame_positive_probabilities[:current_batch_size]
            )

            small_class_probabilities = (
                frame_positive_probabilities[current_batch_size:]
            )

            # Evaluate classification once per patient rather than counting
            # the large and small frames as two independent patients.
            patient_positive_probabilities = (
                large_class_probabilities
                + small_class_probabilities
            ) / 2.0

            patient_classification.update(
                probabilities=patient_positive_probabilities,
                targets=labels,
            )

            total_loss += _as_float(loss)

            segmentation_loss = _find_loss_component(
                loss_dict,
                (
                    "segmentation_loss",
                    "seg_loss",
                    "segmentation",
                    "loss_segmentation",
                ),
            )

            classification_loss = _find_loss_component(
                loss_dict,
                (
                    "classification_loss",
                    "class_loss",
                    "cls_loss",
                    "classification",
                    "loss_classification",
                ),
            )

            if math.isfinite(segmentation_loss):
                total_segmentation_loss += segmentation_loss
                segmentation_loss_batches += 1

            if math.isfinite(classification_loss):
                total_classification_loss += classification_loss
                classification_loss_batches += 1

            number_of_batches += 1
            number_of_patients += current_batch_size

    if number_of_batches == 0:
        raise RuntimeError("The DataLoader contains no batches.")

    elapsed_seconds = time.time() - started_at

    if device.type == "cuda":
        peak_gpu_memory_allocated = int(
            torch.cuda.max_memory_allocated(device)
        )
        peak_gpu_memory_reserved = int(
            torch.cuda.max_memory_reserved(device)
        )
    else:
        peak_gpu_memory_allocated = 0
        peak_gpu_memory_reserved = 0

    return {
        "losses": {
            "total_loss": total_loss / number_of_batches,
            "segmentation_loss": _mean_or_nan(
                total_segmentation_loss,
                segmentation_loss_batches,
            ),
            "classification_loss": _mean_or_nan(
                total_classification_loss,
                classification_loss_batches,
            ),
        },
        "overall": overall_segmentation.compute(),
        "large": large_segmentation.compute(),
        "small": small_segmentation.compute(),
        "classification": patient_classification.compute(),
        "metadata": {
            "elapsed_seconds": elapsed_seconds,
            "number_of_patients": number_of_patients,
            "number_of_frames": number_of_patients * 2,
            "peak_gpu_memory_allocated": peak_gpu_memory_allocated,
            "peak_gpu_memory_reserved": peak_gpu_memory_reserved,
            "batch_size": dataloader.batch_size,
        },
    }


def _base_history_row(
    epoch: int,
    phase: str,
    frame_type: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    losses = result["losses"]
    metadata = result["metadata"]

    return {
        "model": MODEL_NAME,
        "epoch": epoch,
        "phase": phase,
        "frame_type": frame_type,
        "total_loss": losses["total_loss"],
        "segmentation_loss": losses["segmentation_loss"],
        "classification_loss": losses["classification_loss"],
        "elapsed_seconds": metadata["elapsed_seconds"],
        "number_of_patients": metadata["number_of_patients"],
        "number_of_frames": metadata["number_of_frames"],
        "peak_gpu_memory_allocated": metadata[
            "peak_gpu_memory_allocated"
        ],
        "peak_gpu_memory_reserved": metadata[
            "peak_gpu_memory_reserved"
        ],
        "batch_size": metadata["batch_size"],
    }


def segmentation_history_row(
    epoch: int,
    phase: str,
    frame_type: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = result[frame_type]

    row = _base_history_row(
        epoch=epoch,
        phase=phase,
        frame_type=frame_type,
        result=result,
    )

    row.update(
        {
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "specificity": metrics["specificity"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }
    )

    return row


def classification_history_row(
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = result["classification"]

    row = _base_history_row(
        epoch=epoch,
        phase=phase,
        frame_type="ef_classification",
        result=result,
    )

    row.update(
        {
            "dice": "",
            "iou": "",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "specificity": metrics["specificity"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }
    )

    return row


def result_rows(
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    return [
        segmentation_history_row(
            epoch=epoch,
            phase=phase,
            frame_type="overall",
            result=result,
        ),
        segmentation_history_row(
            epoch=epoch,
            phase=phase,
            frame_type="large",
            result=result,
        ),
        segmentation_history_row(
            epoch=epoch,
            phase=phase,
            frame_type="small",
            result=result,
        ),
        classification_history_row(
            epoch=epoch,
            phase=phase,
            result=result,
        ),
    ]


def initialize_history_file(path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_COLUMNS,
        )
        writer.writeheader()


def append_result_rows(
    path: Path,
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
) -> None:
    with path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_COLUMNS,
        )
        writer.writerows(
            result_rows(
                epoch=epoch,
                phase=phase,
                result=result,
            )
        )


def write_validation_metrics(
    path: Path,
    epoch: int,
    result: Mapping[str, Any],
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(
            result_rows(
                epoch=epoch,
                phase="val",
                result=result,
            )
        )


def validate_dataset_files(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {data_root}"
        )

    required_paths = [
        FILE_LIST_PATH,
        VOLUME_TRACINGS_PATH,
        VIDEOS_DIR,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                "Required EchoNet dataset item was not found: "
                f"{required_path}"
            )


def validate_sample_batch(
    sample_batch: Mapping[str, torch.Tensor],
) -> None:
    expected_shapes = {
        "large_image": (4, 3),
        "small_image": (4, 3),
        "large_mask": (4, 1),
        "small_mask": (4, 1),
    }

    for key, (expected_ndim, expected_channels) in expected_shapes.items():
        tensor = sample_batch[key]

        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"{key} must have {expected_ndim} dimensions, "
                f"but received shape {tuple(tensor.shape)}."
            )

        if tensor.shape[1] != expected_channels:
            raise ValueError(
                f"{key} must have {expected_channels} channel(s), "
                f"but received shape {tuple(tensor.shape)}."
            )

    if sample_batch["label"].ndim != 1:
        raise ValueError(
            "Labels must have batch shape [B], "
            f"but received {tuple(sample_batch['label'].shape)}."
        )


def print_epoch_summary(
    epoch: int,
    train_result: Mapping[str, Any],
    val_result: Mapping[str, Any],
) -> None:
    train_overall = train_result["overall"]
    val_overall = val_result["overall"]
    val_classification = val_result["classification"]

    print(f"\nEpoch {epoch}/{EPOCHS}")

    print(
        "Train segmentation: "
        f"loss={train_result['losses']['total_loss']:.6f}, "
        f"dice={train_overall['dice']:.6f}, "
        f"iou={train_overall['iou']:.6f}"
    )

    print(
        "Validation segmentation: "
        f"loss={val_result['losses']['total_loss']:.6f}, "
        f"dice={val_overall['dice']:.6f}, "
        f"iou={val_overall['iou']:.6f}, "
        f"precision={val_overall['precision']:.6f}, "
        f"recall={val_overall['recall']:.6f}, "
        f"specificity={val_overall['specificity']:.6f}, "
        f"roc_auc={val_overall['roc_auc']:.6f}"
    )

    print(
        "Validation EF classification: "
        f"accuracy={val_classification['accuracy']:.6f}, "
        f"precision={val_classification['precision']:.6f}, "
        f"recall={val_classification['recall']:.6f}, "
        f"f1={val_classification['f1']:.6f}, "
        f"specificity={val_classification['specificity']:.6f}, "
        f"roc_auc={val_classification['roc_auc']:.6f}"
    )


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    data_root = Path(DATA_DIR)
    validate_dataset_files(data_root)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialize_history_file(HISTORY_PATH)

    model = MultitaskDeepLabV3(
        num_classes=NUM_CLASSES,
        pretrained=True,
    ).to(device)

    criterion = MultitaskLoss(
        segmentation_loss="bce_dice",
        seg_weight=1.0,
        class_weight=0.3,
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_dataset = EchoMultitaskDataset(
        root=str(data_root),
        split="train",
        ef_threshold=EF_THRESHOLD,
    )

    val_dataset = EchoMultitaskDataset(
        root=str(data_root),
        split="val",
        ef_threshold=EF_THRESHOLD,
    )

    print(f"Training patients:   {len(train_dataset):,}")
    print(f"Validation patients: {len(val_dataset):,}")

    pin_memory = device.type == "cuda"
    persistent_workers = NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    sample_batch = next(iter(train_loader))

    print(
        "Large image batch:",
        tuple(sample_batch["large_image"].shape),
    )
    print(
        "Small image batch:",
        tuple(sample_batch["small_image"].shape),
    )
    print(
        "Large mask batch:",
        tuple(sample_batch["large_mask"].shape),
    )
    print(
        "Small mask batch:",
        tuple(sample_batch["small_mask"].shape),
    )
    print(
        "Label batch:",
        tuple(sample_batch["label"].shape),
    )
    print(
        "Example EF values:",
        sample_batch["ef"][:4],
    )

    validate_sample_batch(sample_batch)

    best_dice = -float("inf")
    best_epoch: Optional[int] = None

    for epoch in range(1, EPOCHS + 1):
        train_result = run_multitask_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_result = run_multitask_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        append_result_rows(
            path=HISTORY_PATH,
            epoch=epoch,
            phase="train",
            result=train_result,
        )

        append_result_rows(
            path=HISTORY_PATH,
            epoch=epoch,
            phase="val",
            result=val_result,
        )

        print_epoch_summary(
            epoch=epoch,
            train_result=train_result,
            val_result=val_result,
        )

        current_validation_dice = val_result["overall"]["dice"]

        checkpoint = {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "validation_metrics": val_result,
            "ef_threshold": EF_THRESHOLD,
            "segmentation_threshold": SEGMENTATION_THRESHOLD,
            "num_classes": NUM_CLASSES,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

        if current_validation_dice > best_dice:
            best_dice = current_validation_dice
            best_epoch = epoch

            checkpoint["best_dice"] = best_dice

            torch.save(
                checkpoint,
                BEST_CHECKPOINT_PATH,
            )

            write_validation_metrics(
                path=VALIDATION_METRICS_PATH,
                epoch=epoch,
                result=val_result,
            )

            print(
                "Saved new best checkpoint: "
                f"epoch={epoch}, "
                f"validation Dice={best_dice:.6f}"
            )

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Dice: {best_dice:.6f}")
    print(f"Latest checkpoint: {CHECKPOINT_PATH}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print(f"Best validation metrics: {VALIDATION_METRICS_PATH}")


if __name__ == "__main__":
    main()