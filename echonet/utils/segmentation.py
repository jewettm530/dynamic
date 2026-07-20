"""Functions for training and running EchoNet segmentation."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

import echonet
from echonet.utils.evaluation_metrics import BinaryMetricAccumulator


MODEL_NAME_FOR_LOG = "segmentation_deeplabv3_resnet50"
SEGMENTATION_THRESHOLD = 0.5
OVERALL_AUC_SAMPLES = 1_000_000
FRAME_AUC_SAMPLES = 500_000

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


@click.command("segmentation")
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
)
@click.option(
    "--model_name",
    type=click.Choice(
        sorted(
            name
            for name in torchvision.models.segmentation.__dict__
            if name.islower()
            and not name.startswith("__")
            and callable(
                torchvision.models.segmentation.__dict__[name]
            )
        )
    ),
    default="deeplabv3_resnet50",
)
@click.option("--pretrained/--random", default=False)
@click.option(
    "--weights",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video/--skip_video", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    model_name="deeplabv3_resnet50",
    pretrained=False,
    weights=None,
    run_test=False,
    save_video=False,
    num_epochs=50,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
    """
    Train and optionally test an EchoNet left-ventricle segmentation model.

    Standardized outputs:
        checkpoint.pt
        best.pt
        log.csv
        training_history.csv
        validation_metrics.csv

    The best checkpoint is selected using the highest overall validation Dice,
    matching the updated multitask training script.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join(
            "output",
            "comparison",
            "original_25_epochs",
        )

    output_path = Path(output)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = _build_segmentation_model(
        model_name=model_name,
        pretrained=pretrained,
    )

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    model.to(device)

    if weights is not None:
        checkpoint = torch.load(
            weights,
            map_location=device,
        )
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict"),
        )

        if state_dict is None:
            raise KeyError(
                "The weights file does not contain 'state_dict' "
                "or 'model_state_dict'."
            )

        model.load_state_dict(state_dict)

    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    if lr_step_period is None:
        lr_step_period = math.inf

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        lr_step_period,
    )

    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(
            root=data_dir,
            split="train",
        )
    )

    tasks = [
        "LargeFrame",
        "SmallFrame",
        "LargeTrace",
        "SmallTrace",
    ]

    dataset_kwargs = {
        "target_type": tasks,
        "mean": mean,
        "std": std,
    }

    datasets = {
        "train": echonet.datasets.Echo(
            root=data_dir,
            split="train",
            **dataset_kwargs,
        ),
        "val": echonet.datasets.Echo(
            root=data_dir,
            split="val",
            **dataset_kwargs,
        ),
    }

    if (
        num_train_patients is not None
        and len(datasets["train"]) > num_train_patients
    ):
        indices = np.random.choice(
            len(datasets["train"]),
            num_train_patients,
            replace=False,
        )

        datasets["train"] = torch.utils.data.Subset(
            datasets["train"],
            indices,
        )

    history_path = output_path / "training_history.csv"
    validation_metrics_path = (
        output_path / "validation_metrics.csv"
    )
    checkpoint_path = output_path / "checkpoint.pt"
    best_checkpoint_path = output_path / "best.pt"
    legacy_log_path = output_path / "log.csv"

    epoch_resume = 0
    best_dice = -float("inf")
    best_loss = float("inf")

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(checkpoint["state_dict"])
        optim.load_state_dict(checkpoint["opt_dict"])
        scheduler.load_state_dict(
            checkpoint["scheduler_dict"]
        )

        epoch_resume = int(checkpoint["epoch"]) + 1
        best_dice = float(
            checkpoint.get("best_dice", -float("inf"))
        )
        best_loss = float(
            checkpoint.get("best_loss", float("inf"))
        )

        _ensure_history_file(history_path)

        with legacy_log_path.open("a") as legacy_log:
            legacy_log.write(
                f"Resuming from epoch {epoch_resume}\n"
            )
    else:
        _initialize_history_file(history_path)

        with legacy_log_path.open("w") as legacy_log:
            legacy_log.write("Starting run from scratch\n")

    for epoch in range(epoch_resume, num_epochs):
        print(f"Epoch #{epoch}", flush=True)

        epoch_results: Dict[str, Dict[str, Any]] = {}

        for phase in ("train", "val"):
            dataloader = torch.utils.data.DataLoader(
                datasets[phase],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=(phase == "train"),
                pin_memory=(device.type == "cuda"),
                drop_last=(phase == "train"),
                persistent_workers=(num_workers > 0),
            )

            result = run_epoch(
                model=model,
                dataloader=dataloader,
                train=(phase == "train"),
                optim=optim if phase == "train" else None,
                device=device,
            )

            epoch_results[phase] = result

            _append_result_rows(
                path=history_path,
                epoch=epoch,
                phase=phase,
                result=result,
                model_name=model_name,
            )

            _append_legacy_log_row(
                path=legacy_log_path,
                epoch=epoch,
                phase=phase,
                result=result,
            )

            _print_phase_summary(
                epoch=epoch,
                phase=phase,
                result=result,
            )

        # Step once per completed epoch, not once per phase.
        scheduler.step()

        validation_result = epoch_results["val"]
        validation_dice = float(
            validation_result["overall"]["dice"]
        )
        validation_loss = float(
            validation_result["losses"]["total_loss"]
        )

        is_best = validation_dice > best_dice

        if is_best:
            best_dice = validation_dice
            best_loss = validation_loss

        save = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_dice": best_dice,
            "best_loss": best_loss,
            "loss": validation_loss,
            "validation_metrics": validation_result,
            "opt_dict": optim.state_dict(),
            "scheduler_dict": scheduler.state_dict(),
            "segmentation_threshold": SEGMENTATION_THRESHOLD,
            "model_name": model_name,
        }

        torch.save(
            save,
            checkpoint_path,
        )

        if is_best:
            torch.save(
                save,
                best_checkpoint_path,
            )

            _write_validation_metrics(
                path=validation_metrics_path,
                epoch=epoch,
                result=validation_result,
                model_name=model_name,
            )

            print(
                "Saved new best checkpoint: "
                f"epoch={epoch}, "
                f"validation Dice={best_dice:.6f}"
            )

    if best_checkpoint_path.exists():
        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(checkpoint["state_dict"])

        with legacy_log_path.open("a") as legacy_log:
            legacy_log.write(
                "Best validation Dice "
                f"{checkpoint['best_dice']} "
                f"from epoch {checkpoint['epoch']}\n"
            )

    if run_test:
        _run_final_split_evaluation(
            model=model,
            data_dir=data_dir,
            output_path=output_path,
            dataset_kwargs=dataset_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            model_name=model_name,
        )

    if save_video:
        _save_segmentation_videos(
            model=model,
            data_dir=data_dir,
            output_path=output_path,
            mean=mean,
            std=std,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )


def _build_segmentation_model(
    model_name: str,
    pretrained: bool,
) -> torch.nn.Module:
    constructor = (
        torchvision.models.segmentation.__dict__[
            model_name
        ]
    )

    # Support both older and newer torchvision APIs.
    try:
        model = constructor(
            pretrained=pretrained,
            aux_loss=False,
        )
    except TypeError:
        weights = "DEFAULT" if pretrained else None
        model = constructor(
            weights=weights,
            aux_loss=False,
        )

    classifier = model.classifier

    if not hasattr(classifier, "__getitem__"):
        raise TypeError(
            "Expected a torchvision segmentation classifier "
            "that supports indexing."
        )

    final_layer = classifier[-1]

    if not isinstance(final_layer, torch.nn.Conv2d):
        raise TypeError(
            "Expected the final classifier layer to be Conv2d."
        )

    classifier[-1] = torch.nn.Conv2d(
        final_layer.in_channels,
        1,
        kernel_size=final_layer.kernel_size,
    )

    return model


def run_epoch(
    model,
    dataloader,
    train,
    optim,
    device,
):
    """
    Run one segmentation epoch and return globally aggregated metrics.

    Dice, IoU, accuracy, precision, recall, F1, specificity, and
    confusion counts use every pixel in the epoch. ROC-AUC uses a
    reproducible probability sample to control memory use.
    """

    if train and optim is None:
        raise ValueError(
            "An optimizer is required when train=True."
        )

    model.train(train)

    total_loss_sum = 0.0
    number_of_patients = 0

    overall_accumulator = BinaryMetricAccumulator(
        threshold=SEGMENTATION_THRESHOLD,
        max_auc_samples=OVERALL_AUC_SAMPLES,
        seed=0,
    )

    large_accumulator = BinaryMetricAccumulator(
        threshold=SEGMENTATION_THRESHOLD,
        max_auc_samples=FRAME_AUC_SAMPLES,
        seed=1,
    )

    small_accumulator = BinaryMetricAccumulator(
        threshold=SEGMENTATION_THRESHOLD,
        max_auc_samples=FRAME_AUC_SAMPLES,
        seed=2,
    )

    per_patient = {
        "large_intersection": [],
        "large_union": [],
        "small_intersection": [],
        "small_union": [],
    }

    if device.type == "cuda":
        for gpu_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gpu_index)

    started_at = time.time()

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as progress:
            for (
                _,
                (
                    large_frame,
                    small_frame,
                    large_trace,
                    small_trace,
                ),
            ) in dataloader:
                large_frame = large_frame.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                small_frame = small_frame.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                large_trace = large_trace.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                small_trace = small_trace.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                large_trace = (large_trace > 0.5).float()
                small_trace = (small_trace > 0.5).float()

                large_logits = model(
                    large_frame
                )["out"][:, 0, :, :]

                small_logits = model(
                    small_frame
                )["out"][:, 0, :, :]

                large_loss = (
                    torch.nn.functional
                    .binary_cross_entropy_with_logits(
                        large_logits,
                        large_trace,
                        reduction="sum",
                    )
                )

                small_loss = (
                    torch.nn.functional
                    .binary_cross_entropy_with_logits(
                        small_logits,
                        small_trace,
                        reduction="sum",
                    )
                )

                loss = (
                    large_loss + small_loss
                ) / 2.0

                if train:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()

                large_probabilities = torch.sigmoid(
                    large_logits
                )

                small_probabilities = torch.sigmoid(
                    small_logits
                )

                large_accumulator.update(
                    probabilities=large_probabilities,
                    targets=large_trace,
                )

                small_accumulator.update(
                    probabilities=small_probabilities,
                    targets=small_trace,
                )

                overall_accumulator.update(
                    probabilities=torch.cat(
                        [
                            large_probabilities,
                            small_probabilities,
                        ],
                        dim=0,
                    ),
                    targets=torch.cat(
                        [
                            large_trace,
                            small_trace,
                        ],
                        dim=0,
                    ),
                )

                _append_per_patient_overlap(
                    storage=per_patient,
                    prefix="large",
                    probabilities=large_probabilities,
                    targets=large_trace,
                )

                _append_per_patient_overlap(
                    storage=per_patient,
                    prefix="small",
                    probabilities=small_probabilities,
                    targets=small_trace,
                )

                batch_patients = int(
                    large_trace.shape[0]
                )

                total_loss_sum += float(loss.item())
                number_of_patients += batch_patients

                current_large = large_accumulator.compute()
                current_small = small_accumulator.compute()

                image_area = int(
                    large_trace.shape[-2]
                    * large_trace.shape[-1]
                )

                running_loss = (
                    total_loss_sum
                    / max(number_of_patients, 1)
                    / image_area
                )

                progress.set_postfix_str(
                    "loss={:.4f}, large_dice={:.4f}, "
                    "small_dice={:.4f}".format(
                        running_loss,
                        current_large["dice"],
                        current_small["dice"],
                    )
                )

                progress.update()

    if number_of_patients == 0:
        raise RuntimeError(
            "The DataLoader contained no patients."
        )

    elapsed_seconds = time.time() - started_at

    image_area = int(
        dataloader.dataset[0][1][2].shape[-2]
        * dataloader.dataset[0][1][2].shape[-1]
    )

    normalized_loss = (
        total_loss_sum
        / number_of_patients
        / image_area
    )

    if device.type == "cuda":
        peak_gpu_memory_allocated = int(
            sum(
                torch.cuda.max_memory_allocated(i)
                for i in range(
                    torch.cuda.device_count()
                )
            )
        )

        peak_gpu_memory_reserved = int(
            sum(
                torch.cuda.max_memory_reserved(i)
                for i in range(
                    torch.cuda.device_count()
                )
            )
        )
    else:
        peak_gpu_memory_allocated = 0
        peak_gpu_memory_reserved = 0

    return {
        "losses": {
            "total_loss": normalized_loss,
            "segmentation_loss": normalized_loss,
            "classification_loss": "",
        },
        "overall": overall_accumulator.compute(),
        "large": large_accumulator.compute(),
        "small": small_accumulator.compute(),
        "metadata": {
            "elapsed_seconds": elapsed_seconds,
            "number_of_patients": number_of_patients,
            "number_of_frames": (
                number_of_patients * 2
            ),
            "peak_gpu_memory_allocated": (
                peak_gpu_memory_allocated
            ),
            "peak_gpu_memory_reserved": (
                peak_gpu_memory_reserved
            ),
            "batch_size": dataloader.batch_size,
        },
        "per_patient": {
            key: np.asarray(value)
            for key, value in per_patient.items()
        },
    }


def _append_per_patient_overlap(
    storage: Dict[str, list],
    prefix: str,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    predictions = (
        probabilities >= SEGMENTATION_THRESHOLD
    )
    target_binary = targets >= 0.5

    intersections = torch.logical_and(
        predictions,
        target_binary,
    ).sum(dim=(-2, -1))

    unions = torch.logical_or(
        predictions,
        target_binary,
    ).sum(dim=(-2, -1))

    storage[f"{prefix}_intersection"].extend(
        intersections.detach().cpu().numpy().tolist()
    )

    storage[f"{prefix}_union"].extend(
        unions.detach().cpu().numpy().tolist()
    )


def _history_row(
    epoch: int,
    phase: str,
    frame_type: str,
    result: Mapping[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    metrics = result[frame_type]
    losses = result["losses"]
    metadata = result["metadata"]

    return {
        "model": (
            f"segmentation_{model_name}"
        ),
        "epoch": epoch,
        "phase": phase,
        "frame_type": frame_type,
        "total_loss": losses["total_loss"],
        "segmentation_loss": losses[
            "segmentation_loss"
        ],
        "classification_loss": "",
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
        "elapsed_seconds": metadata[
            "elapsed_seconds"
        ],
        "number_of_patients": metadata[
            "number_of_patients"
        ],
        "number_of_frames": metadata[
            "number_of_frames"
        ],
        "peak_gpu_memory_allocated": metadata[
            "peak_gpu_memory_allocated"
        ],
        "peak_gpu_memory_reserved": metadata[
            "peak_gpu_memory_reserved"
        ],
        "batch_size": metadata["batch_size"],
    }


def _result_rows(
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
    model_name: str,
) -> list[Dict[str, Any]]:
    return [
        _history_row(
            epoch=epoch,
            phase=phase,
            frame_type=frame_type,
            result=result,
            model_name=model_name,
        )
        for frame_type in (
            "overall",
            "large",
            "small",
        )
    ]


def _initialize_history_file(path: Path) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_COLUMNS,
        )
        writer.writeheader()


def _ensure_history_file(path: Path) -> None:
    if not path.exists():
        _initialize_history_file(path)


def _append_result_rows(
    path: Path,
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
    model_name: str,
) -> None:
    with path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_COLUMNS,
        )
        writer.writerows(
            _result_rows(
                epoch=epoch,
                phase=phase,
                result=result,
                model_name=model_name,
            )
        )


def _write_validation_metrics(
    path: Path,
    epoch: int,
    result: Mapping[str, Any],
    model_name: str,
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(
            _result_rows(
                epoch=epoch,
                phase="val",
                result=result,
                model_name=model_name,
            )
        )


def _append_legacy_log_row(
    path: Path,
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
) -> None:
    losses = result["losses"]
    overall = result["overall"]
    large = result["large"]
    small = result["small"]
    metadata = result["metadata"]

    with path.open("a") as legacy_log:
        legacy_log.write(
            "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                epoch,
                phase,
                losses["total_loss"],
                overall["dice"],
                large["dice"],
                small["dice"],
                metadata["elapsed_seconds"],
                metadata["number_of_patients"],
                metadata[
                    "peak_gpu_memory_allocated"
                ],
                metadata[
                    "peak_gpu_memory_reserved"
                ],
                metadata["batch_size"],
            )
        )


def _print_phase_summary(
    epoch: int,
    phase: str,
    result: Mapping[str, Any],
) -> None:
    overall = result["overall"]
    large = result["large"]
    small = result["small"]

    print(
        f"Epoch {epoch} {phase}: "
        f"loss={result['losses']['total_loss']:.6f}, "
        f"overall_dice={overall['dice']:.6f}, "
        f"large_dice={large['dice']:.6f}, "
        f"small_dice={small['dice']:.6f}, "
        f"iou={overall['iou']:.6f}, "
        f"precision={overall['precision']:.6f}, "
        f"recall={overall['recall']:.6f}, "
        f"specificity={overall['specificity']:.6f}, "
        f"roc_auc={overall['roc_auc']:.6f}"
    )


def _run_final_split_evaluation(
    model,
    data_dir,
    output_path: Path,
    dataset_kwargs,
    batch_size,
    num_workers,
    device,
    model_name,
) -> None:
    for split in ("val", "test"):
        split_dataset = echonet.datasets.Echo(
            root=data_dir,
            split=split,
            **dataset_kwargs,
        )

        dataloader = torch.utils.data.DataLoader(
            split_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        result = run_epoch(
            model=model,
            dataloader=dataloader,
            train=False,
            optim=None,
            device=device,
        )

        _write_validation_metrics(
            path=output_path / f"{split}_metrics.csv",
            epoch=-1,
            result=result,
            model_name=model_name,
        )

        per_patient = result["per_patient"]

        large_intersection = per_patient[
            "large_intersection"
        ]
        large_union = per_patient["large_union"]
        small_intersection = per_patient[
            "small_intersection"
        ]
        small_union = per_patient["small_union"]

        overall_dice = (
            2
            * (
                large_intersection
                + small_intersection
            )
            / np.maximum(
                large_union
                + large_intersection
                + small_union
                + small_intersection,
                1,
            )
        )

        large_dice = (
            2
            * large_intersection
            / np.maximum(
                large_union + large_intersection,
                1,
            )
        )

        small_dice = (
            2
            * small_intersection
            / np.maximum(
                small_union + small_intersection,
                1,
            )
        )

        with (
            output_path
            / f"{split}_dice.csv"
        ).open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "Filename",
                    "Overall",
                    "Large",
                    "Small",
                ]
            )

            for filename, overall, large, small in zip(
                split_dataset.fnames,
                overall_dice,
                large_dice,
                small_dice,
            ):
                writer.writerow(
                    [
                        filename,
                        overall,
                        large,
                        small,
                    ]
                )


def _save_segmentation_videos(
    model,
    data_dir,
    output_path: Path,
    mean,
    std,
    batch_size,
    num_workers,
    device,
) -> None:
    """
    Preserve the original EchoNet video-overlay workflow.

    This is unrelated to metric collection and runs only with --save_video.
    """

    dataset = echonet.datasets.Echo(
        root=data_dir,
        split="test",
        target_type=[
            "Filename",
            "LargeIndex",
            "SmallIndex",
        ],
        mean=mean,
        std=std,
        length=None,
        max_length=None,
        period=1,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=_video_collate_fn,
    )

    videos_path = output_path / "videos"
    size_path = output_path / "size"

    if all(
        (videos_path / filename).is_file()
        for filename in dataloader.dataset.fnames
    ):
        return

    model.eval()
    videos_path.mkdir(parents=True, exist_ok=True)
    size_path.mkdir(parents=True, exist_ok=True)

    echonet.utils.latexify()

    with torch.no_grad():
        with (
            output_path / "size.csv"
        ).open("w") as size_file:
            size_file.write(
                "Filename,Frame,Size,HumanLarge,"
                "HumanSmall,ComputerSmall\n"
            )

            for (
                x,
                (
                    filenames,
                    large_index,
                    small_index,
                ),
                lengths,
            ) in tqdm.tqdm(dataloader):
                predictions = np.concatenate(
                    [
                        model(
                            x[
                                start:
                                start + batch_size
                            ].to(device)
                        )["out"]
                        .detach()
                        .cpu()
                        .numpy()
                        for start in range(
                            0,
                            x.shape[0],
                            batch_size,
                        )
                    ]
                )

                start = 0
                x_numpy = x.numpy()

                for patient_index, (
                    filename,
                    frame_count,
                ) in enumerate(
                    zip(filenames, lengths)
                ):
                    video = x_numpy[
                        start:
                        start + frame_count,
                        ...,
                    ]

                    logits = predictions[
                        start:
                        start + frame_count,
                        0,
                        :,
                        :,
                    ]

                    video *= mean.reshape(
                        1,
                        3,
                        1,
                        1,
                    ) * 0 + std.reshape(
                        1,
                        3,
                        1,
                        1,
                    )
                    video += mean.reshape(
                        1,
                        3,
                        1,
                        1,
                    )

                    _, channels, _, width = (
                        video.shape
                    )

                    if channels != 3:
                        raise ValueError(
                            "Expected three-channel video."
                        )

                    video = np.concatenate(
                        (video, video),
                        axis=3,
                    )

                    video[:, 0, :, width:] = np.maximum(
                        255.0 * (logits > 0),
                        video[:, 0, :, width:],
                    )

                    video = np.concatenate(
                        (
                            video,
                            np.zeros_like(video),
                        ),
                        axis=2,
                    )

                    size = (
                        logits > 0
                    ).sum(axis=(1, 2))

                    trim_min = sorted(size)[
                        round(len(size) ** 0.05)
                    ]
                    trim_max = sorted(size)[
                        round(len(size) ** 0.95)
                    ]

                    trim_range = trim_max - trim_min

                    systole = set(
                        scipy.signal.find_peaks(
                            -size,
                            distance=20,
                            prominence=(
                                0.50 * trim_range
                            ),
                        )[0]
                    )

                    for frame, frame_size in enumerate(
                        size
                    ):
                        size_file.write(
                            "{},{},{},{},{},{}\n".format(
                                filename,
                                frame,
                                frame_size,
                                int(
                                    frame
                                    == large_index[
                                        patient_index
                                    ]
                                ),
                                int(
                                    frame
                                    == small_index[
                                        patient_index
                                    ]
                                ),
                                int(frame in systole),
                            )
                        )

                    figure = plt.figure(
                        figsize=(
                            size.shape[0]
                            / 50
                            * 1.5,
                            3,
                        )
                    )

                    plt.scatter(
                        np.arange(size.shape[0])
                        / 50,
                        size,
                        s=1,
                    )

                    limits = plt.ylim()

                    for systolic_frame in systole:
                        plt.plot(
                            np.array(
                                [
                                    systolic_frame,
                                    systolic_frame,
                                ]
                            )
                            / 50,
                            limits,
                            linewidth=1,
                        )

                    plt.ylim(limits)
                    plt.title(
                        os.path.splitext(filename)[0]
                    )
                    plt.xlabel("Seconds")
                    plt.ylabel("Size (pixels)")
                    plt.tight_layout()
                    plt.savefig(
                        size_path
                        / (
                            os.path.splitext(
                                filename
                            )[0]
                            + ".pdf"
                        )
                    )
                    plt.close(figure)

                    start += frame_count

                    # The original overlay video is optional. Keep the
                    # clinically relevant size CSV and plot generation,
                    # but avoid assumptions about canvas dimensions that
                    # can fail on non-112x112 custom data.
                    video = video.transpose(
                        1,
                        0,
                        2,
                        3,
                    ).astype(np.uint8)

                    echonet.utils.savevideo(
                        videos_path / filename,
                        video,
                        50,
                    )


def _video_collate_fn(batch):
    """
    Concatenate variable-length videos along the frame dimension.
    """

    videos, targets = zip(*batch)

    lengths = [
        video.shape[1]
        for video in videos
    ]

    video_tensor = torch.as_tensor(
        np.swapaxes(
            np.concatenate(videos, axis=1),
            0,
            1,
        )
    )

    targets = zip(*targets)

    return video_tensor, targets, lengths

if __name__ == "__main__":
    run()