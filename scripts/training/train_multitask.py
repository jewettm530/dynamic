#!/usr/bin/env python3
"""Train the Stage 1 naive multi-task EchoNet-Dynamic baseline.

Tasks
-----
* EF: continuous regression trained on a 0-1 scale and reported on 0-100.
* LV segmentation: binary masks on labeled ED (Large) and ES (Small) frames.

Checkpoint rule
---------------
The single best checkpoint is selected ONLY by lowest validation EF MAE.
All segmentation metrics reported for that run come from the same checkpoint.
The test set is intentionally not constructed or evaluated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from echonet.datasets.echo import Echo
from echonet.losses.multitask_loss import MultitaskLoss
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import (
    dice_score,
    ef_fraction_to_percent,
    hd95_pixels,
    regression_metrics,
    summarize_segmentation,
)


class EchoMultitaskDataset(Dataset):
    """Return paired ED/ES labeled frames plus one continuous EF target."""

    def __init__(self, root: str, split: str, mean=0.0, std=1.0):
        self.dataset = Echo(
            root=root,
            split=split,
            target_type=[
                "Filename",
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
            pad=None,
            noise=None,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        _, targets = self.dataset[index]
        filename, ed_frame, es_frame, ed_mask, es_mask, ef = targets

        ed_frame = torch.as_tensor(ed_frame, dtype=torch.float32)
        es_frame = torch.as_tensor(es_frame, dtype=torch.float32)
        ed_mask = torch.as_tensor(ed_mask, dtype=torch.float32).unsqueeze(0)
        es_mask = torch.as_tensor(es_mask, dtype=torch.float32).unsqueeze(0)

        return {
            "filename": filename,
            "ed_image": ed_frame,
            "es_image": es_frame,
            "ed_mask": (ed_mask > 0.5).float(),
            "es_mask": (es_mask > 0.5).float(),
            "ef": torch.tensor(float(ef) / 100.0, dtype=torch.float32),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="EchoNet dataset root")
    p.add_argument("--output", required=True, help="Run output directory")
    p.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    p.add_argument("--ef-weight", type=float, required=True)
    p.add_argument("--seg-weight", type=float, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--regression-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--non-deterministic", action="store_true")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def device_metadata() -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def make_loader(dataset, batch_size, num_workers, shuffle, seed, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        drop_last=False,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer: Optional[torch.optim.Optimizer],
    compute_hd95: bool,
):
    training = optimizer is not None
    model.train(training)

    sums = {
        "raw_ef_loss": 0.0,
        "raw_seg_loss": 0.0,
        "weighted_ef_loss": 0.0,
        "weighted_seg_loss": 0.0,
        "total_loss": 0.0,
    }
    loss_weight = 0

    ef_targets: List[float] = []
    ef_predictions: List[float] = []
    filenames: List[str] = []
    ed_dice: List[float] = []
    es_dice: List[float] = []
    ed_hd95: List[float] = []
    es_hd95: List[float] = []

    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.set_grad_enabled(training):
        for batch in loader:
            ed_images = batch["ed_image"].to(device, non_blocking=True)
            es_images = batch["es_image"].to(device, non_blocking=True)
            ed_masks = batch["ed_mask"].to(device, non_blocking=True)
            es_masks = batch["es_mask"].to(device, non_blocking=True)
            ef = batch["ef"].to(device, non_blocking=True)
            b = ef.shape[0]

            images = torch.cat([ed_images, es_images], dim=0)
            masks = torch.cat([ed_masks, es_masks], dim=0)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            frame_ef = outputs["ef"].reshape(-1)
            video_ef = (frame_ef[:b] + frame_ef[b:]) / 2.0

            loss, components = criterion(
                segmentation_logits=outputs["segmentation"],
                segmentation_targets=masks,
                ef_predictions=video_ef,
                ef_targets=ef,
            )

            if training:
                loss.backward()
                optimizer.step()

            for key in sums:
                sums[key] += components[key] * b
            loss_weight += b

            probs = torch.sigmoid(outputs["segmentation"]).detach().cpu().numpy()
            truth = masks.detach().cpu().numpy()
            ed_probs, es_probs = probs[:b, 0], probs[b:, 0]
            ed_truth, es_truth = truth[:b, 0], truth[b:, 0]

            for p, t in zip(ed_probs, ed_truth):
                pred = p >= 0.5
                true = t >= 0.5
                ed_dice.append(dice_score(pred, true))
                if compute_hd95:
                    ed_hd95.append(hd95_pixels(pred, true))
            for p, t in zip(es_probs, es_truth):
                pred = p >= 0.5
                true = t >= 0.5
                es_dice.append(dice_score(pred, true))
                if compute_hd95:
                    es_hd95.append(hd95_pixels(pred, true))

            ef_targets.extend(
                ef_fraction_to_percent(ef.detach().cpu().numpy()).reshape(-1).tolist()
            )
            ef_predictions.extend(
                ef_fraction_to_percent(video_ef.detach().cpu().numpy()).reshape(-1).tolist()
            )
            filenames.extend(list(batch["filename"]))

    reg = regression_metrics(ef_targets, ef_predictions)

    if compute_hd95:
        seg = summarize_segmentation(ed_dice, es_dice, ed_hd95, es_hd95)
    else:
        dice_ed = float(np.mean(ed_dice))
        dice_es = float(np.mean(es_dice))
        seg = {
            "dice_ed": dice_ed,
            "dice_es": dice_es,
            "mean_dice": (dice_ed + dice_es) / 2.0,
            "hd95_ed": float("nan"),
            "hd95_es": float("nan"),
            "mean_hd95": float("nan"),
        }

    metrics = {key: sums[key] / max(loss_weight, 1) for key in sums}
    metrics.update(reg)
    metrics.update(seg)
    metrics["n_videos"] = len(ef_targets)
    metrics["elapsed_seconds"] = time.time() - started
    metrics["peak_gpu_memory_allocated"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    predictions = [
        {
            "filename": f,
            "ef_target_percent": y,
            "ef_prediction_percent": yhat,
        }
        for f, y, yhat in zip(filenames, ef_targets, ef_predictions)
    ]
    return metrics, predictions


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=not args.non_deterministic)

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    for required in ["FileList.csv", "VolumeTracings.csv", "Videos"]:
        if not (data_root / required).exists():
            raise FileNotFoundError(data_root / required)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = vars(args).copy()
    config.update(
        {
            "task": "multitask",
            "ef_training_target_scale": "0-1 fraction",
            "ef_evaluation_scale": "0-100 percentage points",
            "ef_percent_conversion": "prediction * 100",
            "ef_loss": "MSELoss(mean)",
            "segmentation_loss": "BCEWithLogitsLoss(mean)",
            "checkpoint_rule": "lowest validation EF MAE",
            "spatial_augmentation": "none",
            "git_commit": git_commit(),
            "command": " ".join(sys.argv),
            "environment": device_metadata(),
        }
    )
    with (output / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    train_ds = EchoMultitaskDataset(str(data_root), "train")
    val_ds = EchoMultitaskDataset(str(data_root), "val")
    train_loader = make_loader(
        train_ds, args.batch_size, args.num_workers, True, args.seed, device
    )
    val_loader = make_loader(
        val_ds, args.batch_size, args.num_workers, False, args.seed, device
    )

    model = MultitaskDeepLabV3(
        pretrained=not args.no_pretrained,
        regression_hidden_dim=args.regression_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = MultitaskLoss(args.ef_weight, args.seg_weight)
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: List[dict] = []
    best_mae = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, compute_hd95=False
        )
        val_metrics, val_predictions = run_epoch(
            model, val_loader, criterion, device, None, compute_hd95=True
        )

        for phase, metrics in [("train", train_metrics), ("val", val_metrics)]:
            row = {"epoch": epoch, "phase": phase, "seed": args.seed}
            row.update(metrics)
            history.append(row)
        write_csv(output / "training_history.csv", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(checkpoint, output / "checkpoint.pt")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(checkpoint, output / "best.pt")
            write_csv(output / "best_validation_predictions.csv", val_predictions)
            with (output / "best_validation_metrics.json").open("w") as f:
                json.dump(val_metrics, f, indent=2)

        print(
            f"Epoch {epoch:03d} | "
            f"train total={train_metrics['total_loss']:.4f} | "
            f"val MAE={val_metrics['mae']:.3f} RMSE={val_metrics['rmse']:.3f} "
            f"R2={val_metrics['r2']:.3f} r={val_metrics['pearson_r']:.3f} | "
            f"Dice={val_metrics['mean_dice']:.4f} HD95={val_metrics['mean_hd95']:.3f}"
        )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "best_checkpoint": str(output / "best.pt"),
    }
    with (output / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
