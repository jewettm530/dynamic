#!/usr/bin/env python3
"""Train corrected Stage 1 B1: video EF-only baseline.

B1 receives a multi-frame video clip and produces one EF value after temporal
feature aggregation.  This script never requests or opens VolumeTracings.csv.
Test data are not constructed; the best checkpoint is selected only by lowest
validation EF MAE.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch

from echonet.datasets.stage1_video import Stage1VideoDataset
from echonet.modeling.stage1_video_multitask import Stage1VideoMultitaskModel
from echonet.utils.reproducibility import seed_everything
from echonet.utils.stage1_corrected import (
    assert_expected_count,
    corrected_config_common,
    make_loader,
    write_csv,
    write_json,
)
from echonet.utils.stage1_metrics import ef_fraction_to_percent, regression_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--period", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-step-period", type=int, default=15)
    p.add_argument("--segmentation-decoder-width", type=int, default=128)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--non-deterministic", action="store_true")
    p.add_argument("--skip-count-check", action="store_true")
    return p.parse_args()


def run_epoch(
    model,
    loader,
    device,
    optimizer: Optional[torch.optim.Optimizer],
):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    n = 0
    targets: List[float] = []
    predictions: List[float] = []
    filenames: List[str] = []
    started = time.time()

    with torch.set_grad_enabled(training):
        for batch in loader:
            video = batch["video"].to(device, dtype=torch.float32, non_blocking=True)
            ef = batch["ef"].to(device, dtype=torch.float32, non_blocking=True).reshape(-1)

            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model.forward_ef(video).reshape(-1)
            loss = criterion(pred, ef)
            if training:
                loss.backward()
                optimizer.step()

            b = ef.numel()
            total_loss += float(loss.detach().cpu()) * b
            n += b
            targets.extend(ef_fraction_to_percent(ef.detach().cpu().numpy()).tolist())
            predictions.extend(ef_fraction_to_percent(pred.detach().cpu().numpy()).tolist())
            filenames.extend(list(batch["filename"]))

    metrics = regression_metrics(targets, predictions)
    metrics.update(
        {
            "raw_ef_loss": total_loss / max(n, 1),
            "n_videos": n,
            "elapsed_seconds": time.time() - started,
        }
    )
    rows = [
        {
            "filename": f,
            "ef_target_percent": y,
            "ef_prediction_percent": p,
        }
        for f, y, p in zip(filenames, targets, predictions)
    ]
    return metrics, rows


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=not args.non_deterministic)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Stage1VideoDataset(
        args.data_root,
        "train",
        frames=args.frames,
        period=args.period,
        clip_sampling="random",
        include_segmentation=False,
        include_video=True,
    )
    val_ds = Stage1VideoDataset(
        args.data_root,
        "val",
        frames=args.frames,
        period=args.period,
        clip_sampling="center",
        include_segmentation=False,
        include_video=True,
    )
    assert_expected_count(len(train_ds), "train", False, args.skip_count_check)
    assert_expected_count(len(val_ds), "val", False, args.skip_count_check)

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        device=device,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
        device=device,
    )

    model = Stage1VideoMultitaskModel(
        pretrained=not args.no_pretrained,
        segmentation_decoder_width=args.segmentation_decoder_width,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_period
    )

    config = corrected_config_common(args, model_id="B1")
    config.update(
        {
            "task": "video EF-only baseline",
            "dataset": {
                "train_videos": len(train_ds),
                "validation_videos": len(val_ds),
                "volume_tracings_required": False,
            },
            "loss": "MSELoss(mean) on EF fraction",
            "checkpoint_rule": "lowest validation EF MAE",
            "test_used_during_training": False,
        }
    )
    write_json(output / "run_config.json", config)

    history = []
    best_mae = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, optimizer)
        val_metrics, val_predictions = run_epoch(model, val_loader, device, None)
        scheduler.step()

        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            row = {"epoch": epoch, "phase": phase, "seed": args.seed}
            row.update(metrics)
            history.append(row)
        write_csv(output / "training_history.csv", history)

        checkpoint = {
            "model_id": "B1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(checkpoint, output / "checkpoint.pt")
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(checkpoint, output / "best.pt")
            write_json(output / "best_validation_metrics.json", val_metrics)
            write_csv(output / "best_validation_predictions.csv", val_predictions)

        print(
            f"Epoch {epoch:03d} | val MAE={val_metrics['mae']:.3f} "
            f"RMSE={val_metrics['rmse']:.3f} R2={val_metrics['r2']:.3f} "
            f"r={val_metrics['pearson_r']:.3f}"
        )

    write_json(
        output / "run_summary.json",
        {
            "model_id": "B1",
            "best_epoch": best_epoch,
            "best_validation_mae": best_mae,
            "best_checkpoint": str(output / "best.pt"),
        },
    )


if __name__ == "__main__":
    main()
