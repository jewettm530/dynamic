#!/usr/bin/env python3
"""Train the Stage 1 EF-only continuous regression baseline.

Architecture and main hyperparameters follow the current EchoNet video baseline:
r2plus1d_18, 32 frames, period 2, SGD, LR 1e-4, momentum 0.9,
weight decay 1e-4, StepLR every 15 epochs. The Stage 1 checkpoint rule is
lowest validation MAE rather than lowest MSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from echonet.datasets.echo import Echo
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import regression_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--period", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-step-period", type=int, default=15)
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_model(pretrained: bool):
    if pretrained:
        try:
            weights = torchvision.models.video.R2Plus1D_18_Weights.DEFAULT
            model = torchvision.models.video.r2plus1d_18(weights=weights)
        except Exception:
            model = torchvision.models.video.r2plus1d_18(pretrained=True)
    else:
        try:
            model = torchvision.models.video.r2plus1d_18(weights=None)
        except TypeError:
            model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def make_loader(ds, args, shuffle, device):
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
        drop_last=False,
    )


def run_epoch(model, loader, device, optimizer: Optional[torch.optim.Optimizer]):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    n = 0
    ys: List[float] = []
    yhats: List[float] = []
    started = time.time()

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True).reshape(-1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x).reshape(-1)
            loss = criterion(pred, y)
            if training:
                loss.backward()
                optimizer.step()
            b = y.shape[0]
            total_loss += float(loss.detach().cpu().item()) * b
            n += b
            ys.extend(y.detach().cpu().numpy().tolist())
            yhats.extend(pred.detach().cpu().numpy().tolist())

    metrics = regression_metrics(ys, yhats)
    metrics["raw_ef_loss"] = total_loss / max(n, 1)
    metrics["n_videos"] = n
    metrics["elapsed_seconds"] = time.time() - started
    return metrics


def write_history(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=True)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        root=args.data_root,
        target_type="EF",
        mean=np.array([0.0, 0.0, 0.0]),
        std=np.array([1.0, 1.0, 1.0]),
        length=args.frames,
        period=args.period,
        clips=1,
        noise=None,
    )
    # Stage 1 explicitly uses no spatial augmentation for this clean baseline.
    train_ds = Echo(split="train", pad=None, **kwargs)
    val_ds = Echo(split="val", pad=None, **kwargs)
    train_loader = make_loader(train_ds, args, True, device)
    val_loader = make_loader(val_ds, args, False, device)

    model = build_model(not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate,
        momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_period
    )

    config = vars(args).copy()
    config.update({
        "task": "EF-only continuous regression",
        "ef_target_scale": "0-100 percentage points",
        "ef_loss": "MSELoss(mean)",
        "checkpoint_rule": "lowest validation EF MAE",
        "spatial_augmentation": "none",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
    })
    with (output / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_mae = float("inf"); best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, None)
        scheduler.step()
        for phase, m in [("train", train_metrics), ("val", val_metrics)]:
            row = {"epoch": epoch, "phase": phase, "seed": args.seed}; row.update(m); history.append(row)
        write_history(output / "training_history.csv", history)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(ckpt, output / "checkpoint.pt")
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]; best_epoch = epoch
            torch.save(ckpt, output / "best.pt")
            with (output / "best_validation_metrics.json").open("w") as f:
                json.dump(val_metrics, f, indent=2)
        print(
            f"Epoch {epoch:03d} | val MAE={val_metrics['mae']:.3f} "
            f"RMSE={val_metrics['rmse']:.3f} R2={val_metrics['r2']:.3f} "
            f"r={val_metrics['pearson_r']:.3f}"
        )

    summary = {"best_epoch": best_epoch, "best_validation_mae": best_mae}
    with (output / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
