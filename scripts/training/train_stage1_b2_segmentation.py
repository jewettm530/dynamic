#!/usr/bin/env python3
"""Train corrected Stage 1 B2: segmentation-only baseline.

B2 uses the same R(2+1)D encoder and segmentation decoder used by B3 on the
expert-labeled ED/ES frames.  Because this segmentation architecture differs
from the previous DeepLabV3 baseline, the professor's reuse rule requires B2
to be rerun for seeds 42, 2026, and 3407.

Checkpoint selection uses highest validation Mean Dice. Test data are not
constructed by this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
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
from echonet.utils.stage1_metrics import dice_score, hd95_pixels, summarize_segmentation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--frames", type=int, default=32, help="Recorded for B1/B3 matching; B2 segmentation uses labeled frames")
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


def run_epoch(model, loader, device, optimizer: Optional[torch.optim.Optimizer], compute_hd95: bool):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_sum = 0.0
    n_videos = 0
    ed_dice, es_dice, ed_hd95, es_hd95 = [], [], [], []
    metric_rows = []
    started = time.time()

    with torch.set_grad_enabled(training):
        for batch in loader:
            ed = batch["ed_image"].to(device, dtype=torch.float32, non_blocking=True)
            es = batch["es_image"].to(device, dtype=torch.float32, non_blocking=True)
            ed_mask = batch["ed_mask"].to(device, dtype=torch.float32, non_blocking=True)
            es_mask = batch["es_mask"].to(device, dtype=torch.float32, non_blocking=True)
            b = ed.shape[0]
            frames = torch.cat([ed, es], dim=0)
            masks = torch.cat([ed_mask, es_mask], dim=0)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model.forward_segmentation(frames)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

            loss_sum += float(loss.detach().cpu()) * b
            n_videos += b

            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            truth = masks.detach().cpu().numpy()[:, 0]
            names = list(batch["filename"])
            for i, filename in enumerate(names):
                row = {"filename": filename}
                for phase, p, t in (
                    ("ed", probs[i], truth[i]),
                    ("es", probs[b + i], truth[b + i]),
                ):
                    pred, true = p >= 0.5, t >= 0.5
                    d = dice_score(pred, true)
                    h = hd95_pixels(pred, true) if compute_hd95 else float("nan")
                    if phase == "ed":
                        ed_dice.append(d)
                        if compute_hd95:
                            ed_hd95.append(h)
                    else:
                        es_dice.append(d)
                        if compute_hd95:
                            es_hd95.append(h)
                    row[f"dice_{phase}"] = d
                    row[f"hd95_{phase}"] = h
                row["mean_dice"] = (row["dice_ed"] + row["dice_es"]) / 2.0
                row["mean_hd95"] = (
                    (row["hd95_ed"] + row["hd95_es"]) / 2.0
                    if compute_hd95
                    else float("nan")
                )
                metric_rows.append(row)

    if compute_hd95:
        metrics = summarize_segmentation(ed_dice, es_dice, ed_hd95, es_hd95)
    else:
        de, ds = float(np.mean(ed_dice)), float(np.mean(es_dice))
        metrics = {
            "dice_ed": de,
            "dice_es": ds,
            "mean_dice": (de + ds) / 2.0,
            "hd95_ed": float("nan"),
            "hd95_es": float("nan"),
            "mean_hd95": float("nan"),
        }
    metrics.update(
        {
            "raw_seg_loss": loss_sum / max(n_videos, 1),
            "n_videos": n_videos,
            "elapsed_seconds": time.time() - started,
        }
    )
    return metrics, metric_rows


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
        include_segmentation=True,
        require_segmentation=True,
        include_video=False,
    )
    val_ds = Stage1VideoDataset(
        args.data_root,
        "val",
        frames=args.frames,
        period=args.period,
        include_segmentation=True,
        require_segmentation=True,
        include_video=False,
    )
    assert_expected_count(len(train_ds), "train", True, args.skip_count_check)
    assert_expected_count(len(val_ds), "val", True, args.skip_count_check)

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_period)

    config = corrected_config_common(args, model_id="B2")
    config.update(
        {
            "task": "LV segmentation-only baseline",
            "segmentation_input": "expert-labeled ED and ES frames only",
            "segmentation_supervision": "ED/ES LV masks",
            "shared_with_B3": "R(2+1)D-18 encoder + FPN-style segmentation decoder",
            "dataset": {
                "train_labeled_videos": len(train_ds),
                "validation_labeled_videos": len(val_ds),
                "volume_tracings_required": True,
            },
            "loss": "BCEWithLogitsLoss(mean)",
            "checkpoint_rule": "highest validation Mean Dice",
            "test_used_during_training": False,
            "reuse_previous_B2": False,
            "reuse_reason": "segmentation architecture changed to match corrected B3 shared encoder/decoder",
        }
    )
    write_json(output / "run_config.json", config)

    history = []
    best_dice = -float("inf")
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, optimizer, False)
        val_metrics, val_rows = run_epoch(model, val_loader, device, None, True)
        scheduler.step()

        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            row = {"epoch": epoch, "phase": phase, "seed": args.seed}
            row.update(metrics)
            history.append(row)
        write_csv(output / "training_history.csv", history)

        checkpoint = {
            "model_id": "B2",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(checkpoint, output / "checkpoint.pt")
        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            best_epoch = epoch
            torch.save(checkpoint, output / "best.pt")
            write_json(output / "best_validation_metrics.json", val_metrics)
            write_csv(output / "best_validation_segmentation_metrics.csv", val_rows)

        print(
            f"Epoch {epoch:03d} | Dice ED={val_metrics['dice_ed']:.4f} "
            f"ES={val_metrics['dice_es']:.4f} Mean={val_metrics['mean_dice']:.4f} "
            f"HD95={val_metrics['mean_hd95']:.3f}"
        )

    write_json(
        output / "run_summary.json",
        {
            "model_id": "B2",
            "best_epoch": best_epoch,
            "best_validation_mean_dice": best_dice,
            "best_checkpoint": str(output / "best.pt"),
        },
    )


if __name__ == "__main__":
    main()
