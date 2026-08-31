#!/usr/bin/env python3
"""Train corrected Stage 1 B3: video EF + sparse ED/ES segmentation MTL.

EF path
-------
A multi-frame video clip is encoded by R(2+1)D-18. Spatiotemporal features are
pooled over T/H/W before one scalar EF output. Ground-truth ED/ES locations are
never passed to ``forward_ef``.

Segmentation path
-----------------
During training/segmentation evaluation only, expert-labeled ED/ES frames and
masks are passed to a segmentation decoder that shares the same R(2+1)D encoder.

Checkpoint rule
---------------
One checkpoint per seed/weight is selected only by lowest validation EF MAE.
EF and segmentation metrics for B3 are always reported from that same checkpoint.
Test data are not constructed by this script.
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
    WEIGHTS,
    assert_expected_count,
    corrected_config_common,
    make_loader,
    write_csv,
    write_json,
)
from echonet.utils.stage1_metrics import (
    dice_score,
    ef_fraction_to_percent,
    hd95_pixels,
    regression_metrics,
    summarize_segmentation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    p.add_argument("--weight", required=True, choices=["W1", "W2", "W3"])
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
    *,
    ef_weight: float,
    seg_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    compute_hd95: bool,
):
    training = optimizer is not None
    model.train(training)
    ef_criterion = torch.nn.MSELoss(reduction="mean")
    seg_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    ef_loss_sum = 0.0
    ef_n = 0
    seg_loss_sum = 0.0
    seg_n_videos = 0
    ef_targets, ef_predictions, filenames = [], [], []
    ed_dice, es_dice, ed_hd95, es_hd95 = [], [], [], []
    seg_rows = []
    started = time.time()

    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        # ----- EF: video only; no ED/ES data enters this call. -----
        video = batch["video"].to(device, dtype=torch.float32, non_blocking=True)
        ef = batch["ef"].to(device, dtype=torch.float32, non_blocking=True).reshape(-1)
        with torch.set_grad_enabled(training):
            ef_pred = model.forward_ef(video).reshape(-1)
            ef_loss = ef_criterion(ef_pred, ef)
            if training:
                # Backward now so the large video graph can be released before
                # the segmentation forward pass. Gradients accumulate until step().
                (ef_weight * ef_loss).backward()

        b = ef.numel()
        ef_loss_sum += float(ef_loss.detach().cpu()) * b
        ef_n += b
        ef_targets.extend(ef_fraction_to_percent(ef.detach().cpu().numpy()).tolist())
        ef_predictions.extend(
            ef_fraction_to_percent(ef_pred.detach().cpu().numpy()).tolist()
        )
        filenames.extend(list(batch["filename"]))

        # Release video activations before the second shared-encoder pass.
        del video, ef_pred

        # ----- Sparse segmentation supervision: labeled ED/ES frames only. -----
        has_seg = batch["has_segmentation"].bool()
        labeled = torch.nonzero(has_seg, as_tuple=False).reshape(-1)
        if labeled.numel() > 0:
            ed = batch["ed_image"][labeled].to(
                device, dtype=torch.float32, non_blocking=True
            )
            es = batch["es_image"][labeled].to(
                device, dtype=torch.float32, non_blocking=True
            )
            ed_mask = batch["ed_mask"][labeled].to(
                device, dtype=torch.float32, non_blocking=True
            )
            es_mask = batch["es_mask"][labeled].to(
                device, dtype=torch.float32, non_blocking=True
            )
            n_labeled = int(labeled.numel())
            frames = torch.cat([ed, es], dim=0)
            masks = torch.cat([ed_mask, es_mask], dim=0)

            with torch.set_grad_enabled(training):
                seg_logits = model.forward_segmentation(frames)
                seg_loss = seg_criterion(seg_logits, masks)
                if training:
                    (seg_weight * seg_loss).backward()

            seg_loss_sum += float(seg_loss.detach().cpu()) * n_labeled
            seg_n_videos += n_labeled

            probs = torch.sigmoid(seg_logits).detach().cpu().numpy()[:, 0]
            truth = masks.detach().cpu().numpy()[:, 0]
            batch_names = list(batch["filename"])
            labeled_cpu = labeled.cpu().tolist()
            for local_i, batch_i in enumerate(labeled_cpu):
                row = {"filename": batch_names[batch_i]}
                for phase, p, t in (
                    ("ed", probs[local_i], truth[local_i]),
                    ("es", probs[n_labeled + local_i], truth[n_labeled + local_i]),
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
                seg_rows.append(row)

        if training:
            optimizer.step()

    reg = regression_metrics(ef_targets, ef_predictions)
    if seg_n_videos > 0:
        if compute_hd95:
            seg = summarize_segmentation(ed_dice, es_dice, ed_hd95, es_hd95)
        else:
            de, ds = float(np.mean(ed_dice)), float(np.mean(es_dice))
            seg = {
                "dice_ed": de,
                "dice_es": ds,
                "mean_dice": (de + ds) / 2.0,
                "hd95_ed": float("nan"),
                "hd95_es": float("nan"),
                "mean_hd95": float("nan"),
            }
    else:
        seg = {
            "dice_ed": float("nan"),
            "dice_es": float("nan"),
            "mean_dice": float("nan"),
            "hd95_ed": float("nan"),
            "hd95_es": float("nan"),
            "mean_hd95": float("nan"),
        }

    raw_ef = ef_loss_sum / max(ef_n, 1)
    raw_seg = seg_loss_sum / max(seg_n_videos, 1)
    metrics = {
        **reg,
        **seg,
        "raw_ef_loss": raw_ef,
        "raw_seg_loss": raw_seg,
        "weighted_ef_loss": ef_weight * raw_ef,
        "weighted_seg_loss": seg_weight * raw_seg,
        "total_weighted_loss": ef_weight * raw_ef + seg_weight * raw_seg,
        "n_ef_videos": ef_n,
        "n_segmentation_videos": seg_n_videos,
        "elapsed_seconds": time.time() - started,
    }
    ef_rows = [
        {
            "filename": f,
            "ef_target_percent": y,
            "ef_prediction_percent": p,
        }
        for f, y, p in zip(filenames, ef_targets, ef_predictions)
    ]
    return metrics, ef_rows, seg_rows


def main():
    args = parse_args()
    ef_weight, seg_weight = WEIGHTS[args.weight]
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
        include_segmentation=True,
        require_segmentation=False,
        include_video=True,
    )
    val_ds = Stage1VideoDataset(
        args.data_root,
        "val",
        frames=args.frames,
        period=args.period,
        clip_sampling="center",
        include_segmentation=True,
        require_segmentation=False,
        include_video=True,
    )
    # B3 EF cohort is deliberately the same full split as B1.
    assert_expected_count(len(train_ds), "train", False, args.skip_count_check)
    assert_expected_count(len(val_ds), "val", False, args.skip_count_check)
    assert_expected_count(train_ds.n_with_segmentation, "train", True, args.skip_count_check)
    assert_expected_count(val_ds.n_with_segmentation, "val", True, args.skip_count_check)

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

    config = corrected_config_common(args, model_id="B3")
    config.update(
        {
            "task": "corrected video-based multi-task EF + LV segmentation",
            "weight_name": args.weight,
            "ef_weight": ef_weight,
            "seg_weight": seg_weight,
            "loss_formula": f"{ef_weight} * L_EF + {seg_weight} * L_seg",
            "segmentation": {
                "input": "expert-labeled ED/ES frames during segmentation training/evaluation only",
                "labels": "ED/ES LV masks",
                "loss": "BCEWithLogitsLoss(mean)",
                "shared_encoder": True,
            },
            "dataset": {
                "train_ef_videos": len(train_ds),
                "train_segmentation_videos": train_ds.n_with_segmentation,
                "validation_ef_videos": len(val_ds),
                "validation_segmentation_videos": val_ds.n_with_segmentation,
            },
            "checkpoint_rule": "lowest validation EF MAE",
            "same_checkpoint_for_ef_and_segmentation": True,
            "test_used_during_training": False,
            "oracle_guard": "forward_ef(video) accepts no ED/ES indices, frames, masks, or tracing data",
        }
    )
    write_json(output / "run_config.json", config)

    history = []
    best_mae = float("inf")
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            device,
            ef_weight=ef_weight,
            seg_weight=seg_weight,
            optimizer=optimizer,
            compute_hd95=False,
        )
        val_metrics, val_ef_rows, val_seg_rows = run_epoch(
            model,
            val_loader,
            device,
            ef_weight=ef_weight,
            seg_weight=seg_weight,
            optimizer=None,
            compute_hd95=True,
        )
        scheduler.step()

        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            row = {
                "epoch": epoch,
                "phase": phase,
                "seed": args.seed,
                "weight": args.weight,
            }
            row.update(metrics)
            history.append(row)
        write_csv(output / "training_history.csv", history)

        checkpoint = {
            "model_id": "B3",
            "weight": args.weight,
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
            write_csv(output / "best_validation_predictions.csv", val_ef_rows)
            write_csv(
                output / "best_validation_segmentation_metrics.csv", val_seg_rows
            )

        print(
            f"Epoch {epoch:03d} {args.weight} | val MAE={val_metrics['mae']:.3f} "
            f"RMSE={val_metrics['rmse']:.3f} R2={val_metrics['r2']:.3f} "
            f"r={val_metrics['pearson_r']:.3f} | "
            f"Dice={val_metrics['mean_dice']:.4f} HD95={val_metrics['mean_hd95']:.3f}"
        )

    write_json(
        output / "run_summary.json",
        {
            "model_id": "B3",
            "weight": args.weight,
            "best_epoch": best_epoch,
            "best_validation_mae": best_mae,
            "best_checkpoint": str(output / "best.pt"),
        },
    )


if __name__ == "__main__":
    main()
