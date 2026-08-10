#!/usr/bin/env python3
"""Train the Stage 1 LV-segmentation-only baseline.

Uses DeepLabV3-ResNet50 with one binary output channel and BCEWithLogitsLoss.
The checkpoint is selected by highest validation Mean Dice, where Mean Dice is
the average of dataset-level mean ED Dice and mean ES Dice. HD95 is reported in
pixels. Test data are not loaded by this training script.
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

from echonet.datasets.echo import Echo
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import dice_score, hd95_pixels, summarize_segmentation


class EchoSegmentationDataset(Dataset):
    def __init__(self, root, split):
        self.dataset = Echo(
            root=root, split=split,
            target_type=["Filename", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"],
            mean=0.0, std=1.0, length=16, period=2, clips=1,
            pad=None, noise=None,
        )
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        _, t = self.dataset[index]
        filename, ed, es, ed_mask, es_mask = t
        return {
            "filename": filename,
            "ed_image": torch.as_tensor(ed, dtype=torch.float32),
            "es_image": torch.as_tensor(es, dtype=torch.float32),
            "ed_mask": (torch.as_tensor(ed_mask, dtype=torch.float32).unsqueeze(0) > 0.5).float(),
            "es_mask": (torch.as_tensor(es_mask, dtype=torch.float32).unsqueeze(0) > 0.5).float(),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"


def build_model(pretrained):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(
        weights=weights,
        weights_backbone=(ResNet50_Weights.IMAGENET1K_V1 if pretrained else None),
        aux_loss=True if pretrained else False,
    )
    model.aux_classifier = None
    last = model.classifier[-1]
    model.classifier[-1] = torch.nn.Conv2d(last.in_channels, 1, kernel_size=1)
    return model


def make_loader(ds, args, shuffle, device):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0), worker_init_fn=seed_worker,
        generator=make_generator(args.seed), drop_last=False,
    )


def run_epoch(model, loader, device, optimizer: Optional[torch.optim.Optimizer], compute_hd95: bool):
    training = optimizer is not None; model.train(training)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0; n = 0
    ed_dice: List[float] = []; es_dice: List[float] = []
    ed_hd95: List[float] = []; es_hd95: List[float] = []
    started = time.time()

    with torch.set_grad_enabled(training):
        for batch in loader:
            ed = batch["ed_image"].to(device); es = batch["es_image"].to(device)
            ed_mask = batch["ed_mask"].to(device); es_mask = batch["es_mask"].to(device)
            b = ed.shape[0]
            images = torch.cat([ed, es], dim=0); masks = torch.cat([ed_mask, es_mask], dim=0)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images)["out"]
            loss = criterion(logits, masks)
            if training: loss.backward(); optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * b; n += b

            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            truth = masks.detach().cpu().numpy()[:, 0]
            for p, t in zip(probs[:b], truth[:b]):
                pred, true = p >= 0.5, t >= 0.5
                ed_dice.append(dice_score(pred, true))
                if compute_hd95: ed_hd95.append(hd95_pixels(pred, true))
            for p, t in zip(probs[b:], truth[b:]):
                pred, true = p >= 0.5, t >= 0.5
                es_dice.append(dice_score(pred, true))
                if compute_hd95: es_hd95.append(hd95_pixels(pred, true))

    if compute_hd95:
        seg = summarize_segmentation(ed_dice, es_dice, ed_hd95, es_hd95)
    else:
        de, ds = float(np.mean(ed_dice)), float(np.mean(es_dice))
        seg = {"dice_ed": de, "dice_es": ds, "mean_dice": (de + ds)/2,
               "hd95_ed": float("nan"), "hd95_es": float("nan"), "mean_hd95": float("nan")}
    seg["segmentation_loss"] = total_loss / max(n, 1)
    seg["n_videos"] = n; seg["elapsed_seconds"] = time.time() - started
    return seg


def write_history(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    args = parse_args(); seed_everything(args.seed, deterministic=True)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EchoSegmentationDataset(args.data_root, "train")
    val_ds = EchoSegmentationDataset(args.data_root, "val")
    train_loader = make_loader(train_ds, args, True, device)
    val_loader = make_loader(val_ds, args, False, device)
    model = build_model(not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    config = vars(args).copy(); config.update({
        "task": "segmentation-only",
        "structure": "left-ventricular cavity",
        "mask": "binary",
        "segmentation_loss": "BCEWithLogitsLoss(mean)",
        "checkpoint_rule": "highest validation Mean Dice",
        "spatial_augmentation": "none",
        "git_commit": git_commit(), "command": " ".join(sys.argv),
    })
    with (output / "run_config.json").open("w") as f: json.dump(config, f, indent=2)

    history = []; best_dice = -float("inf"); best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, device, optimizer, False)
        val_m = run_epoch(model, val_loader, device, None, True)
        for phase, m in [("train", train_m), ("val", val_m)]:
            row = {"epoch": epoch, "phase": phase, "seed": args.seed}; row.update(m); history.append(row)
        write_history(output / "training_history.csv", history)
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_m, "config": config}
        torch.save(ckpt, output / "checkpoint.pt")
        if val_m["mean_dice"] > best_dice:
            best_dice = val_m["mean_dice"]; best_epoch = epoch
            torch.save(ckpt, output / "best.pt")
            with (output / "best_validation_metrics.json").open("w") as f: json.dump(val_m, f, indent=2)
        print(f"Epoch {epoch:03d} | Dice ED={val_m['dice_ed']:.4f} ES={val_m['dice_es']:.4f} Mean={val_m['mean_dice']:.4f} HD95={val_m['mean_hd95']:.3f}")

    with (output / "run_summary.json").open("w") as f:
        json.dump({"best_epoch": best_epoch, "best_validation_mean_dice": best_dice}, f, indent=2)


if __name__ == "__main__": main()
