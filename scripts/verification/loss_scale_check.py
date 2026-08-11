#!/usr/bin/env python3
"""Short Stage 1 training check for raw EF-vs-segmentation loss scale.

Runs a limited number of actual optimizer steps and records, for every batch:
raw L_EF, raw L_seg, weighted EF term, weighted segmentation term, total loss.
Use W2 (0.5/0.5) by default only as a diagnostic; this run is not one of the
nine official weighting experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from echonet.losses.multitask_loss import MultitaskLoss
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from torch.utils.data import Dataset
from echonet.datasets.echo import Echo


class EchoMultitaskDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.dataset = Echo(
            root=root, split=split,
            target_type=["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "EF"],
            mean=0.0, std=1.0, length=16, period=2, clips=1, pad=None, noise=None,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        _, targets = self.dataset[index]
        ed, es, ed_mask, es_mask, ef = targets
        ed = torch.as_tensor(ed, dtype=torch.float32)
        es = torch.as_tensor(es, dtype=torch.float32)
        ed_mask = (torch.as_tensor(ed_mask, dtype=torch.float32).unsqueeze(0) > 0.5).float()
        es_mask = (torch.as_tensor(es_mask, dtype=torch.float32).unsqueeze(0) > 0.5).float()
        return {
            "ed_image": ed,
            "es_image": es,
            "ed_mask": ed_mask,
            "es_mask": es_mask,
            "ef": torch.tensor(float(ef) / 100.0, dtype=torch.float32),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="outputs/stage1_audit/loss_scale_check.csv")
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ef-weight", type=float, default=0.5)
    p.add_argument("--seg-weight", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = EchoMultitaskDataset(args.data_root, "train")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )

    model = MultitaskDeepLabV3(pretrained=not args.no_pretrained).to(device)
    criterion = MultitaskLoss(args.ef_weight, args.seg_weight)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    rows = []
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > args.batches:
            break
        ed = batch["ed_image"].to(device)
        es = batch["es_image"].to(device)
        ed_mask = batch["ed_mask"].to(device)
        es_mask = batch["es_mask"].to(device)
        ef = batch["ef"].to(device)
        b = ef.shape[0]

        images = torch.cat([ed, es], dim=0)
        masks = torch.cat([ed_mask, es_mask], dim=0)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        frame_ef = outputs["ef"].reshape(-1)
        video_ef = (frame_ef[:b] + frame_ef[b:]) / 2.0
        loss, components = criterion(
            outputs["segmentation"], masks, video_ef, ef
        )
        loss.backward()
        optimizer.step()

        row = {"batch": batch_index}
        row.update(components)
        rows.append(row)
        print(
            f"batch={batch_index:03d} "
            f"L_EF={components['raw_ef_loss']:.4f} "
            f"L_seg={components['raw_seg_loss']:.4f} "
            f"total={components['total_loss']:.4f}"
        )

    if not rows:
        raise RuntimeError("No batches were processed")

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for key in [
        "raw_ef_loss", "raw_seg_loss", "weighted_ef_loss",
        "weighted_seg_loss", "total_loss",
    ]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["ef_training_target_scale"] = "0-1 fraction"
    summary["ef_evaluation_scale"] = "0-100 percentage points"
    summary["ef_percent_conversion"] = "prediction * 100"
    summary["ef_loss"] = "MSELoss(mean)"
    summary["segmentation_loss"] = "BCEWithLogitsLoss(mean)"

    summary_path = output.with_name(output.stem + "_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved batch losses: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
