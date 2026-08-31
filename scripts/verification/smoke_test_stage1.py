#!/usr/bin/env python3
# HISTORICAL T0 MODEL SMOKE TEST. Use smoke_test_stage1_corrected.py.
"""Fast one-batch smoke test of the fixed multi-task Stage 1 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from echonet.datasets.echo import Echo
from echonet.losses.multitask_loss import MultitaskLoss
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import (
    dice_score,
    ef_fraction_to_percent,
    regression_metrics,
)


class SmokeDataset(Dataset):
    def __init__(self, root, split):
        self.ds = Echo(
            root=root, split=split,
            target_type=["Filename", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "EF"],
            mean=0.0, std=1.0, length=16, period=2, clips=1, pad=None, noise=None,
        )
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        _, t = self.ds[i]
        filename, ed, es, edm, esm, ef = t
        return {
            "filename": filename,
            "ed": torch.as_tensor(ed, dtype=torch.float32),
            "es": torch.as_tensor(es, dtype=torch.float32),
            "edm": (torch.as_tensor(edm, dtype=torch.float32).unsqueeze(0) > 0.5).float(),
            "esm": (torch.as_tensor(esm, dtype=torch.float32).unsqueeze(0) > 0.5).float(),
            "ef": torch.tensor(float(ef) / 100.0, dtype=torch.float32),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="outputs/stage1_audit/smoke_test.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args(); seed_everything(args.seed, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SmokeDataset(args.data_root, "train")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=make_generator(args.seed),
    )
    batch = next(iter(loader))
    model = MultitaskDeepLabV3(pretrained=not args.no_pretrained).to(device)
    criterion = MultitaskLoss(0.5, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ed = batch["ed"].to(device); es = batch["es"].to(device)
    edm = batch["edm"].to(device); esm = batch["esm"].to(device)
    ef = batch["ef"].to(device); b = ef.shape[0]
    images = torch.cat([ed, es], 0); masks = torch.cat([edm, esm], 0)

    optimizer.zero_grad(set_to_none=True)
    out = model(images)
    frame_ef = out["ef"].reshape(-1)
    video_ef = (frame_ef[:b] + frame_ef[b:]) / 2
    loss, components = criterion(out["segmentation"], masks, video_ef, ef)
    loss.backward(); optimizer.step()

    probs = torch.sigmoid(out["segmentation"]).detach().cpu().numpy()[:, 0]
    truth = masks.detach().cpu().numpy()[:, 0]
    dice = [dice_score(p >= 0.5, t >= 0.5) for p, t in zip(probs, truth)]
    ef_percent = ef_fraction_to_percent(ef.detach().cpu().numpy())
    prediction_percent = ef_fraction_to_percent(video_ef.detach().cpu().numpy())
    reg = regression_metrics(ef_percent, prediction_percent)

    result = {
        "device": str(device),
        "filenames": list(batch["filename"]),
        "segmentation_output_shape": list(out["segmentation"].shape),
        "frame_ef_output_shape": list(out["ef"].shape),
        "video_ef_output_shape": list(video_ef.shape),
        "ef_training_target_scale": "0-1 fraction",
        "ef_evaluation_scale": "0-100 percentage points",
        "ef_targets_fraction": ef.detach().cpu().numpy().tolist(),
        "ef_predictions_fraction": video_ef.detach().cpu().numpy().tolist(),
        "ef_targets_percent": ef_percent.tolist(),
        "ef_predictions_percent": prediction_percent.tolist(),
        "loss_components": components,
        "regression_metrics": reg,
        "mean_batch_dice": float(sum(dice) / len(dice)),
        "backward_and_optimizer_step_completed": True,
    }
    path = Path(args.output).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__": main()
