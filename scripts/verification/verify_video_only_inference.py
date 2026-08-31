#!/usr/bin/env python3
"""Prove that corrected EF inference works without VolumeTracings.csv.

A temporary dataset root is created containing only symlinks to FileList.csv
and Videos/. No tracing file is present. The script loads a deterministic test
clip and, optionally, runs a frozen B1/B3 checkpoint on it.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from echonet.datasets.stage1_video import Stage1VideoDataset
from echonet.modeling.stage1_video_multitask import Stage1VideoMultitaskModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default=None, help="Optional B1 or B3 best.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--period", type=int, default=2)
    p.add_argument("--output", default="outputs/stage1_corrected/video_only_inference.json")
    return p.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()
    root = Path(args.data_root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory(prefix="echonet_video_only_") as td:
        temp_root = Path(td)
        os.symlink(root / "FileList.csv", temp_root / "FileList.csv")
        os.symlink(root / "Videos", temp_root / "Videos", target_is_directory=True)
        tracing_path = temp_root / "VolumeTracings.csv"
        if tracing_path.exists():
            raise RuntimeError("Temporary video-only root unexpectedly contains tracing data")

        ds = Stage1VideoDataset(
            str(temp_root),
            args.split,
            frames=args.frames,
            period=args.period,
            clip_sampling="center",
            include_segmentation=False,
            include_video=True,
        )
        sample = ds[0]
        payload = {
            "status": "pass",
            "split": args.split,
            "n_videos": len(ds),
            "sample_filename": sample["filename"],
            "sample_video_shape": list(sample["video"].shape),
            "VolumeTracings_present": False,
            "ED_ES_indices_requested": False,
            "ED_ES_masks_requested": False,
            "checkpoint_inference_run": False,
        }

        if args.checkpoint:
            ck = load_checkpoint(Path(args.checkpoint), device)
            cfg = ck.get("config", {})
            width = int(cfg.get("segmentation_decoder_width", 128))
            model = Stage1VideoMultitaskModel(
                pretrained=False, segmentation_decoder_width=width
            ).to(device)
            model.load_state_dict(ck["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                pred = model.forward_ef(sample["video"].unsqueeze(0).to(device))
            payload["checkpoint_inference_run"] = True
            payload["checkpoint_model_id"] = ck.get("model_id")
            payload["ef_prediction_percent"] = float(pred.item() * 100.0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
