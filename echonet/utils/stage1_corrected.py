"""Shared helpers for the corrected Stage 1 B1/B2/B3 experiments."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from echonet.utils.reproducibility import make_generator, seed_worker

SEEDS = (42, 2026, 3407)
WEIGHTS = {
    "W1": (0.1, 0.9),
    "W2": (0.5, 0.5),
    "W3": (0.9, 0.1),
}

# Full EF split counts. Segmentation counts are lower only where tracings are
# absent (5 train and 1 test in the audited local copy).
EXPECTED_EF_COUNTS = {"train": 7465, "val": 1288, "test": 1277}
EXPECTED_SEG_COUNTS = {"train": 7460, "val": 1288, "test": 1276}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def device_metadata() -> dict:
    info = {
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


def make_loader(dataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int, device):
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


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def assert_expected_count(actual: int, split: str, segmentation: bool, skip: bool = False) -> None:
    if skip:
        return
    expected_map = EXPECTED_SEG_COUNTS if segmentation else EXPECTED_EF_COUNTS
    expected = expected_map[split.lower()]
    if actual != expected:
        kind = "segmentation-labeled" if segmentation else "full EF"
        raise RuntimeError(
            f"Unexpected {kind} {split} count: {actual}; expected {expected}. "
            "Use --skip-count-check only if you intentionally use a different dataset release."
        )


def corrected_config_common(args, *, model_id: str) -> dict:
    return {
        "stage": 1,
        "model_id": model_id,
        "corrected_task": "video-based continuous EF regression with sparse ED/ES LV segmentation supervision",
        "ef_input": {
            "type": "multi-frame video clip",
            "tensor_shape": f"[B, 3, {args.frames}, H, W]",
            "T": args.frames,
            "period": args.period,
            "training_sampling": "random valid clip start",
            "validation_sampling": "deterministic center clip",
            "ground_truth_ed_es_used_for_ef": False,
        },
        "split": {
            "source": "FileList.csv",
            "train": "TRAIN",
            "validation": "VAL",
            "test_used_during_training": False,
        },
        "preprocessing": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "spatial_augmentation": "none",
        },
        "ef_architecture": {
            "encoder": "torchvision r2plus1d_18",
            "temporal_module": "R(2+1)D spatiotemporal convolutions",
            "aggregation": "AdaptiveAvgPool3d((1,1,1)) over T/H/W before one Linear(512,1) EF output",
            "target_scale_training": "0-1 fraction",
            "reporting_scale": "0-100 percentage points",
        },
        "model": {
            "architecture": "Stage1VideoMultitaskModel",
            "encoder": "torchvision r2plus1d_18",
            "pretrained": not args.no_pretrained,
            "segmentation_decoder_width": args.segmentation_decoder_width,
            "ef_bias_fraction": 0.556,
        },
        "segmentation_decoder_width": args.segmentation_decoder_width,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "optimizer": "SGD",
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "lr_scheduler": "StepLR",
            "lr_step_period": args.lr_step_period,
            "seed": args.seed,
            "deterministic": not args.non_deterministic,
        },
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "environment": device_metadata(),
    }
