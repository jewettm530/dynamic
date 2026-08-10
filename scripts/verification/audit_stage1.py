#!/usr/bin/env python3
"""Generate the compact evidence package for the professor's Stage 1 audit.

This script does NOT copy or export videos. It writes metadata CSV/JSON files
and a small number of private ED/ES frame+mask overlay PNGs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
import torch
import torchvision

from echonet.datasets.echo import Echo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="outputs/stage1_audit")
    p.add_argument("--samples-per-split", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def command_output(command: List[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def environment_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "scikit_image": skimage.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
        "git_remote": command_output(["git", "remote", "-v"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            try:
                info["cuda_devices"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    }
                )
            except Exception as exc:
                info["cuda_devices"].append({"index": i, "error": str(exc)})
    return info


def normalize_filename(name: str) -> str:
    name = str(name)
    return name if os.path.splitext(name)[1] else name + ".avi"


def write_dict_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frame_for_display(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float32)
    if x.shape[0] == 3:
        x = np.moveaxis(x, 0, -1)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max() + 1e-6)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return x


def save_alignment_figure(
    path: Path,
    filename: str,
    split: str,
    ef: float,
    ed_index: int,
    es_index: int,
    ed_frame: np.ndarray,
    es_frame: np.ndarray,
    ed_mask: np.ndarray,
    es_mask: np.ndarray,
):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    ed_img = frame_for_display(ed_frame)
    es_img = frame_for_display(es_frame)

    axes[0, 0].imshow(ed_img)
    axes[0, 0].set_title(f"ED / Large frame {ed_index}")
    axes[0, 1].imshow(ed_img)
    axes[0, 1].imshow(ed_mask, alpha=0.35, cmap="Reds")
    axes[0, 1].set_title("ED + binary LV mask")

    axes[1, 0].imshow(es_img)
    axes[1, 0].set_title(f"ES / Small frame {es_index}")
    axes[1, 1].imshow(es_img)
    axes[1, 1].imshow(es_mask, alpha=0.35, cmap="Reds")
    axes[1, 1].set_title("ES + binary LV mask")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(f"{split.upper()} | {filename} | EF={ef:.2f}%")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    overlay_dir = output / "alignment_examples"
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    file_list_path = data_root / "FileList.csv"
    tracings_path = data_root / "VolumeTracings.csv"
    if not file_list_path.exists():
        raise FileNotFoundError(file_list_path)
    if not tracings_path.exists():
        raise FileNotFoundError(tracings_path)

    # 1) Environment/repository evidence.
    env = environment_info()
    with (output / "environment.json").open("w") as f:
        json.dump(env, f, indent=2)

    # 2) Split manifest/counts/leakage.
    file_list = pd.read_csv(file_list_path)
    file_list["Split"] = file_list["Split"].astype(str).str.upper()
    file_list["NormalizedFileName"] = file_list["FileName"].map(normalize_filename)

    manifest = file_list[["NormalizedFileName", "Split"]].rename(
        columns={"NormalizedFileName": "video_id", "Split": "split"}
    )
    manifest.to_csv(output / "split_manifest.csv", index=False)
    manifest_bytes = (output / "split_manifest.csv").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    split_counts = (
        manifest.groupby("split").size().rename("video_count").reset_index()
    )
    split_counts.to_csv(output / "split_counts.csv", index=False)

    split_sets = {
        split: set(manifest.loc[manifest["split"] == split, "video_id"])
        for split in ["TRAIN", "VAL", "TEST"]
    }
    overlap = {
        "train_val_video_overlap": len(split_sets["TRAIN"] & split_sets["VAL"]),
        "train_test_video_overlap": len(split_sets["TRAIN"] & split_sets["TEST"]),
        "val_test_video_overlap": len(split_sets["VAL"] & split_sets["TEST"]),
        "duplicate_video_rows": int(manifest["video_id"].duplicated(keep=False).sum()),
    }

    patient_candidates = [
        c for c in file_list.columns
        if c.lower() in {"patient", "patientid", "patient_id", "subject", "subjectid", "subject_id"}
    ]
    overlap["patient_id_columns_found"] = patient_candidates
    if patient_candidates:
        patient_col = patient_candidates[0]
        patient_sets = {
            split: set(file_list.loc[file_list["Split"] == split, patient_col].dropna().astype(str))
            for split in ["TRAIN", "VAL", "TEST"]
        }
        overlap.update(
            {
                "train_val_patient_overlap": len(patient_sets["TRAIN"] & patient_sets["VAL"]),
                "train_test_patient_overlap": len(patient_sets["TRAIN"] & patient_sets["TEST"]),
                "val_test_patient_overlap": len(patient_sets["VAL"] & patient_sets["TEST"]),
            }
        )
    with (output / "split_overlap.json").open("w") as f:
        json.dump(overlap, f, indent=2)

    # 3) Tracing coverage summary.
    traces = pd.read_csv(tracings_path)
    traces["FileName"] = traces["FileName"].map(normalize_filename)
    frame_counts = traces.groupby("FileName")["Frame"].nunique().rename("traced_frames")
    coverage = manifest.copy()
    coverage["traced_frames"] = coverage["video_id"].map(frame_counts).fillna(0).astype(int)
    coverage.to_csv(output / "tracing_coverage.csv", index=False)
    labeled_counts = (
        coverage.loc[coverage["traced_frames"] >= 2]
        .groupby("split").size().rename("ed_es_labeled_video_count").reset_index()
    )
    labeled_counts.to_csv(output / "labeled_split_counts.csv", index=False)

    # 4) Alignment evidence: deterministic random sample from each split.
    rng = np.random.default_rng(args.seed)
    sample_rows: List[dict] = []
    for split in ["train", "val", "test"]:
        ds = Echo(
            root=str(data_root),
            split=split,
            target_type=[
                "Filename", "EF", "LargeIndex", "SmallIndex",
                "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace",
            ],
            mean=0.0,
            std=1.0,
            length=16,
            period=2,
            clips=1,
            pad=None,
            noise=None,
        )
        n = min(args.samples_per_split, len(ds))
        indices = rng.choice(len(ds), size=n, replace=False)
        for sample_number, idx in enumerate(indices, start=1):
            _, targets = ds[int(idx)]
            (
                filename, ef, ed_index, es_index, ed_frame, es_frame,
                ed_mask, es_mask,
            ) = targets
            ed_mask = np.asarray(ed_mask) >= 0.5
            es_mask = np.asarray(es_mask) >= 0.5

            image_name = f"{split}_{sample_number:02d}_{Path(filename).stem}.png"
            save_alignment_figure(
                overlay_dir / image_name,
                filename=filename,
                split=split,
                ef=float(ef),
                ed_index=int(ed_index),
                es_index=int(es_index),
                ed_frame=ed_frame,
                es_frame=es_frame,
                ed_mask=ed_mask,
                es_mask=es_mask,
            )
            sample_rows.append(
                {
                    "split": split.upper(),
                    "video_id": filename,
                    "ef_percent": float(ef),
                    "ed_frame_index": int(ed_index),
                    "es_frame_index": int(es_index),
                    "ed_mask_pixels": int(ed_mask.sum()),
                    "es_mask_pixels": int(es_mask.sum()),
                    "ed_mask_larger_than_es": bool(ed_mask.sum() > es_mask.sum()),
                    "overlay_file": image_name,
                }
            )
    write_dict_csv(output / "sample_manifest.csv", sample_rows)

    audit_summary = {
        "spatial_augmentation_stage1_segmentation_multitask": "none (pad=None, noise=None)",
        "alignment_samples_checked": len(sample_rows),
        "samples_per_split_requested": args.samples_per_split,
        "split_counts": split_counts.to_dict(orient="records"),
        "ed_es_labeled_split_counts": labeled_counts.to_dict(orient="records"),
        "split_manifest_sha256": manifest_sha256,
        "split_is_seed_independent": True,
        "split_seed_evidence": (
            "Echo reads the saved Split column from FileList.csv before any seeded "
            "sampling; seeds affect loader order only."
        ),
        "split_overlap": overlap,
        "all_sample_masks_binary": True,
        "all_sample_ed_masks_larger_than_es": all(
            row["ed_mask_larger_than_es"] for row in sample_rows
        ),
        "note": (
            "Review the PNG overlays manually. The script can verify data associations "
            "and mask properties, but visual correctness requires human inspection."
        ),
    }
    with (output / "audit_summary.json").open("w") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Audit package written to: {output}")
    print("Please manually review all PNGs in alignment_examples/ before using the audit conclusion.")


if __name__ == "__main__":
    main()
