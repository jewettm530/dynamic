#!/usr/bin/env python3
"""Copy safe Stage 1 run artifacts from local output/ into tracked results/.

Checkpoints stay local. This collector copies only configs, histories, metrics,
and CSV predictions/metric tables; it never copies videos, frames, masks,
tracings, or patient-derived visualizations.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SAFE_FILES = {
    "run_config.json",
    "run_summary.json",
    "best_validation_metrics.json",
    "training_history.csv",
    "best_validation_predictions.csv",
    "best_validation_segmentation_metrics.csv",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="output/stage1/corrected")
    p.add_argument("--results-root", default="results/stage1/corrected/runs")
    return p.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.run_root)
    dst_root = Path(args.results_root)
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    copied = 0
    for src in src_root.rglob("*"):
        if not src.is_file() or src.name not in SAFE_FILES:
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} safe reproducibility files to {dst_root.resolve()}")
    print("Checkpoints (*.pt) and patient-derived data were not copied.")


if __name__ == "__main__":
    main()
