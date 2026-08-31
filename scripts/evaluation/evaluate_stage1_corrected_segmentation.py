#!/usr/bin/env python3
"""Evaluate corrected B2/B3 LV segmentation on labeled ED/ES frames.

For B3 this loads the *same checkpoint selected by validation EF MAE*; no
segmentation-specific B3 checkpoint is selected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from echonet.datasets.stage1_video import Stage1VideoDataset
from echonet.utils.stage1_corrected import (
    SEEDS,
    assert_expected_count,
    make_loader,
    write_csv,
    write_json,
)
from echonet.utils.stage1_evaluation import (
    build_model,
    evaluate_segmentation,
    load_checkpoint,
    video_settings,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-root", default="output/stage1/corrected")
    p.add_argument("--results-root", default="results/stage1/corrected")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-count-check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    lock_path = results_root / "validation_weight_selection" / "LOCKED_WEIGHT.txt"
    if not lock_path.exists():
        raise RuntimeError(f"No locked B3 weight at {lock_path}")
    locked = lock_path.read_text().strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_root = results_root / "evaluation"

    for seed in SEEDS:
        b2_path = run_root / "B2_segmentation" / f"seed_{seed}" / "best.pt"
        b3_path = run_root / "B3_video_mtl" / locked / f"seed_{seed}" / "best.pt"
        b2_ck = load_checkpoint(b2_path, device)
        b3_ck = load_checkpoint(b3_path, device)
        frames = video_settings(b3_ck)["frames"]
        period = video_settings(b3_ck)["period"]
        b2_model = build_model(b2_ck, device)
        b3_model = build_model(b3_ck, device)

        for split in ("val", "test"):
            dataset = Stage1VideoDataset(
                args.data_root,
                split,
                frames=frames,
                period=period,
                include_segmentation=True,
                require_segmentation=True,
                include_video=False,
            )
            assert_expected_count(
                len(dataset), split, True, args.skip_count_check
            )
            loader = make_loader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                seed=seed,
                device=device,
            )
            for model_dir, model, ck, model_id in (
                ("B2_segmentation", b2_model, b2_ck, "B2"),
                ("B3_video_mtl", b3_model, b3_ck, "B3"),
            ):
                metrics, rows = evaluate_segmentation(model, loader, device)
                metrics.update(
                    {
                        "seed": seed,
                        "split": split,
                        "model_id": model_id,
                        "checkpoint_epoch": ck.get("epoch"),
                        "locked_weight": locked if model_id == "B3" else None,
                        "B3_checkpoint_selected_by": (
                            "validation EF MAE" if model_id == "B3" else None
                        ),
                    }
                )
                out = eval_root / model_dir / f"seed_{seed}"
                write_json(out / f"{split}_seg_metrics.json", metrics)
                write_csv(out / f"{split}_segmentation_metrics.csv", rows)

        del b2_model, b3_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Completed segmentation evaluation for seed {seed}")


if __name__ == "__main__":
    main()
