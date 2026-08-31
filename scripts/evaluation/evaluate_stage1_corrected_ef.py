#!/usr/bin/env python3
"""Evaluate corrected B1/B3 EF from VIDEO ONLY on validation and test.

This is the professor-requested EF test script. It deliberately constructs
``Stage1VideoDataset(include_segmentation=False)`` and therefore requires only
``FileList.csv`` and ``Videos/``. It never opens ``VolumeTracings.csv`` and does
not request masks or ground-truth ED/ES indices.
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
    assert_b1_b3_matched,
    build_model,
    evaluate_ef,
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
    data_root = Path(args.data_root).resolve()
    if not (data_root / "FileList.csv").exists() or not (data_root / "Videos").is_dir():
        raise FileNotFoundError("EF evaluation requires FileList.csv and Videos/")
    # Intentionally no check for VolumeTracings.csv.

    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    lock_path = results_root / "validation_weight_selection" / "LOCKED_WEIGHT.txt"
    if not lock_path.exists():
        raise RuntimeError(
            f"No locked B3 weight at {lock_path}; validation-only selection must occur first."
        )
    locked = lock_path.read_text().strip()
    if locked not in {"W1", "W2", "W3"}:
        raise ValueError(f"Invalid locked weight: {locked!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_root = results_root / "evaluation"

    for seed in SEEDS:
        b1_path = run_root / "B1_video_ef" / f"seed_{seed}" / "best.pt"
        b3_path = run_root / "B3_video_mtl" / locked / f"seed_{seed}" / "best.pt"
        b1_ck = load_checkpoint(b1_path, device)
        b3_ck = load_checkpoint(b3_path, device)
        assert_b1_b3_matched(b1_ck, b3_ck)
        settings = video_settings(b1_ck)
        frames, period = settings["frames"], settings["period"]

        b1_model = build_model(b1_ck, device)
        b3_model = build_model(b3_ck, device)
        for split in ("val", "test"):
            dataset = Stage1VideoDataset(
                str(data_root),
                split,
                frames=frames,
                period=period,
                clip_sampling="center",
                include_segmentation=False,
                include_video=True,
            )
            assert_expected_count(
                len(dataset), split, False, args.skip_count_check
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
                ("B1_video_ef", b1_model, b1_ck, "B1"),
                ("B3_video_mtl", b3_model, b3_ck, "B3"),
            ):
                metrics, rows = evaluate_ef(model, loader, device)
                metrics.update(
                    {
                        "seed": seed,
                        "split": split,
                        "model_id": model_id,
                        "checkpoint_epoch": ck.get("epoch"),
                        "locked_weight": locked if model_id == "B3" else None,
                        "volume_tracings_used": False,
                        "ground_truth_ed_es_indices_used": False,
                        "masks_used": False,
                    }
                )
                out = eval_root / model_dir / f"seed_{seed}"
                write_json(out / f"{split}_ef_metrics.json", metrics)
                write_csv(out / f"{split}_ef_predictions.csv", rows)

        del b1_model, b3_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Completed video-only EF evaluation for seed {seed}")

    write_json(
        eval_root / "ef_video_only_manifest.json",
        {
            "status": "complete",
            "locked_weight": locked,
            "seeds": list(SEEDS),
            "required_data_files": ["FileList.csv", "Videos/"],
            "not_used": [
                "VolumeTracings.csv",
                "ground-truth ED/ES indices",
                "ED/ES masks",
            ],
            "test_used_for_selection": False,
        },
    )


if __name__ == "__main__":
    main()
