#!/usr/bin/env python3
"""Summarize corrected Stage 1 Tables B/C with mean ± sample SD (ddof=1)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from echonet.utils.stage1_corrected import SEEDS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/stage1/corrected")
    return p.parse_args()


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    if arr.size != 3:
        raise ValueError(f"Expected 3 seed values, got {arr.size}")
    return float(arr.mean()), float(arr.std(ddof=1))


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def main():
    args = parse_args()
    root = Path(args.results_root)
    eval_root = root / "evaluation"
    final_root = root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    lock = (root / "validation_weight_selection" / "LOCKED_WEIGHT.txt").read_text().strip()

    ef_seed_rows = []
    seg_seed_rows = []
    for model_dir, label in (("B1_video_ef", "B1"), ("B3_video_mtl", "B3")):
        for split in ("val", "test"):
            for seed in SEEDS:
                m = load_json(eval_root / model_dir / f"seed_{seed}" / f"{split}_ef_metrics.json")
                ef_seed_rows.append(
                    {
                        "model": label,
                        "split": split,
                        "seed": seed,
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                        "r2": m["r2"],
                        "pearson_r": m["pearson_r"],
                        "n_videos": m["n_videos"],
                    }
                )
    for model_dir, label in (("B2_segmentation", "B2"), ("B3_video_mtl", "B3")):
        for split in ("val", "test"):
            for seed in SEEDS:
                m = load_json(eval_root / model_dir / f"seed_{seed}" / f"{split}_seg_metrics.json")
                seg_seed_rows.append(
                    {
                        "model": label,
                        "split": split,
                        "seed": seed,
                        "dice_ed": m["dice_ed"],
                        "dice_es": m["dice_es"],
                        "mean_dice": m["mean_dice"],
                        "mean_hd95": m["mean_hd95"],
                        "n_videos": m["n_videos"],
                    }
                )

    def write_rows(path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_rows(final_root / "ef_seed_results.csv", ef_seed_rows)
    write_rows(final_root / "segmentation_seed_results.csv", seg_seed_rows)

    ef_summary = []
    for model in ("B1", "B3"):
        for split in ("val", "test"):
            rows = [r for r in ef_seed_rows if r["model"] == model and r["split"] == split]
            s = {"model": model, "split": split}
            for metric in ("mae", "rmse", "r2", "pearson_r"):
                mean, sd = mean_sd([r[metric] for r in rows])
                s[f"{metric}_mean"] = mean; s[f"{metric}_sd"] = sd
            s["n_videos"] = rows[0]["n_videos"]
            ef_summary.append(s)

    seg_summary = []
    for model in ("B2", "B3"):
        for split in ("val", "test"):
            rows = [r for r in seg_seed_rows if r["model"] == model and r["split"] == split]
            s = {"model": model, "split": split}
            for metric in ("dice_ed", "dice_es", "mean_dice", "mean_hd95"):
                mean, sd = mean_sd([r[metric] for r in rows])
                s[f"{metric}_mean"] = mean; s[f"{metric}_sd"] = sd
            s["n_videos"] = rows[0]["n_videos"]
            seg_summary.append(s)

    write_rows(final_root / "table_b_ef.csv", ef_summary)
    write_rows(final_root / "table_c_segmentation.csv", seg_summary)

    def fmt(row, metric, digits=3):
        return f"{row[metric + '_mean']:.{digits}f} ± {row[metric + '_sd']:.{digits}f}"

    lines = [
        "# Corrected Stage 1 Final Results",
        "",
        f"Locked B3 weight: **{lock}** (selected using validation EF MAE only).",
        "All summary values are mean ± sample SD across seeds 42, 2026, and 3407 (ddof=1).",
        "",
        "## Table B. Final EF regression results",
        "",
        "| Model | Split | MAE ↓ | RMSE ↓ | R² ↑ | Pearson r ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ef_summary:
        model_name = "B1 Video EF-only" if row["model"] == "B1" else "B3 Corrected video MTL"
        split_name = "Validation" if row["split"] == "val" else "Test"
        lines.append(
            f"| {model_name} | {split_name} | {fmt(row,'mae')} | {fmt(row,'rmse')} | "
            f"{fmt(row,'r2')} | {fmt(row,'pearson_r')} |"
        )

    lines += [
        "",
        "## Table C. Final LV segmentation results",
        "",
        "| Model | Split | Dice ED ↑ | Dice ES ↑ | Mean Dice ↑ | Mean HD95 ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in seg_summary:
        model_name = "B2 Segmentation-only" if row["model"] == "B2" else "B3 Corrected video MTL"
        split_name = "Validation" if row["split"] == "val" else "Test"
        lines.append(
            f"| {model_name} | {split_name} | {fmt(row,'dice_ed',4)} | {fmt(row,'dice_es',4)} | "
            f"{fmt(row,'mean_dice',4)} | {fmt(row,'mean_hd95')} |"
        )

    b1_test = next(r for r in ef_summary if r["model"] == "B1" and r["split"] == "test")
    b3_test = next(r for r in ef_summary if r["model"] == "B3" and r["split"] == "test")
    b2_seg_test = next(r for r in seg_summary if r["model"] == "B2" and r["split"] == "test")
    b3_seg_test = next(r for r in seg_summary if r["model"] == "B3" and r["split"] == "test")

    ef_checks = {
        "MAE": b3_test["mae_mean"] < b1_test["mae_mean"],
        "RMSE": b3_test["rmse_mean"] < b1_test["rmse_mean"],
        "R²": b3_test["r2_mean"] > b1_test["r2_mean"],
        "Pearson r": b3_test["pearson_r_mean"] > b1_test["pearson_r_mean"],
    }
    if all(ef_checks.values()):
        ef_direction = "improved EF regression across all four requested metrics"
    elif any(ef_checks.values()):
        ef_direction = "showed mixed EF changes across the four requested metrics"
    else:
        ef_direction = "did not improve EF regression on any of the four requested metrics"
    dice_delta = b3_seg_test["mean_dice_mean"] - b2_seg_test["mean_dice_mean"]
    hd_delta = b3_seg_test["mean_hd95_mean"] - b2_seg_test["mean_hd95_mean"]
    lines += [
        "",
        "## Required student conclusion (auto-filled from the tables)",
        "",
        f"- **EF result:** B3 {ef_direction}. Test results: "
        f"MAE {fmt(b3_test,'mae')} vs {fmt(b1_test,'mae')}; "
        f"RMSE {fmt(b3_test,'rmse')} vs {fmt(b1_test,'rmse')}; "
        f"R² {fmt(b3_test,'r2')} vs {fmt(b1_test,'r2')}; "
        f"Pearson r {fmt(b3_test,'pearson_r')} vs {fmt(b1_test,'pearson_r')}.",
        f"- **Segmentation result:** B3 minus B2 test Mean Dice = {dice_delta:+.4f}; "
        f"B3 minus B2 test Mean HD95 = {hd_delta:+.3f} pixels.",
        "- **Task correction:** B3 produces EF from a multi-frame video clip via R(2+1)D temporal/spatial feature aggregation. The EF evaluation path does not load VolumeTracings.csv or use ED/ES indices/masks.",
        "",
    ]
    (final_root / "final_results.md").write_text("\n".join(lines))
    print(f"Wrote final summaries to {final_root.resolve()}")


if __name__ == "__main__":
    main()
