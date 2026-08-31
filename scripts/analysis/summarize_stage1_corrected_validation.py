#!/usr/bin/env python3
"""Create professor Table A and lock B3 weight using validation EF MAE only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from echonet.utils.stage1_corrected import SEEDS, WEIGHTS

METRICS = ["mae", "rmse", "r2", "pearson_r", "mean_dice", "mean_hd95"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-root",
        default="output/stage1/corrected",
        help="Root containing B3_video_mtl/W*/seed_*/best_validation_metrics.json",
    )
    p.add_argument(
        "--results-root",
        default="results/stage1/corrected/validation_weight_selection",
    )
    return p.parse_args()


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    if arr.size != 3:
        raise ValueError(f"Expected exactly 3 seeds, got {arr.size}")
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    summaries = {}
    for weight in ("W1", "W2", "W3"):
        rows = []
        for seed in SEEDS:
            path = run_root / "B3_video_mtl" / weight / f"seed_{seed}" / "best_validation_metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open() as f:
                metrics = json.load(f)
            row = {"weight": weight, "seed": seed}
            for metric in METRICS:
                if metric not in metrics:
                    raise KeyError(f"{metric} missing from {path}")
                row[metric] = float(metrics[metric])
            seed_rows.append(row)
            rows.append(row)

        summary = {"weight": weight}
        for metric in METRICS:
            mean, sd = mean_sd([r[metric] for r in rows])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
        summaries[weight] = summary

    # Professor's rule: validation EF MAE only.
    locked = min(summaries, key=lambda w: summaries[w]["mae_mean"])

    with (results_root / "weighting_seed_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_rows = [summaries[w] for w in ("W1", "W2", "W3")]
    with (results_root / "weighting_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (results_root / "LOCKED_WEIGHT.txt").write_text(locked + "\n")
    selection = {
        "locked_weight": locked,
        "selection_metric": "mean validation EF MAE across seeds 42, 2026, 3407",
        "test_data_used": False,
        "weights": {
            name: {"ef_weight": vals[0], "seg_weight": vals[1]}
            for name, vals in WEIGHTS.items()
        },
        "summaries": summaries,
    }
    with (results_root / "weighting_selection.json").open("w") as f:
        json.dump(selection, f, indent=2)

    def fmt(weight, metric, digits=4):
        s = summaries[weight]
        return f"{s[metric + '_mean']:.{digits}f} ± {s[metric + '_sd']:.{digits}f}"

    lines = [
        "# Corrected Stage 1 — Validation-Only Loss-Weight Selection",
        "",
        "All values are mean ± sample SD (ddof=1) across seeds 42, 2026, and 3407.",
        "The locked weight is selected using validation EF MAE only; test data are not used.",
        "",
        "| Setting | Val MAE ↓ | Val RMSE ↓ | Val R² ↑ | Val Pearson r ↑ | Val Mean Dice ↑ | Val HD95 ↓ | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for weight in ("W1", "W2", "W3"):
        ew, sw = WEIGHTS[weight]
        decision = "LOCKED" if weight == locked else ""
        lines.append(
            f"| {weight}: {ew:.1f} EF + {sw:.1f} Seg | {fmt(weight,'mae',3)} | "
            f"{fmt(weight,'rmse',3)} | {fmt(weight,'r2',3)} | {fmt(weight,'pearson_r',3)} | "
            f"{fmt(weight,'mean_dice',4)} | {fmt(weight,'mean_hd95',3)} | {decision} |"
        )
    lines += [
        "",
        f"**Locked weight: {locked}.** Reason: lowest mean validation EF MAE across the three required seeds.",
        "",
    ]
    (results_root / "table_a_validation_weight_selection.md").write_text("\n".join(lines))
    print(f"Locked B3 weight: {locked}")
    print(f"Wrote {results_root.resolve()}")


if __name__ == "__main__":
    main()
