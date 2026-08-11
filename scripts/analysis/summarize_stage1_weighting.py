#!/usr/bin/env python3
"""Summarize EchoNet-Dynamic Stage 1 multi-task loss-weighting results.

This script is intended for Step 2 of the Stage 1 baseline plan. It reads the
three seed-level validation results for W1/W2/W3, calculates mean +/- sample SD
(ddof=1), writes auditable seed-level and summary CSV files, and recommends the
weight according to the prescribed validation-only rule:

    1. lowest mean validation EF MAE
    2. lower mean validation RMSE if MAE is tied
    3. higher mean validation Mean Dice if RMSE is tied

The research plan also says to reject a setting if segmentation clearly fails.
Because the plan does not define a numeric failure threshold, this script does
NOT invent one. You may optionally provide --min-acceptable-mean-dice, or mark
specific weights with --reject-weight. Otherwise, the script reports a numeric
recommendation and explicitly requires manual segmentation review before the
weight is considered locked.

The script NEVER reads or evaluates the test set.

Expected run layout
-------------------
output/stage1/weighting/
    W1/seed_42/
    W1/seed_2026/
    W1/seed_3407/
    W2/...
    W3/...

Each completed run should contain at least:
    best_validation_metrics.json
    run_summary.json
    run_config.json
    config.yaml

Typical usage
-------------
After all nine runs finish:

    python scripts/analysis/summarize_stage1_weighting.py

While runs are still in progress:

    python scripts/analysis/summarize_stage1_weighting.py --allow-incomplete

If your lab/professor later defines a minimum acceptable mean Dice:

    python scripts/analysis/summarize_stage1_weighting.py \
        --min-acceptable-mean-dice 0.80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


EXPECTED_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "W1": (0.1, 0.9),
    "W2": (0.5, 0.5),
    "W3": (0.9, 0.1),
}
EXPECTED_SEEDS: Tuple[int, ...] = (42, 2026, 3407)

# Metrics required for the professor's Step 2 comparison table.
SELECTION_METRICS: Tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
    "mean_dice",
    "mean_hd95",
)

# Additional metrics worth preserving in the seed-level audit CSV.
ALL_METRICS: Tuple[str, ...] = (
    "raw_ef_loss",
    "raw_seg_loss",
    "weighted_ef_loss",
    "weighted_seg_loss",
    "total_loss",
    "mae",
    "rmse",
    "r2",
    "pearson_r",
    "dice_ed",
    "dice_es",
    "mean_dice",
    "hd95_ed",
    "hd95_es",
    "mean_hd95",
    "n_videos",
    "elapsed_seconds",
    "peak_gpu_memory_allocated",
)

# Fields that must stay fixed across all nine official runs.
FIXED_RUN_CONFIG_FIELDS: Tuple[str, ...] = (
    "epochs",
    "batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "regression_hidden_dim",
    "dropout",
    "no_pretrained",
    "non_deterministic",
    "ef_training_target_scale",
    "ef_evaluation_scale",
    "ef_percent_conversion",
    "ef_loss",
    "segmentation_loss",
    "checkpoint_rule",
    "spatial_augmentation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Stage 1 W1/W2/W3 validation results across three seeds."
    )
    parser.add_argument(
        "--input-root",
        default="output/stage1/weighting",
        help="Root containing W1/W2/W3 run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/stage1/weighting",
        help="Directory for seed-level, summary, and selection files.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Summarize completed runs even if all nine are not finished. "
            "No final weight recommendation is made until all nine exist."
        ),
    )
    parser.add_argument(
        "--min-acceptable-mean-dice",
        type=float,
        default=None,
        help=(
            "Optional explicit segmentation-failure threshold. A weight whose "
            "mean validation Mean Dice is below this value is rejected. "
            "Leave unset unless you have a defensible threshold."
        ),
    )
    parser.add_argument(
        "--reject-weight",
        action="append",
        default=[],
        choices=sorted(EXPECTED_WEIGHTS),
        help=(
            "Manually reject a weighting because segmentation clearly failed. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places used for formatted mean +/- SD strings.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def parse_identity_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def validate_expected_config(
    weight_name: str,
    seed: int,
    yaml_config: dict,
    run_config: dict,
    run_dir: Path,
) -> Tuple[float, float]:
    """Validate weight/seed identity and return (ef_weight, seg_weight)."""
    expected_ef, expected_seg = EXPECTED_WEIGHTS[weight_name]

    weighting = yaml_config.get("weighting", {})
    yaml_name = str(weighting.get("name", "")).upper()
    yaml_ef = float(weighting.get("ef_weight"))
    yaml_seg = float(weighting.get("seg_weight"))

    if yaml_name != weight_name:
        raise ValueError(
            f"{run_dir}: config.yaml says weighting.name={yaml_name!r}, "
            f"expected {weight_name!r}."
        )
    if not is_close(yaml_ef, expected_ef) or not is_close(yaml_seg, expected_seg):
        raise ValueError(
            f"{run_dir}: unexpected weights ({yaml_ef}, {yaml_seg}); "
            f"expected ({expected_ef}, {expected_seg})."
        )
    if not is_close(yaml_ef + yaml_seg, 1.0):
        raise ValueError(f"{run_dir}: loss weights do not sum to 1.0.")

    yaml_seeds = [int(x) for x in yaml_config.get("training", {}).get("seeds", [])]
    if yaml_seeds and yaml_seeds != list(EXPECTED_SEEDS):
        raise ValueError(
            f"{run_dir}: config seed list is {yaml_seeds}, expected {list(EXPECTED_SEEDS)}."
        )

    # The run-specific JSON should agree with the YAML and directory identity.
    if int(run_config.get("seed", seed)) != seed:
        raise ValueError(
            f"{run_dir}: run_config seed={run_config.get('seed')} does not match directory seed={seed}."
        )
    if not is_close(float(run_config.get("ef_weight", yaml_ef)), yaml_ef):
        raise ValueError(f"{run_dir}: run_config EF weight disagrees with config.yaml.")
    if not is_close(float(run_config.get("seg_weight", yaml_seg)), yaml_seg):
        raise ValueError(f"{run_dir}: run_config segmentation weight disagrees with config.yaml.")

    # Step 2 must not use the test split for weight selection.
    test_used = yaml_config.get("data", {}).get("test_split_used", False)
    if bool(test_used):
        raise ValueError(f"{run_dir}: config.yaml indicates that the test split was used.")

    return yaml_ef, yaml_seg


def collect_runs(input_root: Path, allow_incomplete: bool) -> pd.DataFrame:
    rows: List[dict] = []
    missing: List[str] = []

    for weight_name in ("W1", "W2", "W3"):
        for seed in EXPECTED_SEEDS:
            run_dir = input_root / weight_name / f"seed_{seed}"
            required = {
                "metrics": run_dir / "best_validation_metrics.json",
                "summary": run_dir / "run_summary.json",
                "run_config": run_dir / "run_config.json",
                "yaml_config": run_dir / "config.yaml",
            }

            missing_here = [str(path) for path in required.values() if not path.exists()]
            if missing_here:
                missing.extend(missing_here)
                continue

            metrics = read_json(required["metrics"])
            summary = read_json(required["summary"])
            run_config = read_json(required["run_config"])
            yaml_config = read_yaml(required["yaml_config"])
            identity = parse_identity_file(run_dir / "run_identity.txt")

            ef_weight, seg_weight = validate_expected_config(
                weight_name, seed, yaml_config, run_config, run_dir
            )

            for metric in SELECTION_METRICS:
                if metric not in metrics:
                    raise KeyError(f"{required['metrics']}: missing required metric {metric!r}.")

            best_epoch = summary.get("best_epoch")
            if best_epoch is None:
                raise ValueError(f"{required['summary']}: best_epoch is missing.")

            row = {
                "weight": weight_name,
                "ef_weight": ef_weight,
                "seg_weight": seg_weight,
                "seed": seed,
                "best_epoch": int(best_epoch),
                "best_checkpoint": summary.get("best_checkpoint", str(run_dir / "best.pt")),
                "git_commit": run_config.get("git_commit", identity.get("git_commit", "unknown")),
                "checkpoint_rule": run_config.get("checkpoint_rule", "unknown"),
                "ef_training_target_scale": run_config.get(
                    "ef_training_target_scale", "unknown"
                ),
                "ef_evaluation_scale": run_config.get("ef_evaluation_scale", "unknown"),
                "ef_loss": run_config.get("ef_loss", "unknown"),
                "segmentation_loss": run_config.get("segmentation_loss", "unknown"),
                "run_dir": str(run_dir),
            }
            for metric in ALL_METRICS:
                row[f"val_{metric}"] = metrics.get(metric, np.nan)

            # Preserve fixed training settings so consistency can be audited.
            for field in FIXED_RUN_CONFIG_FIELDS:
                row[f"config_{field}"] = run_config.get(field)

            rows.append(row)

    expected_total = len(EXPECTED_WEIGHTS) * len(EXPECTED_SEEDS)
    if missing and not allow_incomplete:
        print("ERROR: Step 2 is incomplete. Missing required files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nRe-run with --allow-incomplete only if you want an interim summary.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not rows:
        raise SystemExit(f"No completed Stage 1 weighting runs found under {input_root}.")

    df = pd.DataFrame(rows).sort_values(["weight", "seed"]).reset_index(drop=True)

    # Duplicate identity is always an error.
    if df.duplicated(["weight", "seed"]).any():
        duplicates = df.loc[df.duplicated(["weight", "seed"], keep=False), ["weight", "seed"]]
        raise ValueError(f"Duplicate weight/seed runs found:\n{duplicates}")

    print(f"Found {len(df)}/{expected_total} completed runs.")
    if missing and allow_incomplete:
        print(f"Interim mode: {len(missing)} required files are still missing.")

    return df


def validate_run_consistency(seed_df: pd.DataFrame) -> List[str]:
    """Return human-readable consistency messages; raise on invalid experiments."""
    messages: List[str] = []

    # All completed runs must use the same Git commit.
    commits = [str(x) for x in seed_df["git_commit"].dropna().unique()]
    if len(commits) > 1:
        raise ValueError(
            "Official weighting runs use more than one Git commit: " + ", ".join(commits)
        )
    if commits:
        messages.append(f"Git commit consistent across completed runs: {commits[0]}")

    # The required checkpoint rule must be the same for every run.
    rules = [str(x) for x in seed_df["checkpoint_rule"].dropna().unique()]
    if len(rules) > 1:
        raise ValueError("Checkpoint rule differs across runs: " + ", ".join(rules))
    if rules:
        messages.append(f"Checkpoint rule consistent: {rules[0]}")

    # Check all fixed hyperparameters/settings.
    for field in FIXED_RUN_CONFIG_FIELDS:
        col = f"config_{field}"
        if col not in seed_df.columns:
            continue
        normalized = seed_df[col].map(lambda x: json.dumps(x, sort_keys=True, default=str))
        unique_values = normalized.dropna().unique()
        if len(unique_values) > 1:
            examples = seed_df[["weight", "seed", col]].to_string(index=False)
            raise ValueError(
                f"Fixed setting {field!r} differs across official runs:\n{examples}"
            )

    messages.append(
        "Fixed architecture/training settings are consistent across completed runs; "
        "only seed and EF/segmentation loss weights vary."
    )
    return messages


def mean_sd(series: pd.Series) -> Tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size >= 2 else float("nan")
    return mean, sd


def format_mean_sd(mean: float, sd: float, precision: int) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(sd):
        return f"{mean:.{precision}f} +/- NA"
    return f"{mean:.{precision}f} +/- {sd:.{precision}f}"


def build_summary(seed_df: pd.DataFrame, precision: int) -> pd.DataFrame:
    rows: List[dict] = []

    metric_columns = {
        "mae": "val_mae",
        "rmse": "val_rmse",
        "r2": "val_r2",
        "pearson_r": "val_pearson_r",
        "dice_ed": "val_dice_ed",
        "dice_es": "val_dice_es",
        "mean_dice": "val_mean_dice",
        "hd95_ed": "val_hd95_ed",
        "hd95_es": "val_hd95_es",
        "mean_hd95": "val_mean_hd95",
    }

    for weight_name in ("W1", "W2", "W3"):
        group = seed_df[seed_df["weight"] == weight_name]
        if group.empty:
            continue

        expected_ef, expected_seg = EXPECTED_WEIGHTS[weight_name]
        row = {
            "weight": weight_name,
            "ef_weight": expected_ef,
            "seg_weight": expected_seg,
            "n_runs": int(len(group)),
            "seeds_present": ",".join(str(x) for x in sorted(group["seed"].astype(int))),
        }

        for short_name, col in metric_columns.items():
            mean, sd = mean_sd(group[col])
            row[f"val_{short_name}_mean"] = mean
            row[f"val_{short_name}_sd"] = sd
            row[f"val_{short_name}_mean_sd"] = format_mean_sd(mean, sd, precision)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("weight").reset_index(drop=True)


def segmentation_rejection_reason(
    row: pd.Series,
    manual_rejections: Sequence[str],
    min_acceptable_mean_dice: Optional[float],
) -> Optional[str]:
    weight = str(row["weight"])
    mean_dice = float(row["val_mean_dice_mean"])
    mean_hd95 = float(row["val_mean_hd95_mean"])

    if weight in manual_rejections:
        return "manually rejected because segmentation clearly failed"
    if not np.isfinite(mean_dice) or not np.isfinite(mean_hd95):
        return "rejected because segmentation summary contains non-finite values"
    if min_acceptable_mean_dice is not None and mean_dice < min_acceptable_mean_dice:
        return (
            f"rejected because mean validation Mean Dice {mean_dice:.6f} is below "
            f"the supplied threshold {min_acceptable_mean_dice:.6f}"
        )
    return None


def choose_weight(
    summary_df: pd.DataFrame,
    min_acceptable_mean_dice: Optional[float],
    manual_rejections: Sequence[str],
) -> dict:
    selection: dict = {
        "complete": False,
        "recommended_weight": None,
        "locked_weight": None,
        "manual_segmentation_review_required": min_acceptable_mean_dice is None,
        "segmentation_failure_threshold": min_acceptable_mean_dice,
        "rejections": {},
        "selection_rule": [
            "lowest mean validation EF MAE",
            "lower mean validation RMSE if MAE is tied",
            "higher mean validation Mean Dice if RMSE is tied",
            "reject a setting if segmentation clearly fails",
        ],
    }

    # Selection is only valid after 3 runs for all 3 settings.
    complete = (
        set(summary_df["weight"]) == set(EXPECTED_WEIGHTS)
        and (summary_df["n_runs"] == len(EXPECTED_SEEDS)).all()
    )
    selection["complete"] = bool(complete)
    if not complete:
        selection["reason"] = "Step 2 is incomplete; no weight recommendation was made."
        return selection

    candidate_rows: List[pd.Series] = []
    for _, row in summary_df.iterrows():
        reason = segmentation_rejection_reason(
            row, manual_rejections, min_acceptable_mean_dice
        )
        if reason:
            selection["rejections"][str(row["weight"])] = reason
        else:
            candidate_rows.append(row)

    if not candidate_rows:
        selection["reason"] = "All three settings were rejected for segmentation failure."
        return selection

    # Apply the professor's hierarchy exactly. Floating-point equality is treated
    # as a tie only when the stored mean values are exactly equal.
    candidates = pd.DataFrame(candidate_rows).copy()

    min_mae = candidates["val_mae_mean"].min()
    candidates = candidates[candidates["val_mae_mean"] == min_mae]
    criterion = "lowest mean validation EF MAE"

    if len(candidates) > 1:
        min_rmse = candidates["val_rmse_mean"].min()
        candidates = candidates[candidates["val_rmse_mean"] == min_rmse]
        criterion += "; MAE tie broken by lower mean validation RMSE"

    if len(candidates) > 1:
        max_dice = candidates["val_mean_dice_mean"].max()
        candidates = candidates[candidates["val_mean_dice_mean"] == max_dice]
        criterion += "; RMSE tie broken by higher mean validation Mean Dice"

    # A remaining exact tie is possible in principle. Keep it explicit rather
    # than inventing an extra criterion.
    if len(candidates) != 1:
        selection["reason"] = (
            "The prescribed criteria produced an unresolved exact tie among: "
            + ", ".join(candidates["weight"].astype(str))
        )
        return selection

    winner = candidates.iloc[0]
    selection["recommended_weight"] = str(winner["weight"])
    selection["reason"] = criterion
    selection["winner_metrics"] = {
        "mean_val_mae": float(winner["val_mae_mean"]),
        "mean_val_rmse": float(winner["val_rmse_mean"]),
        "mean_val_r2": float(winner["val_r2_mean"]),
        "mean_val_mean_dice": float(winner["val_mean_dice_mean"]),
        "mean_val_mean_hd95": float(winner["val_mean_hd95_mean"]),
    }

    # Only call it "locked" automatically if an explicit segmentation-failure
    # rule was supplied. Otherwise the plan requires a human judgment that
    # segmentation did not clearly fail.
    if min_acceptable_mean_dice is not None or manual_rejections:
        selection["locked_weight"] = selection["recommended_weight"]
        selection["manual_segmentation_review_required"] = False
    else:
        selection["reason"] += (
            ". Numeric recommendation only; manually verify that segmentation "
            "did not clearly fail before recording this as the locked weight."
        )

    return selection


def write_markdown_table(summary_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Stage 1 Step 2 — Validation Loss-Weighting Summary",
        "",
        "All values are mean +/- sample SD across seeds 42, 2026, and 3407.",
        "",
        "| Setting | Formula | Val MAE ↓ | Val RMSE ↓ | Val R² ↑ | Val Mean Dice ↑ | Val HD95 ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    formula = {
        "W1": "0.1 L_EF + 0.9 L_seg",
        "W2": "0.5 L_EF + 0.5 L_seg",
        "W3": "0.9 L_EF + 0.1 L_seg",
    }

    for _, row in summary_df.iterrows():
        lines.append(
            "| {w} | {formula} | {mae} | {rmse} | {r2} | {dice} | {hd95} |".format(
                w=row["weight"],
                formula=formula[str(row["weight"])],
                mae=row["val_mae_mean_sd"],
                rmse=row["val_rmse_mean_sd"],
                r2=row["val_r2_mean_sd"],
                dice=row["val_mean_dice_mean_sd"],
                hd95=row["val_mean_hd95_mean_sd"],
            )
        )

    path.write_text("\n".join(lines) + "\n")


def write_selection_text(selection: dict, summary_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "EchoNet-Dynamic Stage 1 Step 2 Weight Selection",
        "=" * 52,
        "",
        "Selection rule:",
        "  1. Lowest mean validation EF MAE",
        "  2. Lower mean validation RMSE if tied",
        "  3. Higher mean validation Mean Dice if still tied",
        "  4. Reject a setting if segmentation clearly fails",
        "",
    ]

    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['weight']}:",
                f"  Val MAE       = {row['val_mae_mean_sd']}",
                f"  Val RMSE      = {row['val_rmse_mean_sd']}",
                f"  Val R2        = {row['val_r2_mean_sd']}",
                f"  Val Mean Dice = {row['val_mean_dice_mean_sd']}",
                f"  Val Mean HD95 = {row['val_mean_hd95_mean_sd']}",
                "",
            ]
        )

    if not selection.get("complete"):
        lines.append("Step 2 is incomplete; no weight should be locked yet.")
    elif selection.get("recommended_weight") is None:
        lines.append("No weight could be recommended.")
        lines.append(str(selection.get("reason", "")))
    else:
        lines.append(f"Numeric recommendation: {selection['recommended_weight']}")
        lines.append(f"Reason: {selection['reason']}")
        if selection.get("locked_weight"):
            lines.append(f"Locked multi-task weight: {selection['locked_weight']}")
        else:
            lines.append(
                "Locked multi-task weight: NOT YET LOCKED — manually confirm that "
                "segmentation did not clearly fail."
            )

    if selection.get("rejections"):
        lines.append("")
        lines.append("Rejected settings:")
        for weight, reason in selection["rejections"].items():
            lines.append(f"  {weight}: {reason}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    seed_df = collect_runs(input_root, allow_incomplete=args.allow_incomplete)
    consistency_messages = validate_run_consistency(seed_df)
    summary_df = build_summary(seed_df, precision=args.precision)

    seed_csv = output_dir / "weighting_seed_results.csv"
    summary_csv = output_dir / "weighting_summary.csv"
    summary_md = output_dir / "weighting_summary.md"
    selection_json = output_dir / "weighting_selection.json"
    selection_txt = output_dir / "weighting_selection.txt"
    consistency_txt = output_dir / "weighting_consistency_check.txt"

    seed_df.to_csv(seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    write_markdown_table(summary_df, summary_md)

    selection = choose_weight(
        summary_df,
        min_acceptable_mean_dice=args.min_acceptable_mean_dice,
        manual_rejections=args.reject_weight,
    )
    with selection_json.open("w") as f:
        json.dump(selection, f, indent=2)
    write_selection_text(selection, summary_df, selection_txt)
    consistency_txt.write_text("\n".join(consistency_messages) + "\n")

    print("\nStage 1 Step 2 summary written to:")
    for path in (
        seed_csv,
        summary_csv,
        summary_md,
        selection_json,
        selection_txt,
        consistency_txt,
    ):
        print(f"  {path}")

    print("\nProfessor-format validation table:")
    display_cols = [
        "weight",
        "val_mae_mean_sd",
        "val_rmse_mean_sd",
        "val_r2_mean_sd",
        "val_mean_dice_mean_sd",
        "val_mean_hd95_mean_sd",
    ]
    print(summary_df[display_cols].to_string(index=False))

    print("\nSelection status:")
    print(json.dumps(selection, indent=2))

    if selection.get("complete") and selection.get("recommended_weight"):
        if selection.get("locked_weight"):
            print(f"\nLocked weight: {selection['locked_weight']}")
        else:
            print(
                f"\nRecommended by numeric rule: {selection['recommended_weight']}\n"
                "Manual segmentation review is still required before you call it locked."
            )


if __name__ == "__main__":
    main()