#!/usr/bin/env python3

"""
Create binary reduced-EF classification labels from EchoNet FileList.csv.

Class definition:
    0 = EF >= threshold
    1 = EF < threshold
"""

import argparse
from pathlib import Path

import pandas as pd
from echonet.paths import (
    FILE_LIST_PATH,
    LABELS_OUTPUT_DIR,
)


DEFAULT_INPUT_PATH = FILE_LIST_PATH

DEFAULT_OUTPUT_PATH = (
    LABELS_OUTPUT_DIR
    / "ef_classification_labels.csv"
)


def create_ef_labels(
    csv_path: Path,
    output_path: Path,
    ef_column: str = "EF",
    id_column: str = "FileName",
    threshold: float = 40.0,
) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"FileList CSV was not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {id_column, ef_column}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "FileList.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    output = df[[id_column, ef_column]].copy()

    output[ef_column] = pd.to_numeric(
        output[ef_column],
        errors="coerce",
    )

    missing_ef_count = int(output[ef_column].isna().sum())

    if missing_ef_count:
        print(
            f"Removing {missing_ef_count} rows with missing or invalid EF."
        )
        output = output.dropna(subset=[ef_column]).copy()

    output["reduced_ef_label"] = (
        output[ef_column] < threshold
    ).astype("int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    class_counts = (
        output["reduced_ef_label"]
        .value_counts()
        .sort_index()
    )

    print(f"Input:  {csv_path}")
    print(f"Output: {output_path}")
    print(f"EF threshold: {threshold:g}")
    print(f"Total labeled videos: {len(output):,}")
    print(f"Normal/preserved EF, label 0: {class_counts.get(0, 0):,}")
    print(f"Reduced EF, label 1:     {class_counts.get(1, 0):,}")

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create binary EF classification labels."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to EchoNet FileList.csv.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated label CSV.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="EF below this value receives label 1.",
    )

    parser.add_argument(
        "--ef-column",
        default="EF",
        help="Name of the EF column.",
    )

    parser.add_argument(
        "--id-column",
        default="FileName",
        help="Name of the video identifier column.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    create_ef_labels(
        csv_path=args.input,
        output_path=args.output,
        ef_column=args.ef_column,
        id_column=args.id_column,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()