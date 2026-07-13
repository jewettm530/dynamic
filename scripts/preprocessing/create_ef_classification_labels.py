# utils/create_ef_classification_labels.py

import pandas as pd


def create_ef_labels(csv_path, output_path, ef_column="EF", id_column="FileName"):
    df = pd.read_csv(csv_path)

    df["reduced_ef_label"] = (df[ef_column] < 40).astype(int)

    keep_cols = [id_column, ef_column, "reduced_ef_label"]
    df[keep_cols].to_csv(output_path, index=False)

    print(f"Saved labels to {output_path}")
    print(df["reduced_ef_label"].value_counts())


if __name__ == "__main__":
    create_ef_labels(
        csv_path="FileList.csv",
        output_path="outputs/ef_classification_labels.csv",
        ef_column="EF",
        id_column="FileName"
    )