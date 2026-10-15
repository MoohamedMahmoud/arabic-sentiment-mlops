"""Small data I/O helpers used by the training and preparation pipelines."""

from pathlib import Path

import pandas as pd

LABEL_COLUMN = "label"
TEXT_COLUMN = "text"


def read_raw_alldata(path: Path) -> pd.DataFrame:
    """Read this project's headerless two-column raw corpus."""

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    frame = pd.read_csv(
        path,
        header=None,
        names=[TEXT_COLUMN, LABEL_COLUMN],
        encoding="utf-8-sig",
        dtype={TEXT_COLUMN: "string", LABEL_COLUMN: "string"},
    )

    return frame


def load_processed_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the DVC-produced train, validation, and test Parquet files."""

    filenames = {
        "train": "train.parquet",
        "validation": "val.parquet",
        "test": "test.parquet",
    }

    splits: dict[str, pd.DataFrame] = {}

    for split_name, filename in filenames.items():
        path = processed_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Processed split not found: {path}")

        frame = pd.read_parquet(path)
        validate_processed_split(frame, split_name)
        splits[split_name] = frame

    return splits


def validate_processed_split(frame: pd.DataFrame, split_name: str) -> None:
    """Validate the simple schema expected by the training pipeline."""

    expected_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = expected_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} split is missing columns: {sorted(missing_columns)}"
        )

    if frame[TEXT_COLUMN].isna().any():
        raise ValueError(f"{split_name} split contains missing text values.")

    if frame[LABEL_COLUMN].isna().any():
        raise ValueError(f"{split_name} split contains missing labels.")

    labels = set(frame[LABEL_COLUMN].astype(str).str.strip().str.upper())
    allowed = {"NEGATIVE", "NEUTRAL", "POSITIVE"}
    unknown = labels.difference(allowed)

    if unknown:
        raise ValueError(
            f"{split_name} split contains unknown labels: {sorted(unknown)}"
        )
