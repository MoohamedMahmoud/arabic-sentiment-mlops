"""Create the deterministic, leakage-free Arabic sentiment dataset.

This file contains the actual data-cleaning logic that was previously explored
in the data-understanding notebook. DVC calls this file as the ``prepare``
stage; DVC does not replace this logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import read_raw_alldata

LOGGER = logging.getLogger(__name__)

VALID_LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")

# Used only to detect duplicates and leakage.
_WHITESPACE_RE = re.compile(r"\s+")
_ARABIC_MARKS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def canonical_text(text: str) -> str:
    """Build the duplicate-detection key without changing model input text."""

    text_without_marks = _ARABIC_MARKS_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text_without_marks).strip()


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the project's conservative cleaning policy.

    The policy is:
    1. Drop rows with missing labels.
    2. Remove every canonical-text group with conflicting labels.
    3. Keep one occurrence of each remaining canonical text.

    The original review text remains unchanged apart from stripping leading
    and trailing whitespace used for the stored text field.
    """

    data = frame.copy()

    rows_in = len(data)
    data = data.dropna(subset=["label"]).copy()

    data["text"] = data["text"].astype("string").str.strip()
    data["label"] = data["label"].astype("string").str.strip().str.upper()

    unknown_mask = ~data["label"].isin(VALID_LABELS)
    dropped_unknown = int(unknown_mask.sum())
    data = data.loc[~unknown_mask].copy()

    # Missing/blank text was not present in the analyzed corpus. Treat it as a
    # data error instead of silently inventing a cleaning rule.
    if data["text"].isna().any() or data["text"].eq("").any():
        raise ValueError("Raw data contains missing or blank review text.")

    data["canonical_text"] = data["text"].map(canonical_text)

    label_counts = data.groupby("canonical_text")["label"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)

    conflicting_mask = data["canonical_text"].isin(conflicting_keys)
    dropped_conflicts = int(conflicting_mask.sum())
    data = data.loc[~conflicting_mask].copy()

    before_dedup = len(data)
    data = data.drop_duplicates(subset=["canonical_text"], keep="first")
    dropped_duplicates = before_dedup - len(data)

    result = data[["text", "label"]].reset_index(drop=True)

    summary = {
        "rows_in": rows_in,
        "dropped_missing_label": rows_in - int(frame["label"].notna().sum()),
        "dropped_unknown_label": dropped_unknown,
        "conflicting_canonical_groups": len(conflicting_keys),
        "dropped_conflicting_rows": dropped_conflicts,
        "dropped_duplicate_rows": dropped_duplicates,
        "rows_after_cleaning": len(result),
    }

    return result, summary


def split_data(
    clean_frame: pd.DataFrame,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create a fresh stratified 80/10/10 train/validation/test split."""

    train_val, test = train_test_split(
        clean_frame,
        test_size=0.10,
        random_state=seed,
        stratify=clean_frame["label"],
    )

    train, validation = train_test_split(
        train_val,
        test_size=1 / 9,
        random_state=seed,
        stratify=train_val["label"],
    )

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    verify_no_overlap(splits)
    return splits


def verify_no_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the same canonical review appears in two splits."""

    keys = {
        name: set(frame["text"].map(canonical_text)) for name, frame in splits.items()
    }

    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]

    for left, right in pairs:
        overlap = keys[left].intersection(keys[right])
        if overlap:
            raise ValueError(
                f"Data leakage detected between {left} and {right}: "
                f"{len(overlap)} overlapping canonical texts."
            )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_data_card(
    raw_path: Path,
    clean_summary: dict[str, int],
    splits: dict[str, pd.DataFrame],
    seed: int,
) -> dict[str, object]:
    """Build a small provenance record for DVC/MLflow lineage."""

    return {
        "raw_file": raw_path.as_posix(),
        "raw_file_sha256": sha256_file(raw_path),
        "seed": seed,
        "labels": list(VALID_LABELS),
        "cleaning": clean_summary,
        "splits": {
            name: {
                "rows": len(frame),
                "label_counts": frame["label"].value_counts().to_dict(),
            }
            for name, frame in splits.items()
        },
    }


def prepare_dataset(
    raw_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Run the complete preparation pipeline and write the processed files."""

    raw = read_raw_alldata(raw_path)
    clean, clean_summary = clean_data(raw)

    splits = split_data(clean, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    splits["train"].to_parquet(output_dir / "train.parquet", index=False)
    splits["validation"].to_parquet(output_dir / "val.parquet", index=False)
    splits["test"].to_parquet(output_dir / "test.parquet", index=False)

    card = build_data_card(raw_path, clean_summary, splits, seed)
    (output_dir / "data_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Prepared rows: %s", len(clean))
    LOGGER.info(
        "Split sizes: train=%s validation=%s test=%s",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    LOGGER.info("Prepared data written to %s", output_dir)

    return card


def main() -> None:
    """Run the preparation pipeline from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/alldata.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_dataset(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
