"""Tests for raw-data reading and processed split validation."""

import pandas as pd
import pytest

from arabic_sentiment_mlops.data import read_raw_alldata, validate_processed_split


def test_read_raw_alldata_reads_headerless_two_column_csv(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "المنتج ممتاز,POSITIVE\nالخدمة سيئة,NEGATIVE\n", encoding="utf-8"
    )

    frame = read_raw_alldata(raw_path)

    assert frame.columns.tolist() == ["text", "label"]
    assert frame.to_dict(orient="records") == [
        {"text": "المنتج ممتاز", "label": "POSITIVE"},
        {"text": "الخدمة سيئة", "label": "NEGATIVE"},
    ]


def test_read_raw_alldata_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        read_raw_alldata(tmp_path / "absent.csv")


def test_validate_processed_split_accepts_valid_data() -> None:
    frame = pd.DataFrame(
        {"text": ["منتج جيد", "خدمة عادية"], "label": ["POSITIVE", "NEUTRAL"]}
    )

    validate_processed_split(frame, "train")


def test_validate_processed_split_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"text": ["نص"], "sentiment": ["POSITIVE"]})

    with pytest.raises(ValueError, match="missing columns"):
        validate_processed_split(frame, "train")


def test_validate_processed_split_rejects_missing_text() -> None:
    frame = pd.DataFrame({"text": [None], "label": ["POSITIVE"]})

    with pytest.raises(ValueError, match="missing text"):
        validate_processed_split(frame, "validation")


def test_validate_processed_split_rejects_unknown_label() -> None:
    frame = pd.DataFrame({"text": ["نص"], "label": ["MIXED"]})

    with pytest.raises(ValueError, match="unknown labels"):
        validate_processed_split(frame, "test")
