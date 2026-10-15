"""Tests for cleaning, splitting, and leakage prevention."""

import pandas as pd
import pytest

from arabic_sentiment_mlops.prepare import (
    canonical_text,
    clean_data,
    split_data,
    verify_no_overlap,
)


def test_canonical_text_removes_diacritics_tatweel_and_extra_whitespace() -> None:
    assert canonical_text("  مُمتازـ   جدا  ") == "ممتاز جدا"


def test_clean_data_applies_the_documented_cleaning_policy() -> None:
    raw = pd.DataFrame(
        {
            "text": [
                "  رائعُ   المنتج ",
                "رائع المنتج",
                "سيء المنتج",
                "سيء المنتج ",
                "تقييم بلا تصنيف",
                "تعليق بلا تصنيف",
            ],
            "label": ["positive", "POSITIVE", "NEGATIVE", "positive", "HAPPY", None],
        }
    )

    clean, summary = clean_data(raw)

    assert clean.to_dict(orient="records") == [
        {"text": "رائعُ   المنتج", "label": "POSITIVE"}
    ]
    assert summary["rows_in"] == 6
    assert summary["dropped_missing_label"] == 1
    assert summary["dropped_unknown_label"] == 1
    assert summary["conflicting_canonical_groups"] == 1
    assert summary["dropped_conflicting_rows"] == 2
    assert summary["dropped_duplicate_rows"] == 1
    assert summary["rows_after_cleaning"] == 1


def test_clean_data_rejects_blank_review_text() -> None:
    raw = pd.DataFrame({"text": ["   "], "label": ["POSITIVE"]})

    with pytest.raises(ValueError, match="missing or blank review text"):
        clean_data(raw)


def test_split_data_is_deterministic_stratified_and_leakage_free() -> None:
    raw = pd.DataFrame(
        {
            "text": [
                f"review {label.lower()} {i}"
                for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"]
                for i in range(30)
            ],
            "label": [
                label
                for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"]
                for _ in range(30)
            ],
        }
    )

    first = split_data(raw, seed=42)
    second = split_data(raw, seed=42)

    assert set(first) == {"train", "validation", "test"}
    assert sum(len(frame) for frame in first.values()) == len(raw)
    assert [len(first[name]) for name in ("train", "validation", "test")] == [72, 9, 9]

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
        assert set(first[name]["label"]) == {"NEGATIVE", "NEUTRAL", "POSITIVE"}

    verify_no_overlap(first)


def test_verify_no_overlap_fails_if_text_appears_in_two_splits() -> None:
    splits = {
        "train": pd.DataFrame({"text": ["نص مكرر"], "label": ["POSITIVE"]}),
        "validation": pd.DataFrame({"text": ["نص مكرر"], "label": ["POSITIVE"]}),
        "test": pd.DataFrame({"text": ["نص مختلف"], "label": ["NEGATIVE"]}),
    }

    with pytest.raises(
        ValueError, match="Data leakage detected between train and validation"
    ):
        verify_no_overlap(splits)
