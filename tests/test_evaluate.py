"""Tests for the model evaluation helpers."""

import numpy as np

from arabic_sentiment_mlops.evaluate import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_frame,
)


def test_compute_metrics_returns_accuracy_and_macro_f1() -> None:
    labels = np.array([0, 1, 2])
    logits = np.array(
        [
            [9.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 9.0, 0.0],
        ]
    )

    metrics = compute_metrics((logits, labels))

    assert set(metrics) == {"accuracy", "f1_macro"}
    assert np.isclose(metrics["accuracy"], 2 / 3)
    assert np.isclose(metrics["f1_macro"], (1 + 2 / 3 + 0) / 3)


def test_classification_report_has_all_three_classes() -> None:
    report = classification_report_dict(
        labels=np.array([0, 1, 2]),
        predictions=np.array([0, 2, 2]),
    )

    assert {"NEGATIVE", "NEUTRAL", "POSITIVE"}.issubset(report)
    assert "macro avg" in report


def test_confusion_matrix_frame_labels_rows_and_columns() -> None:
    matrix = confusion_matrix_frame(
        labels=np.array([0, 1, 2]),
        predictions=np.array([0, 1, 1]),
    )

    assert matrix.index.tolist() == ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    assert matrix.columns.tolist() == ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    assert int(matrix.to_numpy().sum()) == 3
