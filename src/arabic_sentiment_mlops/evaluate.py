"""Evaluation helpers used by training and reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import ID2LABEL


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    """Return Accuracy and Macro-F1 for a Hugging Face Trainer evaluation."""

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


def classification_report_dict(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Return the full per-class classification report."""

    return classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in [0, 1, 2]],
        output_dict=True,
        digits=4,
        zero_division=0,
    )


def confusion_matrix_frame(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Return a labeled confusion matrix DataFrame."""

    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])

    return pd.DataFrame(
        matrix,
        index=[ID2LABEL[i] for i in [0, 1, 2]],
        columns=[ID2LABEL[i] for i in [0, 1, 2]],
    )
