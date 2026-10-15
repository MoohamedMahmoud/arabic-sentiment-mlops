"""Tests for API request/response validation."""

import pytest
from pydantic import ValidationError

from arabic_sentiment_mlops.api.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)


def test_prediction_request_accepts_nonempty_text() -> None:
    request = PredictionRequest(text="المنتج ممتاز")

    assert request.text == "المنتج ممتاز"


@pytest.mark.parametrize("text", ["", "x" * 5001])
def test_prediction_request_rejects_empty_or_oversized_text(text: str) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(text=text)


def test_prediction_response_rejects_confidence_outside_zero_to_one() -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(
            label="POSITIVE",
            confidence=1.1,
            model_version="1",
            correlation_id="test-correlation-id",
            latency_ms=1.0,
        )


def test_batch_request_requires_at_least_one_text() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest(texts=[])
