"""FastAPI request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Validated request for one review."""

    text: str = Field(..., min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    """Prediction response returned to the client."""

    label: Literal["NEGATIVE", "NEUTRAL", "POSITIVE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    correlation_id: str
    latency_ms: float = Field(..., ge=0.0)


class BatchPredictionRequest(BaseModel):
    """Validated request for multiple reviews."""

    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
