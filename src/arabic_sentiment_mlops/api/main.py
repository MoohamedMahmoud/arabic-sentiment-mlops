"""FastAPI application for Arabic sentiment prediction."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..predict import ProductionPredictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registered production model once when the API starts."""

    app.state.predictor = ProductionPredictor(
        model_uri=settings.production_model_uri,
        model_name=settings.registered_model_name,
        alias=settings.production_alias,
    )
    yield


app = FastAPI(
    title="Arabic Sentiment API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Add a correlation ID to each request and response."""

    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Return a clean 500 response without exposing a traceback."""

    LOGGER.exception("Unexpected API error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "correlation_id": request.state.correlation_id,
        },
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Return healthy only when the model is loaded in memory."""

    predictor = request.app.state.predictor

    if not predictor.loaded:
        return JSONResponse(status_code=503, content={"status": "unhealthy"})

    return {"status": "healthy"}


@app.get("/metadata")
def metadata(request: Request) -> dict[str, str]:
    """Return the currently loaded model metadata."""

    predictor = request.app.state.predictor

    return {
        "model_name": predictor.model_name,
        "model_version": str(predictor.model_version.version),
        "model_alias": predictor.alias,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """Predict the sentiment of one Arabic review."""

    start = time.perf_counter()
    predictor = request.app.state.predictor
    result = predictor.predict_one(payload.text)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        **result,
        correlation_id=request.state.correlation_id,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict the sentiment of multiple Arabic reviews."""

    start = time.perf_counter()
    predictor = request.app.state.predictor
    results = predictor.predict_batch(payload.texts)
    latency_ms = (time.perf_counter() - start) * 1000

    per_item_latency = latency_ms / len(results)

    predictions = [
        PredictionResponse(
            **result,
            correlation_id=request.state.correlation_id,
            latency_ms=per_item_latency,
        )
        for result in results
    ]

    return BatchPredictionResponse(predictions=predictions)
