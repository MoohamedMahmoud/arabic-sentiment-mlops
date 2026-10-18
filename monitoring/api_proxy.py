"""Observability proxy in front of the existing BentoML service.

It records request metrics and aggregate prediction events only; it never persists raw text.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

BENTO_BASE_URL = os.getenv("BENTO_BASE_URL", "http://host.docker.internal:3001").rstrip(
    "/"
)
EVENTS_DB = Path(os.getenv("EVENTS_DB", "/data/predictions.sqlite3"))
REFERENCE_PATH = Path(
    os.getenv("REFERENCE_DISTRIBUTION_PATH", "/app/reference_distribution.json")
)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
RECENT_WINDOW = int(os.getenv("DRIFT_WINDOW_PREDICTIONS", "500"))
LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
LENGTH_BINS = ("0-20", "21-50", "51-100", "101-200", "201+")

HTTP_REQUESTS = Counter(
    "sentiment_http_requests_total",
    "HTTP requests reaching the observability proxy.",
    ["endpoint", "method", "status_code"],
)
HTTP_LATENCY = Histogram(
    "sentiment_http_request_duration_seconds",
    "End-to-end proxy request duration in seconds, including upstream inference.",
    ["endpoint"],
    buckets=(0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0, 120.0),
)
PREDICTIONS = Counter(
    "sentiment_predictions_total",
    "Successful predictions grouped by predicted label.",
    ["label"],
)
CONFIDENCE = Histogram(
    "sentiment_prediction_confidence",
    "Distribution of returned top-class confidence values.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
INPUT_LENGTH = Histogram(
    "sentiment_input_text_characters",
    "Length of incoming texts in Unicode characters (raw text is not stored).",
    buckets=(10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000),
)
UPSTREAM_HEALTHY = Gauge(
    "sentiment_upstream_healthy",
    "1 when BentoML /readyz is reachable and healthy, otherwise 0.",
)
REFERENCE_AVAILABLE = Gauge(
    "sentiment_drift_reference_available",
    "1 when an evaluated reference distribution is configured.",
)
LABEL_PSI = Gauge(
    "sentiment_prediction_distribution_psi",
    "PSI between reference and recent predicted label distributions.",
)
LENGTH_PSI = Gauge(
    "sentiment_text_length_psi",
    "PSI between reference and recent input text-length distributions.",
)
RECENT_SAMPLES = Gauge(
    "sentiment_drift_window_predictions",
    "Number of recent prediction events used for drift calculations.",
)


class PredictionRequest(BaseModel):
    texts: list[str]


def _connect() -> sqlite3.Connection:
    EVENTS_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(EVENTS_DB, timeout=10)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS prediction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time REAL NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            text_length INTEGER NOT NULL
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_events_time ON prediction_events(event_time)"
    )
    connection.commit()
    return connection


def _length_bin(length: int) -> str:
    if length <= 20:
        return "0-20"
    if length <= 50:
        return "21-50"
    if length <= 100:
        return "51-100"
    if length <= 200:
        return "101-200"
    return "201+"


def _load_reference() -> dict[str, Any] | None:
    try:
        raw = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        labels = raw.get("label_distribution")
        lengths = raw.get("text_length_bin_distribution")
        if not isinstance(labels, dict) or not isinstance(lengths, dict):
            return None
        if any(label not in labels for label in LABELS) or any(
            name not in lengths for name in LENGTH_BINS
        ):
            return None
        label_total = sum(float(labels[name]) for name in LABELS)
        length_total = sum(float(lengths[name]) for name in LENGTH_BINS)
        if label_total <= 0 or length_total <= 0:
            return None
        return {
            "label_distribution": {
                name: float(labels[name]) / label_total for name in LABELS
            },
            "text_length_bin_distribution": {
                name: float(lengths[name]) / length_total for name in LENGTH_BINS
            },
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _psi(
    reference: dict[str, float],
    current_counts: dict[str, int],
    categories: tuple[str, ...],
) -> float:
    epsilon = 1e-4
    total = sum(current_counts.get(category, 0) for category in categories)
    if total <= 0:
        return 0.0
    actual = {
        category: (current_counts.get(category, 0) + epsilon) for category in categories
    }
    actual_total = sum(actual.values())
    actual = {category: count / actual_total for category, count in actual.items()}
    ref = {
        category: max(float(reference.get(category, 0.0)), epsilon)
        for category in categories
    }
    ref_total = sum(ref.values())
    ref = {category: value / ref_total for category, value in ref.items()}
    return max(
        0.0,
        sum((actual[c] - ref[c]) * math.log(actual[c] / ref[c]) for c in categories),
    )


def _record_and_update(events: list[dict[str, Any]]) -> None:
    with closing(_connect()) as connection, connection:
        connection.executemany(
            "INSERT INTO prediction_events(event_time, label, confidence, text_length) VALUES (?, ?, ?, ?)",
            [
                (
                    time.time(),
                    item["label"],
                    item["confidence"],
                    item["text_length"],
                )
                for item in events
            ],
        )
        connection.execute(
            "DELETE FROM prediction_events WHERE id NOT IN (SELECT id FROM prediction_events ORDER BY id DESC LIMIT 50000)"
        )
        rows = connection.execute(
            "SELECT label, text_length FROM prediction_events ORDER BY id DESC LIMIT ?",
            (RECENT_WINDOW,),
        ).fetchall()
    RECENT_SAMPLES.set(len(rows))
    reference = _load_reference()
    if reference is None:
        REFERENCE_AVAILABLE.set(0)
        LABEL_PSI.set(0)
        LENGTH_PSI.set(0)
        return
    REFERENCE_AVAILABLE.set(1)
    label_counts = {label: 0 for label in LABELS}
    length_counts = {name: 0 for name in LENGTH_BINS}
    for label, text_length in rows:
        if label in label_counts:
            label_counts[label] += 1
        length_counts[_length_bin(int(text_length))] += 1
    if len(rows) < 30:
        LABEL_PSI.set(0)
        LENGTH_PSI.set(0)
        return
    LABEL_PSI.set(_psi(reference["label_distribution"], label_counts, LABELS))
    LENGTH_PSI.set(
        _psi(reference["text_length_bin_distribution"], length_counts, LENGTH_BINS)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    with closing(_connect()):
        pass
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="Arabic Sentiment Observability Proxy", version="1.0.0", lifespan=lifespan
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    path = request.url.path
    if path.startswith("/metrics"):
        return await call_next(request)
    endpoint = (
        path if path in {"/predict", "/health", "/readyz", "/metadata"} else "other"
    )
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(
            endpoint=endpoint, method=request.method, status_code=str(status)
        ).inc()
        HTTP_LATENCY.labels(endpoint=endpoint).observe(elapsed)


@app.get("/health")
@app.get("/readyz")
async def health(request: Request):
    try:
        response = await request.app.state.http.get(f"{BENTO_BASE_URL}/readyz")
        ok = response.is_success
    except httpx.HTTPError:
        ok = False
    UPSTREAM_HEALTHY.set(1 if ok else 0)
    if not ok:
        return JSONResponse(
            status_code=503, content={"status": "degraded", "upstream_ok": False}
        )
    return {"status": "ok", "upstream_ok": True}


@app.get("/metadata")
async def metadata(request: Request):
    try:
        response = await request.app.state.http.post(
            f"{BENTO_BASE_URL}/metadata",
            json={},
        )
        response.raise_for_status()
        UPSTREAM_HEALTHY.set(1)
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        UPSTREAM_HEALTHY.set(0)
        raise HTTPException(
            status_code=502,
            detail=f"BentoML metadata is unavailable: {exc}",
        ) from exc


@app.post("/predict")
async def predict(payload: PredictionRequest, request: Request):
    if not payload.texts:
        raise HTTPException(status_code=422, detail="Provide at least one text.")
    if len(payload.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"This gateway accepts at most {MAX_BATCH_SIZE} texts per request.",
        )
    texts = [text.strip() for text in payload.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=422, detail="Texts must not be empty.")
    try:
        response = await request.app.state.http.post(
            f"{BENTO_BASE_URL}/predict", json={"texts": texts}
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        UPSTREAM_HEALTHY.set(0 if exc.response.status_code >= 500 else 1)
        raise HTTPException(
            status_code=502, detail=f"BentoML returned HTTP {exc.response.status_code}."
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        UPSTREAM_HEALTHY.set(0)
        raise HTTPException(
            status_code=502, detail=f"Could not reach BentoML: {exc}"
        ) from exc
    predictions = body.get("predictions") if isinstance(body, dict) else body
    if isinstance(body, dict) and isinstance(body.get("value"), list):
        predictions = body["value"]
    if not isinstance(predictions, list) or len(predictions) != len(texts):
        raise HTTPException(
            status_code=502, detail="BentoML returned an unexpected prediction payload."
        )
    events = []
    clean_predictions = []
    for text, item in zip(texts, predictions, strict=True):
        try:
            label = str(item["label"]).upper()
            confidence = float(item["confidence"])
            if label not in LABELS or not 0.0 <= confidence <= 1.0:
                raise ValueError("Unexpected label or confidence.")
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail="BentoML returned an invalid prediction row."
            ) from exc
        PREDICTIONS.labels(label=label).inc()
        CONFIDENCE.observe(confidence)
        INPUT_LENGTH.observe(len(text))
        events.append(
            {"label": label, "confidence": confidence, "text_length": len(text)}
        )
        clean_predictions.append({"label": label, "confidence": confidence})
    await asyncio.to_thread(_record_and_update, events)
    UPSTREAM_HEALTHY.set(1)
    return {"predictions": clean_predictions}
