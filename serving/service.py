from __future__ import annotations

from typing import Any

import bentoml
import pandas as pd


BENTO_MODEL_TAG = "arabic_sentiment_v2:ui5zcfgkc2qxoaav"
MLFLOW_MODEL_NAME = "ArabicSentiment"
MLFLOW_MODEL_VERSION = 2
MODEL_MAX_LENGTH = 256

MAX_BATCH_SIZE = 8
MAX_LATENCY_MS = 2000


@bentoml.service(
    name="arabic-sentiment-service",
    traffic={"timeout": 120},
    workers=1,
)
class ArabicSentimentService:
    """Serve the pinned Arabic sentiment model using BentoML."""

    bento_model = bentoml.models.BentoModel(BENTO_MODEL_TAG)

    def __init__(self) -> None:
        # Load from BentoML's local model store, not from the MLflow server.
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(
        batchable=True,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def predict(self, texts: list[str]) -> list[dict[str, str | float]]:
        """Predict sentiment and confidence for one or more Arabic texts."""

        if not texts:
            raise ValueError("'texts' must contain at least one text.")

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"'texts[{index}]' must be a non-empty string."
                )
            cleaned_texts.append(text.strip())

        inputs = pd.DataFrame({"text": cleaned_texts})
        predictions = self.model.predict(inputs)

        return [
            {
                "label": str(row["label"]),
                "confidence": float(row["confidence"]),
            }
            for row in predictions.to_dict(orient="records")
        ]

    @bentoml.api
    def metadata(self) -> dict[str, Any]:
        """Return information about the model release being served."""

        return {
            "service": "arabic-sentiment-service",
            "registered_model": MLFLOW_MODEL_NAME,
            "mlflow_model_version": MLFLOW_MODEL_VERSION,
            "bento_model_tag": BENTO_MODEL_TAG,
            "max_length": MODEL_MAX_LENGTH,
            "adaptive_batching": {
                "max_batch_size": MAX_BATCH_SIZE,
                "max_latency_ms": MAX_LATENCY_MS,
            },
        }
