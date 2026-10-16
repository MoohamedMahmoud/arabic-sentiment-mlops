"""Production prediction through the MLflow Model Registry."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from .config import settings

from .mlflow_utils import get_model_version_by_alias


class ProductionPredictor:
    """Load one registered model once and use it for all requests."""

    def __init__(self, model_uri: str, model_name: str, alias: str) -> None:
        self.model_uri = model_uri
        self.model_name = model_name
        self.alias = alias
        # Use the configured Docker-based MLflow tracking server.
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        self.model_version = get_model_version_by_alias(model_name, alias)
        self.model = mlflow.pyfunc.load_model(model_uri)

    @property
    def loaded(self) -> bool:
        """Return True when the model is loaded."""

        return self.model is not None

    def predict_one(self, text: str) -> dict[str, Any]:
        """Return one prediction."""

        frame = pd.DataFrame({"text": [text]})
        prediction = self.model.predict(frame).iloc[0].to_dict()

        return {
            "label": str(prediction["label"]),
            "confidence": float(prediction["confidence"]),
            "model_version": str(self.model_version.version),
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return predictions for multiple reviews."""

        frame = pd.DataFrame({"text": texts})
        predictions = self.model.predict(frame)

        return [
            {
                "label": str(row["label"]),
                "confidence": float(row["confidence"]),
                "model_version": str(self.model_version.version),
            }
            for _, row in predictions.iterrows()
        ]
