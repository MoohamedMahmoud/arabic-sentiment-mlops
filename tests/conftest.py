"""Shared lightweight test fixtures for the API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient


class DummyProductionPredictor:
    """Small fake predictor; tests do not need MLflow or a trained model."""

    def __init__(self, model_uri: str, model_name: str, alias: str) -> None:
        self.model_uri = model_uri
        self.model_name = model_name
        self.alias = alias
        self.model_version = SimpleNamespace(version="test-version-1")
        self.loaded = True

    @staticmethod
    def _prediction(text: str) -> dict[str, Any]:
        """Return a fixed prediction while retaining the given input path."""

        return {
            "label": "POSITIVE",
            "confidence": 0.91,
            "model_version": "test-version-1",
        }

    def predict_one(self, text: str) -> dict[str, Any]:
        return self._prediction(text)

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self._prediction(text) for text in texts]


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    """Create a test client whose startup loads a fake rather than a real model."""

    from arabic_sentiment_mlops.api import main as api_main

    monkeypatch.setattr(api_main, "ProductionPredictor", DummyProductionPredictor)

    with TestClient(api_main.app) as client:
        yield client
