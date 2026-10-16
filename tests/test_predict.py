"""Tests for the production predictor with MLflow calls mocked."""

from types import SimpleNamespace

import pandas as pd

from arabic_sentiment_mlops import predict as predict_module


class FakePyFuncModel:
    """Fake MLflow model that returns predictable predictions."""

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "label": ["POSITIVE"] * len(frame),
                "confidence": [0.9] * len(frame),
            }
        )


def configure_mlflow_mocks(monkeypatch) -> dict[str, object]:
    """Prevent tests from contacting the real MLflow registry or server."""

    fake_model = FakePyFuncModel()
    calls: dict[str, object] = {}

    def fake_load_model(uri: str) -> FakePyFuncModel:
        calls["uri"] = uri
        return fake_model

    def fake_set_tracking_uri(uri: str) -> None:
        calls["tracking_uri"] = uri

    def fake_get_model_version_by_alias(
        model_name: str,
        alias: str,
    ) -> SimpleNamespace:
        calls["model_name"] = model_name
        calls["alias"] = alias
        return SimpleNamespace(version="17")

    monkeypatch.setattr(
        predict_module,
        "get_model_version_by_alias",
        fake_get_model_version_by_alias,
    )

    # Mock MLflow so no real registry, artifact store, or server is contacted.
    monkeypatch.setattr(
        predict_module,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=fake_set_tracking_uri,
            pyfunc=SimpleNamespace(load_model=fake_load_model),
        ),
    )

    return calls


def test_predictor_loads_registry_version_and_returns_expected_fields(
    monkeypatch,
) -> None:
    calls = configure_mlflow_mocks(monkeypatch)

    predictor = predict_module.ProductionPredictor(
        model_uri="models:/ArabicSentiment@production",
        model_name="ArabicSentiment",
        alias="production",
    )

    prediction = predictor.predict_one("المنتج ممتاز")

    assert calls["tracking_uri"] == predict_module.settings.mlflow_tracking_uri
    assert calls["model_name"] == "ArabicSentiment"
    assert calls["alias"] == "production"
    assert calls["uri"] == "models:/ArabicSentiment@production"

    assert predictor.loaded is True
    assert prediction == {
        "label": "POSITIVE",
        "confidence": 0.9,
        "model_version": "17",
    }


def test_predictor_returns_one_prediction_per_input(monkeypatch) -> None:
    configure_mlflow_mocks(monkeypatch)

    predictor = predict_module.ProductionPredictor(
        model_uri="models:/ArabicSentiment@production",
        model_name="ArabicSentiment",
        alias="production",
    )

    predictions = predictor.predict_batch(
        ["نص أول", "نص ثان"]
    )

    assert len(predictions) == 2
    assert all(item["model_version"] == "17" for item in predictions)
    assert all(item["label"] == "POSITIVE" for item in predictions)
    assert all(item["confidence"] == 0.9 for item in predictions)