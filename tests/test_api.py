"""API integration tests using the fake predictor defined in conftest.py."""


def test_health_endpoint_reports_loaded_model(api_client) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metadata_endpoint_reports_registered_model(api_client) -> None:
    response = api_client.get("/metadata")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "ArabicSentiment",
        "model_version": "test-version-1",
        "model_alias": "production",
    }


def test_predict_endpoint_returns_schema_and_correlation_id(api_client) -> None:
    response = api_client.post("/predict", json={"text": "المنتج ممتاز"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "POSITIVE"
    assert payload["confidence"] == 0.91
    assert payload["model_version"] == "test-version-1"
    assert payload["correlation_id"]
    assert payload["latency_ms"] >= 0
    assert response.headers.get("X-Correlation-ID") == payload["correlation_id"]


def test_predict_endpoint_rejects_empty_text(api_client) -> None:
    response = api_client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_batch_endpoint_returns_one_result_per_text(api_client) -> None:
    response = api_client.post(
        "/predict/batch", json={"texts": ["المنتج ممتاز", "الخدمة جيدة"]}
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
