"""Import the approved MLflow model into the BentoML model store."""

from __future__ import annotations

import bentoml
import mlflow
from mlflow import MlflowClient

from arabic_sentiment_mlops.config import settings


def main() -> None:
    tracking_uri = settings.mlflow_tracking_uri

    # Explicitly use the Docker-based MLflow server, not a local SQLite registry.
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolve the alias once, then pin this import to the exact model version.
    version = client.get_model_version_by_alias(
        settings.registered_model_name,
        settings.production_alias,
    )

    model_name = version.name
    model_version = str(version.version)
    model_uri = f"models:/{model_name}/{model_version}"
    bento_model_name = f"arabic_sentiment_v{model_version}"

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Registered model: {model_name}")
    print(f"Resolved alias: {settings.production_alias}")
    print(f"Exact version: {model_version}")
    print(f"Pinned model URI: {model_uri}")
    print(f"BentoML model name: {bento_model_name}")

    # Avoid importing and downloading the same release again.
    for existing in bentoml.models.list():
        if existing.tag.name == bento_model_name:
            print(f"BentoML model already exists: {existing.tag}")
            print("Keeping the existing local model; no import performed.")
            return

    print("\nImporting the MLflow artifact into BentoML. This can take several minutes.")

    imported = bentoml.mlflow.import_model(
        name=bento_model_name,
        model_uri=model_uri,
        signatures={
            "predict": {"batchable": True},
        },
        labels={
            "source": "mlflow",
            "mlflow_model_name": model_name,
            "mlflow_model_version": model_version,
        },
        metadata={
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_alias_at_import": settings.production_alias,
            "mlflow_model_version": model_version,
            "mlflow_run_id": str(version.run_id or ""),
        },
    )

    print("\nModel import completed.")
    print(f"BentoML model tag: {imported.tag}")
    print(f"MLflow source URI: {model_uri}")


if __name__ == "__main__":
    main()
