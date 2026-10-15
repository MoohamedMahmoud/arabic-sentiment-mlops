"""Register the best locally saved Arabic sentiment model in MLflow."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from .config import settings
from .mlflow_model import ArabicSentimentMLflowModel
from .mlflow_utils import configure_mlflow

# Only files required to load the model and tokenizer for inference.
# In particular, do not package the training checkpoints.
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _has_model_files(model_dir: Path) -> bool:
    """Return whether a directory contains the required inference files."""

    return model_dir.is_dir() and all(
        (model_dir / filename).is_file() for filename in REQUIRED_MODEL_FILES
    )


def _model_dir_for_run(run: Run) -> Path:
    """Resolve the local model directory associated with an MLflow run."""

    tagged_path = run.data.tags.get("local_model_path")
    if tagged_path:
        tagged_dir = Path(tagged_path)
        if _has_model_files(tagged_dir):
            return tagged_dir

    run_name = run.data.tags.get("mlflow.runName", "")

    candidates = [
        settings.models_dir / "runs" / run_name,
        settings.models_dir / run_name,
    ]

    for candidate in candidates:
        if _has_model_files(candidate):
            return candidate

    return candidates[0]


def _select_best_run(
    client: MlflowClient,
    experiment_id: str,
) -> tuple[Run, Path, float]:
    """Select the highest-scoring run with complete local model files."""

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.val_f1_macro DESC"],
        max_results=1000,
    )

    candidates: list[tuple[Run, Path, float]] = []

    for run in runs:
        score = run.data.metrics.get("val_f1_macro")
        if score is None:
            continue

        model_dir = _model_dir_for_run(run)
        if not _has_model_files(model_dir):
            continue

        candidates.append((run, model_dir, float(score)))

    if not candidates:
        raise RuntimeError(
            "No MLflow run with val_f1_macro and complete local model files was found."
        )

    return max(candidates, key=lambda item: item[2])


def _stage_inference_model(
    model_dir: Path,
    staging_dir: Path,
) -> Path:
    """Copy only inference files, excluding checkpoints and training files."""

    staging_dir.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_MODEL_FILES:
        source = model_dir / filename

        if not source.is_file():
            raise FileNotFoundError(f"Required inference file is missing: {source}")

        shutil.copy2(source, staging_dir / filename)

    return staging_dir


def main() -> None:
    """Register one selected model without retraining."""

    # Force the client to use a regular proxied upload rather than
    # MLflow's proxy multipart-upload path for this registration.
    # This setting does not change the Docker container configuration.
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        help="Select a training run explicitly instead of automatic selection.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the local model directory; requires --run-id.",
    )
    parser.add_argument(
        "--set-production",
        action="store_true",
        help="Assign the production alias after successful registration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow another registration if this source run was registered before.",
    )
    args = parser.parse_args()

    if args.model_path is not None and args.run_id is None:
        parser.error("--model-path requires --run-id")

    configure_mlflow(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )

    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
    )

    experiment = client.get_experiment_by_name(
        settings.mlflow_experiment_name,
    )
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {settings.mlflow_experiment_name}"
        )

    if args.run_id:
        source_run = client.get_run(args.run_id)
        model_dir = (
            args.model_path
            if args.model_path is not None
            else _model_dir_for_run(source_run)
        )

        score = source_run.data.metrics.get("val_f1_macro")
        if score is None:
            raise RuntimeError(f"Run {args.run_id} has no val_f1_macro metric.")

        score = float(score)
    else:
        source_run, model_dir, score = _select_best_run(
            client,
            experiment.experiment_id,
        )

    model_dir = model_dir.resolve()

    if not _has_model_files(model_dir):
        raise FileNotFoundError(
            f"Required inference files were not found in: {model_dir}"
        )

    source_run_name = source_run.data.tags.get(
        "mlflow.runName",
        source_run.info.run_id,
    )

    existing_registrations = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(f"tags.registration_source_run_id = '{source_run.info.run_id}'"),
        max_results=100,
    )

    already_registered = any(
        run.info.status == "FINISHED" for run in existing_registrations
    )

    if already_registered and not args.force:
        raise RuntimeError(
            "This source run already has a successful registration run. "
            "Use --force only if you intentionally want another version."
        )

    params = source_run.data.params
    model_name = params.get("model_name", settings.model_name)
    max_length = int(params.get("max_length", settings.max_length))

    print("Selected source run:", source_run.info.run_id)
    print("Source run status:", source_run.info.status)
    print("Source run name:", source_run_name)
    print("Validation macro-F1:", score)
    print("Local model path:", model_dir)
    print(
        "Proxy multipart upload:",
        os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"],
    )
    print("Only inference files will be packaged; checkpoints are excluded.")
    print("No training will be performed.")

    # Keep the temporary inference-only copy alive until MLflow has
    # finished packaging and logging the model.
    with tempfile.TemporaryDirectory(
        prefix="arabic_sentiment_model_"
    ) as temporary_directory:
        staging_dir = Path(temporary_directory) / "model_dir"

        _stage_inference_model(model_dir, staging_dir)

        with mlflow.start_run(
            run_name=f"register_best_{source_run_name}",
        ) as registration_run:
            registration_run_id = registration_run.info.run_id

            mlflow.set_tags(
                {
                    "run_type": "model_registration",
                    "registration_source_run_id": source_run.info.run_id,
                    "registration_source_run_name": source_run_name,
                    "local_model_path": str(model_dir),
                    "selection_metric": "val_f1_macro",
                    "proxy_multipart_upload": "false",
                }
            )
            mlflow.log_metric("selected_val_f1_macro", score)

            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=ArabicSentimentMLflowModel(),
                artifacts={"model_dir": str(staging_dir)},
                code_paths=[str(settings.project_root / "src")],
                registered_model_name=settings.registered_model_name,
                model_config={
                    "model_name": model_name,
                    "max_length": max_length,
                },
            )

    versions = client.search_model_versions(
        filter_string=(
            f"name = '{settings.registered_model_name}' "
            f"and run_id = '{registration_run_id}'"
        )
    )

    if not versions:
        raise RuntimeError(
            "The registration run completed, but no registered model "
            "version was found. Inspect the registration run in MLflow."
        )

    version = max(versions, key=lambda item: int(item.version))

    print("Logged model URI:", model_info.model_uri)
    print("Registered model:", version.name)
    print("Registered version:", version.version)

    if args.set_production:
        client.set_registered_model_alias(
            settings.registered_model_name,
            settings.production_alias,
            version.version,
        )
        print(
            f"Alias '{settings.production_alias}' now points to "
            f"version {version.version}."
        )


if __name__ == "__main__":
    main()
