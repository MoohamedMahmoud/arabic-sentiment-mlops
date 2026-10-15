"""Run the small, controlled AraBERTv0.2 tuning sweep."""

from __future__ import annotations

import json
import logging

import pandas as pd

from .config import settings
from .mlflow_utils import configure_mlflow, register_model, set_model_alias
from .train import TrainingConfig, train_one

LOGGER = logging.getLogger(__name__)


TRIALS = [
    TrainingConfig(learning_rate=1e-5, num_train_epochs=2, max_length=160),
    TrainingConfig(learning_rate=3e-5, num_train_epochs=2, max_length=160),
    TrainingConfig(learning_rate=2e-5, num_train_epochs=3, max_length=160),
    TrainingConfig(learning_rate=2e-5, num_train_epochs=2, max_length=256),
]

TRIAL_NAMES = [
    "lr_1e-5",
    "lr_3e-5",
    "epochs_3",
    "max_length_256",
]


def run_tuning() -> pd.DataFrame:
    """Run every trial and return results ordered by validation Macro-F1."""

    results: list[dict[str, object]] = []

    for name, config in zip(TRIAL_NAMES, TRIALS, strict=True):
        result = train_one(name, config, register_model=False)
        results.append(result)

    frame = pd.DataFrame(results).sort_values(
        "val_f1_macro",
        ascending=False,
    )
    frame = frame.reset_index(drop=True)

    report_path = settings.reports_dir / "mlflow" / "tuning_results.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_path, index=False)

    LOGGER.info("Best validation Macro-F1: %.4f", frame.iloc[0]["val_f1_macro"])
    return frame


def register_best(frame: pd.DataFrame) -> str:
    """Register the best validation run without promoting it automatically."""

    configure_mlflow(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )

    best = frame.iloc[0]
    version = register_model(
        model_uri=str(best["model_uri"]),
        model_name=settings.registered_model_name,
    )

    summary = {
        "selected_run": best["run_name"],
        "selected_validation_f1_macro": float(best["val_f1_macro"]),
        "registered_model": settings.registered_model_name,
        "registered_version": version,
    }

    path = settings.reports_dir / "mlflow" / "selected_model.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return version


def promote_version(version: str) -> None:
    """Assign the production alias after an explicit human decision."""

    set_model_alias(
        settings.registered_model_name,
        settings.production_alias,
        version,
    )


def main() -> None:
    """Run the sweep, register the selected candidate, and stop before promotion."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_tuning()
    LOGGER.info(
        "Tuning results:\n%s",
        results[["run_name", "val_accuracy", "val_f1_macro", "val_loss"]].to_string(
            index=False
        ),
    )

    version = register_best(results)
    LOGGER.info(
        "Registered %s version %s. Production alias was not changed.",
        settings.registered_model_name,
        version,
    )


if __name__ == "__main__":
    main()
