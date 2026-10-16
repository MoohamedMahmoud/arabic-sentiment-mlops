"""Run a sequential, resumable AraBERT tuning sweep."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .config import settings
from .train import TrainingConfig, train_one

LOGGER = logging.getLogger(__name__)

TRIALS = [
    ("lr_1e-5", TrainingConfig(learning_rate=1e-5, num_train_epochs=2, max_length=160)),
    ("lr_3e-5", TrainingConfig(learning_rate=3e-5, num_train_epochs=2, max_length=160)),
    (
        "epochs_3",
        TrainingConfig(learning_rate=2e-5, num_train_epochs=3, max_length=160),
    ),
    (
        "max_length_256",
        TrainingConfig(learning_rate=2e-5, num_train_epochs=2, max_length=256),
    ),
]

METRIC_NAMES = (
    "train_loss",
    "train_accuracy",
    "train_f1_macro",
    "val_loss",
    "val_accuracy",
    "val_f1_macro",
    "training_time_seconds",
)

REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

REPORT_DIR = settings.reports_dir / "mlflow"
STATE_PATH = REPORT_DIR / "tuning_sweep_state.json"
RESULTS_PATH = REPORT_DIR / "tuning_sweep_results.csv"
SUMMARY_PATH = REPORT_DIR / "tuning_sweep_summary.json"


def _write_json_atomic(path: Path, content: dict[str, object]) -> None:
    """Write JSON atomically so interrupted writes do not corrupt progress."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _new_state(
    expected_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Create a clean progress record for this exact sweep configuration."""

    return {
        "schema_version": 1,
        "trial_configs": expected_configs,
        "completed": {},
        "failures": {},
    }


def _load_state(
    expected_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Load compatible progress, or start a new sweep state."""

    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if (
                state.get("schema_version") == 1
                and state.get("trial_configs") == expected_configs
                and isinstance(state.get("completed"), dict)
                and isinstance(state.get("failures"), dict)
            ):
                LOGGER.info("Loaded existing tuning progress.")
                return state
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not read prior tuning state; starting fresh.")

        LOGGER.warning(
            "Saved tuning state is incompatible with the current configurations. "
            "Completed results will not be reused."
        )

    return _new_state(expected_configs)


def _result_is_usable(name: str, result: object) -> bool:
    """Check that a completed trial has metrics, reports, and local weights."""

    if not isinstance(result, dict) or result.get("run_name") != name:
        return False

    if any(result.get(metric) is None for metric in METRIC_NAMES):
        return False

    model_path = Path(str(result.get("model_path", "")))
    if not model_path.is_dir():
        return False

    if not all((model_path / filename).is_file() for filename in REQUIRED_MODEL_FILES):
        return False

    metrics_path = REPORT_DIR / name / "train_metrics.json"
    if not metrics_path.is_file():
        return False

    try:
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return all(saved_metrics.get(metric) is not None for metric in METRIC_NAMES)


def _completed_results(
    state: dict[str, object],
) -> list[dict[str, object]]:
    completed = state["completed"]
    assert isinstance(completed, dict)

    results = []
    for name, _ in TRIALS:
        result = completed.get(name)
        if _result_is_usable(name, result):
            assert isinstance(result, dict)
            results.append(result)

    return results


def _write_progress(state: dict[str, object]) -> pd.DataFrame:
    """Persist progress after each trial, including a best-so-far summary."""

    _write_json_atomic(STATE_PATH, state)
    results = _completed_results(state)

    if results:
        frame = pd.DataFrame(results).sort_values(
            "val_f1_macro",
            ascending=False,
        )
        temporary_csv = RESULTS_PATH.with_name(RESULTS_PATH.name + ".tmp")
        frame.to_csv(temporary_csv, index=False)
        temporary_csv.replace(RESULTS_PATH)
        best = max(results, key=lambda result: float(result["val_f1_macro"]))
    else:
        frame = pd.DataFrame()
        best = None

    completed_count = len(results)
    failures = state["failures"]
    assert isinstance(failures, dict)

    status = (
        "completed"
        if completed_count == len(TRIALS) and not failures
        else "partial"
        if failures
        else "running"
    )

    summary: dict[str, object] = {
        "status": status,
        "expected_trials": len(TRIALS),
        "completed_trials": completed_count,
        "failed_trials": dict(failures),
    }

    if best is not None:
        summary.update(
            {
                "best_run_id": best["run_id"],
                "best_run_name": best["run_name"],
                "best_model_path": best["model_path"],
                "best_config": best["config"],
            }
        )
        for metric in METRIC_NAMES:
            summary[f"best_{metric}"] = float(best[metric])

    _write_json_atomic(SUMMARY_PATH, summary)
    return frame


def run_tuning(*, dry_run: bool = False) -> pd.DataFrame | None:
    """Run four trials sequentially, preserving completed results for resumption."""

    expected_configs = {name: asdict(config) for name, config in TRIALS}

    if dry_run:
        print(f"Planned sequential trials: {len(TRIALS)}")
        for name, config in TRIALS:
            print(f"\n{name}")
            print(json.dumps(asdict(config), indent=2))
        print("\nMetrics per trial:", ", ".join(METRIC_NAMES))
        print("No training, uploads, or model registration performed.")
        return None

    state = _load_state(expected_configs)
    completed = state["completed"]
    failures = state["failures"]
    assert isinstance(completed, dict)
    assert isinstance(failures, dict)

    # Discard saved results if their local artifacts are no longer complete.
    for name, _ in TRIALS:
        if not _result_is_usable(name, completed.get(name)):
            completed.pop(name, None)
        else:
            failures.pop(name, None)

    _write_progress(state)

    pending = [
        name for name, _ in TRIALS if not _result_is_usable(name, completed.get(name))
    ]

    if pending and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. No trials were started; check the GPU first."
        )

    for name, config in TRIALS:
        cached_result = completed.get(name)
        if _result_is_usable(name, cached_result):
            LOGGER.info(
                "Skipping completed trial %s; its local model and metrics exist.",
                name,
            )
            continue

        # Clear the previous failure entry before retrying this trial.
        completed.pop(name, None)
        failures.pop(name, None)
        _write_progress(state)

        LOGGER.info("Starting trial %s with config=%s", name, asdict(config))

        try:
            result = train_one(name, config)

            if not _result_is_usable(name, result):
                raise RuntimeError(
                    f"Trial {name} returned without all expected metrics, "
                    "reports, and local model files."
                )

            completed[name] = result
            failures.pop(name, None)
            _write_progress(state)

            LOGGER.info(
                "Completed trial %s: val_f1_macro=%.4f",
                name,
                float(result["val_f1_macro"]),
            )

        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            _write_progress(state)
            LOGGER.exception(
                "Trial %s failed. Progress was saved; stopping to avoid wasting compute.",
                name,
            )
            raise

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = _write_progress(state)
    results = _completed_results(state)

    if len(results) != len(TRIALS) or failures:
        incomplete = [
            name
            for name, _ in TRIALS
            if not _result_is_usable(name, completed.get(name))
        ]
        raise RuntimeError(
            f"Tuning incomplete: {len(results)}/{len(TRIALS)} trials succeeded. "
            f"Incomplete trials: {incomplete}. Completed trials were preserved; "
            "rerun the same command after fixing the reported failure to resume."
        )

    LOGGER.info(
        "Tuning results:\n%s",
        frame[
            [
                "run_name",
                "train_loss",
                "train_accuracy",
                "train_f1_macro",
                "val_loss",
                "val_accuracy",
                "val_f1_macro",
                "training_time_seconds",
            ]
        ].to_string(index=False),
    )

    best = max(results, key=lambda result: float(result["val_f1_macro"]))
    LOGGER.info(
        "Best trial: %s | validation Macro-F1: %.4f",
        best["run_name"],
        float(best["val_f1_macro"]),
    )
    LOGGER.info("Best-model summary saved to %s", SUMMARY_PATH)
    LOGGER.info("No model was uploaded or registered by the tuning sweep.")

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial configurations without starting training.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    run_tuning(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
