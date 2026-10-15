"""Train one AraBERTv0.2 configuration and track it with MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import ID2LABEL, LABEL2ID, settings
from .data import load_processed_splits
from .evaluate import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_frame,
)
from .mlflow_utils import configure_mlflow, get_dvc_data_hash, get_git_commit
from .preprocessing import ArabicTextPreprocessor

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training values for one reproducible experiment."""

    learning_rate: float = settings.learning_rate
    num_train_epochs: int = settings.num_train_epochs
    max_length: int = settings.max_length
    train_batch_size: int = settings.train_batch_size
    eval_batch_size: int = settings.eval_batch_size
    gradient_accumulation_steps: int = settings.gradient_accumulation_steps
    weight_decay: float = settings.weight_decay
    warmup_steps: float = settings.warmup_steps
    fp16: bool = settings.fp16
    gradient_checkpointing: bool = settings.gradient_checkpointing
    seed: int = settings.seed


def _build_tokenized_datasets(
    splits: dict[str, pd.DataFrame],
    preprocessor: ArabicTextPreprocessor,
) -> DatasetDict:
    """Convert prepared DataFrames to tokenized Hugging Face datasets."""

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        data = frame.copy()
        data["labels"] = data["label"].map(LABEL2ID).astype(int)
        return Dataset.from_pandas(
            data[["text", "labels"]],
            preserve_index=False,
        )

    datasets = DatasetDict(
        {
            "train": to_dataset(splits["train"]),
            "validation": to_dataset(splits["validation"]),
            "test": to_dataset(splits["test"]),
        }
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, object]:
        return preprocessor.tokenizer(
            batch["text"],
            truncation=True,
            max_length=preprocessor.max_length,
            padding=False,
        )

    return datasets.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )


def _save_validation_artifacts(
    trainer: Trainer,
    validation_dataset: Dataset,
    report_dir: Path,
) -> None:
    """Save the validation classification report and confusion matrix."""

    prediction = trainer.predict(validation_dataset)
    predictions = prediction.predictions.argmax(axis=-1)
    labels = prediction.label_ids

    report = classification_report_dict(labels, predictions)
    matrix = confusion_matrix_frame(labels, predictions)

    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    matrix.to_csv(report_dir / "confusion_matrix.csv")


def _log_trainer_history(trainer: Trainer) -> None:
    """Log step-level training loss and epoch-level validation metrics."""

    validation_metrics = {
        "eval_loss": "val_loss",
        "eval_accuracy": "val_accuracy",
        "eval_f1_macro": "val_f1_macro",
    }

    for record in trainer.state.log_history:
        step_value = record.get("step")
        if step_value is None:
            continue

        step = int(step_value)

        if "loss" in record:
            mlflow.log_metric(
                "train_step_loss",
                float(record["loss"]),
                step=step,
            )

        for source_name, metric_name in validation_metrics.items():
            if source_name in record:
                mlflow.log_metric(
                    metric_name,
                    float(record[source_name]),
                    step=step,
                )


def train_one(
    run_name: str,
    config: TrainingConfig,
) -> dict[str, object]:
    """Train one configuration and track metrics without uploading model weights."""

    set_seed(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Fix the GPU environment before training."
        )

    configure_mlflow(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )

    splits = load_processed_splits(settings.processed_data_dir)

    preprocessor = ArabicTextPreprocessor(
        model_name=settings.model_name,
        max_length=config.max_length,
    )
    datasets = _build_tokenized_datasets(splits, preprocessor)

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_name,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    collator = DataCollatorWithPadding(
        tokenizer=preprocessor.tokenizer,
        pad_to_multiple_of=8,
    )

    run_dir = settings.models_dir / "runs" / run_name
    report_dir = settings.reports_dir / "mlflow" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        dataloader_num_workers=0,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        processing_class=preprocessor.tokenizer,
        compute_metrics=compute_metrics,
    )

    git_commit = get_git_commit(settings.project_root)
    data_hash = get_dvc_data_hash(settings.raw_data_path)

    start = time.perf_counter()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_family": "AraBERTv0.2",
                "model_name": settings.model_name,
                "git_commit": git_commit,
                "data_version": data_hash,
                "seed": str(config.seed),
                "local_model_path": str(run_dir.resolve()),
                "model_artifact_uploaded": "false",
            }
        )
        mlflow.log_params(asdict(config))
        mlflow.log_param("model_name", settings.model_name)

        trainer.train()
        training_seconds = time.perf_counter() - start

        # Evaluate the selected model on the training and validation sets.
        train_evaluation = trainer.evaluate(
            datasets["train"],
            metric_key_prefix="train",
        )
        validation = trainer.evaluate(
            datasets["validation"],
            metric_key_prefix="val",
        )

        metrics = {
            "train_loss": float(train_evaluation["train_loss"]),
            "train_accuracy": float(train_evaluation["train_accuracy"]),
            "train_f1_macro": float(train_evaluation["train_f1_macro"]),
            "val_loss": float(validation["val_loss"]),
            "val_accuracy": float(validation["val_accuracy"]),
            "val_f1_macro": float(validation["val_f1_macro"]),
            "training_time_seconds": float(training_seconds),
        }

        # Log learning curves and final summary metrics.
        _log_trainer_history(trainer)
        mlflow.log_metrics(
            metrics,
            step=int(trainer.state.global_step),
        )

        # Keep model weights and tokenizer on the local machine.
        trainer.save_model(run_dir)
        preprocessor.tokenizer.save_pretrained(run_dir)

        # Save small evaluation reports.
        _save_validation_artifacts(
            trainer,
            datasets["validation"],
            report_dir,
        )

        # DVC expects this metrics file to exist after training.
        metrics_path = report_dir / "train_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Upload reports, not the model weights.
        mlflow.log_artifacts(
            str(report_dir),
            artifact_path="evaluation",
        )

        result = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            **metrics,
            "model_path": str(run_dir),
            "config": asdict(config),
        }

    LOGGER.info("Finished %s: val_f1_macro=%.4f", run_name, result["val_f1_macro"])
    return result


def main() -> None:
    """Train the configured best-so-far model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="arabert_v02_final")
    parser.add_argument("--learning-rate", type=float, default=settings.learning_rate)
    parser.add_argument("--epochs", type=int, default=settings.num_train_epochs)
    parser.add_argument("--max-length", type=int, default=settings.max_length)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
    )

    train_one(args.run_name, config)


if __name__ == "__main__":
    main()
