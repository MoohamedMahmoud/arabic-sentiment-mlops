"""Project configuration.

Keep paths, model settings, and MLflow settings in one place so they are not
repeated throughout the project.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL2ID = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


class Settings(BaseSettings):
    """Application and training settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT

    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "alldata.csv"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    model_name: str = "aubmindlab/bert-base-arabertv02"
    max_length: int = 160

    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "arabic-sentiment-arabert-v02"
    registered_model_name: str = "ArabicSentiment"
    production_alias: str = "production"

    @property
    def production_model_uri(self) -> str:
        """Return the MLflow model URI used by production inference."""

        return f"models:/{self.registered_model_name}@{self.production_alias}"


settings = Settings()
