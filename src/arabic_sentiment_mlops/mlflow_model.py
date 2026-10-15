"""Custom MLflow PyFunc wrapper for the AraBERT model."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd

from .model import ArabicSentimentModel


class ArabicSentimentMLflowModel(mlflow.pyfunc.PythonModel):
    """Expose the same sentiment model through MLflow's generic interface."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the trained model from its MLflow artifact."""

        model_dir = context.artifacts["model_dir"]
        model_name = context.model_config["model_name"]
        max_length = int(context.model_config["max_length"])

        self.model = ArabicSentimentModel(
            model_path=model_dir,
            model_name=model_name,
            max_length=max_length,
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Predict from a DataFrame containing a ``text`` column."""

        if isinstance(model_input, pd.Series):
            texts = model_input.tolist()
        elif isinstance(model_input, pd.DataFrame):
            if "text" not in model_input.columns:
                raise ValueError("MLflow model input must contain a 'text' column.")
            texts = model_input["text"].tolist()
        else:
            raise TypeError("MLflow model input must be a pandas DataFrame or Series.")

        predictions = self.model.predict_batch([str(text) for text in texts])
        return pd.DataFrame(predictions)
