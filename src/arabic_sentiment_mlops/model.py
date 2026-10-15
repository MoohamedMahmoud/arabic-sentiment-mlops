"""AraBERTv0.2 model wrapper."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification

from .config import ID2LABEL, LABEL2ID
from .preprocessing import ArabicTextPreprocessor


class ArabicSentimentModel:
    """Load and run the Arabic sentiment classifier."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        max_length: int,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Use the tokenizer saved with the trained model whenever possible.
        # This keeps training and production tokenization exactly aligned.
        self.preprocessor = ArabicTextPreprocessor(
            model_name=model_name,
            max_length=max_length,
            tokenizer_source=model_path,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(LABEL2ID),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_one(self, text: str) -> dict[str, Any]:
        """Return one sentiment prediction."""

        encoded = self.preprocessor.encode_one(text)
        batch = {
            key: torch.tensor([value], dtype=torch.long, device=self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "token_type_ids", "attention_mask"}
        }

        with torch.inference_mode():
            outputs = self.model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item())

        return {
            "label": ID2LABEL[predicted_id],
            "confidence": confidence,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return predictions for multiple reviews."""

        return [self.predict_one(text) for text in texts]
