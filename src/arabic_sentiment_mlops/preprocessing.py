"""Shared AraBERTv0.2 model-input preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class ArabicTextPreprocessor:
    """Prepare Arabic review text for AraBERTv0.2.

    There is intentionally no Farasa step here. The selected checkpoint is
    AraBERTv0.2, and its tokenizer is used directly.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        tokenizer_source: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer_source = tokenizer_source or model_name
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.tokenizer_source
        )

    def clean_text(self, text: str) -> str:
        """Apply the small, model-safe input cleanup used at inference time."""

        cleaned = str(text).strip()

        if not cleaned:
            raise ValueError("Input text must not be empty.")

        return cleaned

    def encode_one(self, text: str) -> dict[str, list[int]]:
        """Tokenize one review using the same settings as training."""

        cleaned = self.clean_text(text)

        return self.tokenizer(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def tokenize_dataset(self, frame: pd.DataFrame) -> Dataset:
        """Convert a prepared DataFrame into a tokenized Hugging Face dataset."""

        dataset = Dataset.from_pandas(
            frame[["text", "label"]].copy(),
            preserve_index=False,
        )

        def tokenize_batch(batch: dict[str, Sequence[str]]) -> dict[str, object]:
            texts = [self.clean_text(text) for text in batch["text"]]
            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            return encoded

        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text", "label"],
            desc="Tokenizing",
        )

        return tokenized
