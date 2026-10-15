"""Tests for input cleanup/tokenization without downloading AraBERT."""

import pytest

from arabic_sentiment_mlops.preprocessing import ArabicTextPreprocessor


class FakeTokenizer:
    """A predictable tokenizer substitute for fast unit tests."""

    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None

    def __call__(self, text, **kwargs):
        self.last_call = {"text": text, **kwargs}
        return {"input_ids": [101, 102], "attention_mask": [1, 1]}


def test_clean_text_strips_outer_whitespace() -> None:
    preprocessor = ArabicTextPreprocessor.__new__(ArabicTextPreprocessor)

    assert preprocessor.clean_text("   المنتج ممتاز   ") == "المنتج ممتاز"


def test_clean_text_rejects_empty_or_whitespace_only_text() -> None:
    preprocessor = ArabicTextPreprocessor.__new__(ArabicTextPreprocessor)

    with pytest.raises(ValueError, match="must not be empty"):
        preprocessor.clean_text("   ")


def test_encode_one_uses_truncation_and_configured_max_length(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "arabic_sentiment_mlops.preprocessing.AutoTokenizer.from_pretrained",
        classmethod(lambda cls, source: fake_tokenizer),
    )
    preprocessor = ArabicTextPreprocessor(model_name="fake-model", max_length=16)

    encoded = preprocessor.encode_one("المنتج ممتاز")

    assert encoded == {"input_ids": [101, 102], "attention_mask": [1, 1]}
    assert fake_tokenizer.last_call == {
        "text": "المنتج ممتاز",
        "truncation": True,
        "max_length": 16,
        "padding": False,
    }
