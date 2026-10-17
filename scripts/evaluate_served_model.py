"""Evaluate the running HTTP serving path against the held-out test Parquet.

Run from the repository root after the BentoML service and observability proxy are up.
A complete successful run writes metrics and updates the monitored reference distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
API_URL_DEFAULT = "http://127.0.0.1:8000/predict"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LENGTH_BINS = ["0-20", "21-50", "51-100", "101-200", "201+"]


def length_bin(length: int) -> str:
    if length <= 20:
        return "0-20"
    if length <= 50:
        return "21-50"
    if length <= 100:
        return "51-100"
    if length <= 200:
        return "101-200"
    return "201+"


def post_json(
    url: str, payload: dict[str, Any], timeout: float, retries: int
) -> tuple[Any, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
                elapsed = time.perf_counter() - started
                predictions = (
                    decoded.get("predictions") if isinstance(decoded, dict) else decoded
                )
                if isinstance(decoded, dict) and isinstance(decoded.get("value"), list):
                    predictions = decoded["value"]
                if not isinstance(predictions, list):
                    raise TypeError(f"Unexpected API response: {str(decoded)[:300]}")
                return predictions, elapsed
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(
                f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}"
            )
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                break
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt >= retries:
                break
        time.sleep(min(0.25 * (2**attempt), 3.0))
    raise RuntimeError(
        f"Request failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    parser.add_argument(
        "--test-path", type=Path, default=ROOT / "data/processed/test.parquet"
    )
    parser.add_argument("--batch-size", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate a small smoke-test subset; does not update reference baseline.",
    )
    args = parser.parse_args()

    if not args.test_path.is_file():
        raise SystemExit(f"Test file not found: {args.test_path}")
    frame = pd.read_parquet(args.test_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise SystemExit("The test Parquet must have text and label columns.")
    frame = frame[["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(str).str.upper()
    frame = frame[
        frame["text"].str.strip().ne("") & frame["label"].isin(LABELS)
    ].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit).copy()
    if frame.empty:
        raise SystemExit("No valid test rows were found.")

    print(
        f"Evaluating {len(frame):,} test rows in batches of {args.batch_size}; API={args.api_url}"
    )
    predictions: list[dict[str, Any]] = []
    batch_latencies: list[float] = []
    texts = frame["text"].tolist()
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        batch_predictions, elapsed = post_json(
            args.api_url, {"texts": batch}, args.timeout, args.retries
        )
        if len(batch_predictions) != len(batch):
            raise SystemExit(
                f"Batch starting at {start} returned {len(batch_predictions)} predictions for {len(batch)} texts."
            )
        batch_latencies.append(elapsed)
        for item in batch_predictions:
            predictions.append(
                {
                    "label": str(item["label"]).upper(),
                    "confidence": float(item["confidence"]),
                }
            )
        completed = min(start + len(batch), len(texts))
        if completed % max(args.batch_size * 25, 1) == 0 or completed == len(texts):
            print(f"Progress: {completed:,}/{len(texts):,}")

    actual = frame["label"].tolist()
    predicted = [item["label"] for item in predictions]
    confidence = [item["confidence"] for item in predictions]
    metrics = {
        "evaluation": "served_api",
        "api_url": args.api_url,
        "sample_count": len(frame),
        "complete_test_set": args.limit is None
        and len(frame) == len(pd.read_parquet(args.test_path)),
        "batch_size": args.batch_size,
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1_macro": float(
            f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels_order": LABELS,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "confidence_mean": sum(confidence) / max(len(confidence), 1),
        "batch_latency_seconds": {
            "p50": percentile(batch_latencies, 0.50),
            "p95": percentile(batch_latencies, 0.95),
            "p99": percentile(batch_latencies, 0.99),
            "mean": sum(batch_latencies) / max(len(batch_latencies), 1),
            "unit": "seconds per HTTP batch, not per item",
        },
        "per_class_f1": {
            label: float(
                f1_score(
                    actual, predicted, labels=[label], average="macro", zero_division=0
                )
            )
            for label in LABELS
        },
    }
    outdir = ROOT / "reports/serving"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "api_evaluation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_rows = []
    for i, (actual_label, prediction, text) in enumerate(
        zip(actual, predictions, texts, strict=True)
    ):
        output_rows.append(
            {
                "row_index": i,
                "actual": actual_label,
                "predicted": prediction["label"],
                "confidence": prediction["confidence"],
                "text_length": len(text),
            }
        )
    pd.DataFrame(output_rows).to_csv(outdir / "api_test_predictions.csv", index=False)
    print(f"Accuracy: {metrics['accuracy']:.4f}; macro-F1: {metrics['f1_macro']:.4f}")
    print("Per-class F1:", json.dumps(metrics["per_class_f1"], indent=2))
    print(
        "Batch latency p50/p95/p99 (s):",
        json.dumps(metrics["batch_latency_seconds"], indent=2),
    )
    print(
        f"Saved {outdir / 'api_evaluation.json'} and {outdir / 'api_test_predictions.csv'}"
    )

    if metrics["complete_test_set"]:
        label_counts = Counter(predicted)
        length_counts = Counter(length_bin(len(text)) for text in texts)
        reference = {
            "description": "Generated from successful API predictions on the complete held-out test set; regenerate only when intentionally changing the reference model or reference dataset.",
            "sample_count": len(frame),
            "label_distribution": {
                label: label_counts.get(label, 0) / len(frame) for label in LABELS
            },
            "text_length_bin_distribution": {
                name: length_counts.get(name, 0) / len(frame) for name in LENGTH_BINS
            },
        }
        reference_path = ROOT / "monitoring/reference_distribution.json"
        reference_path.write_text(
            json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Updated monitored reference baseline: {reference_path}")
        print(
            "Restart the proxy stack if the reference JSON was replaced by a read-only bind mount and drift metrics do not refresh."
        )
    else:
        print("Smoke test only: reference distribution was NOT changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
