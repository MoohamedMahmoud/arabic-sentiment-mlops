"""Benchmark client-side request batch sizes against the running sentiment API.

This is a performance experiment, not a model-quality evaluation. It does not
modify the API evaluation report, monitoring baseline, model, or running service.

Example (from repository root):
    uv run python scripts/benchmark_served_model.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
DEFAULT_TEST_PATH = ROOT / "data" / "processed" / "test.parquet"
DEFAULT_OUTPUT = ROOT / "reports" / "serving" / "runtime_benchmark.json"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def request_predictions(
    url: str, texts: list[str], timeout: float
) -> tuple[int, list[dict[str, Any]] | None, str | None]:
    payload = json.dumps({"texts": texts}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:250]
        return exc.code, None, f"HTTP {exc.code}: {detail}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"

    predictions: Any
    if isinstance(body, dict):
        predictions = body.get("predictions", body.get("value"))
    else:
        predictions = body
    if not isinstance(predictions, list) or len(predictions) != len(texts):
        return (
            status,
            None,
            f"Unexpected response or prediction count; expected {len(texts)}",
        )
    if any(
        not isinstance(item, dict)
        or item.get("label") not in {"NEGATIVE", "NEUTRAL", "POSITIVE"}
        for item in predictions
    ):
        return status, None, "Response contains an invalid prediction label"
    return status, predictions, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8",
        help="Comma-separated request payload sizes, maximum 8",
    )
    parser.add_argument(
        "--requests-per-size",
        type=int,
        default=10,
        help="Measured requests per batch size; 10 is exploratory, increase for stronger estimates",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=32,
        help="Number of fixed test texts sampled from for each configuration",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        batch_sizes = sorted(
            {int(value.strip()) for value in args.batch_sizes.split(",")}
        )
    except ValueError as exc:
        raise SystemExit(
            "--batch-sizes must be comma-separated integers, e.g. 1,4,8"
        ) from exc
    if not batch_sizes or any(size < 1 or size > 8 for size in batch_sizes):
        raise SystemExit("Batch sizes must be integers between 1 and 8")
    if args.requests_per_size < 1 or args.pool_size < max(batch_sizes):
        raise SystemExit(
            "--requests-per-size must be >= 1 and --pool-size must be >= the largest batch size"
        )
    if not args.test_path.is_file():
        raise SystemExit(f"Test file not found: {args.test_path}")

    frame = pd.read_parquet(args.test_path)
    if "text" not in frame.columns:
        raise SystemExit("Test Parquet must contain a 'text' column")
    texts = frame["text"].dropna().astype(str)
    texts = texts[texts.str.strip().ne("")].head(args.pool_size).tolist()
    if len(texts) < max(batch_sizes):
        raise SystemExit(
            f"Need at least {max(batch_sizes)} valid texts, found {len(texts)}"
        )

    # Untimed warm-up checks reachability and response shape before measurement.
    status, _predictions, error = request_predictions(
        args.api_url, [texts[0]], args.timeout
    )
    if error:
        raise SystemExit(f"API preflight failed (status {status}): {error}")

    results: list[dict[str, Any]] = []
    print(f"API: {args.api_url}")
    print(
        f"Test text pool: {len(texts)}; measured requests per size: {args.requests_per_size}"
    )
    print("No retries are used, so HTTP failures remain visible in the report.\n")

    for batch_size in batch_sizes:
        rng = random.Random(
            args.seed
        )  # same deterministic sampling strategy for each size
        latencies: list[float] = []
        successful_requests = 0
        failed_requests = 0
        successful_texts = 0
        errors: list[str] = []
        config_started = time.perf_counter()

        for request_index in range(args.requests_per_size):
            # Cycle through a fixed pool, avoiding confounding each configuration
            # with a different split of the test set.
            start = (request_index * batch_size) % len(texts)
            batch = [
                texts[(start + offset) % len(texts)] for offset in range(batch_size)
            ]
            # Stable per-request variation while keeping all settings reproducible.
            if len(batch) > 1:
                rng.shuffle(batch)
            request_started = time.perf_counter()
            status, predictions, error = request_predictions(
                args.api_url, batch, args.timeout
            )
            elapsed = time.perf_counter() - request_started
            latencies.append(elapsed)
            if error is None and predictions is not None:
                successful_requests += 1
                successful_texts += len(predictions)
            else:
                failed_requests += 1
                if len(errors) < 5:
                    errors.append(
                        error or f"Unexpected failure with HTTP status {status}"
                    )
            print(
                f"batch_size={batch_size} request={request_index + 1}/{args.requests_per_size} "
                f"status={'OK' if error is None else 'FAIL'} latency={elapsed:.3f}s"
            )

        wall_seconds = time.perf_counter() - config_started
        result = {
            "batch_size": batch_size,
            "requests_attempted": args.requests_per_size,
            "requests_succeeded": successful_requests,
            "requests_failed": failed_requests,
            "texts_succeeded": successful_texts,
            "wall_time_seconds": round(wall_seconds, 6),
            "successful_texts_per_second": round(successful_texts / wall_seconds, 6)
            if wall_seconds > 0
            else 0.0,
            "request_latency_seconds": {
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "mean": mean(latencies) if latencies else None,
                "unit": "seconds per HTTP request including failures",
            },
            "sample_errors": errors,
        }
        results.append(result)
        print(
            f"SUMMARY size={batch_size}: successful={successful_requests}/{args.requests_per_size}, "
            f"texts/s={result['successful_texts_per_second']:.4f}, "
            f"p50={result['request_latency_seconds']['p50']}, "
            f"p95={result['request_latency_seconds']['p95']}\n"
        )

    report = {
        "benchmark": "served_api_client_batch_size_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "api_url": args.api_url,
        "test_path": str(args.test_path),
        "test_text_pool_size": len(texts),
        "requests_per_batch_size": args.requests_per_size,
        "batch_sizes": batch_sizes,
        "timeout_seconds": args.timeout,
        "retries": 0,
        "notes": [
            "Performance-only experiment; this does not recompute model-quality metrics.",
            "Tests client request payload sizes against the same currently running service; it does not compare different server configurations or model formats.",
            "The small default run is exploratory; increase --requests-per-size for more stable tail-latency estimates.",
            "Failures are recorded, not retried. Do not use these results as a substitute for a full concurrency/load test.",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved benchmark report: {args.output}")
    return 0 if all(item["requests_failed"] == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
