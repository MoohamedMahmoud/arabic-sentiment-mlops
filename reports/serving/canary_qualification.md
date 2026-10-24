# Canary Qualification Record — Arabic Sentiment MLOps

**Record date:** 2026-10-23  
**Candidate:** MLflow registered model `ArabicSentiment`, version `2` (alias `production`)  
**BentoML model tag:** `arabic_sentiment_v2:ui5zcfgkc2qxoaav`  
**Serving path evaluated:** API proxy `http://127.0.0.1:8000/predict`  
**Decision:** **NO-GO for unrestricted production release; conditional use for a controlled demonstration capped at 5 concurrent users.**

## 1. Model-quality qualification

| Check | Acceptance criterion | Observed result | Status |
|---|---:|---:|---|
| Complete held-out evaluation | All 8,874 test rows evaluated | 8,874 / 8,874 | PASS |
| Macro-F1 | ≥ 0.8500 | 0.8702 | PASS |
| Neutral-class F1 | ≥ 0.8000 | 0.8311 | PASS |
| Accuracy (reported) | Informational | 0.8781 | — |
| Negative-class F1 (reported) | Informational | 0.9139 | — |
| Positive-class F1 (reported) | Informational | 0.8654 | — |

The `scripts/quality_gate.py` command completed with `QUALITY GATE PASSED`. The full API evaluation saved `reports/serving/api_evaluation.json` and `reports/serving/api_test_predictions.csv`, and updated `monitoring/reference_distribution.json`.

## 2. Load-test evidence

These are the previously captured 60-second Locust runs. They are short-duration observations, not a long-term availability guarantee.

| Concurrent users | Requests | Failures | Average latency | P95 latency | P99 latency | Maximum observed latency | Assessment |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 88 | 0 | 560 ms | 880 ms | Not recorded | 973 ms | PASS for observed run |
| 5 | 103 | 0 | 708 ms | 1,100 ms | Not recorded | 1,311 ms | PASS for observed run |
| 10 | 196 | 1 (0.51%) | 818 ms | 1,800 ms | 2,700 ms | 2,915 ms | FAIL for zero-error acceptance |

Two 10-user trials were reported; each had one upstream BentoML HTTP 503-related failure (approximately 0.5%). The corresponding proxy error was HTTP 502, and BentoML logs reported `ServiceUnavailable: process is overloaded`.

## 3. Full-evaluation latency

| Metric | Observed value |
|---|---:|
| Batch size | 8 examples per HTTP batch |
| Batch latency p50 | 2.987 s |
| Batch latency p95 | 9.288 s |
| Batch latency p99 | 22.560 s |
| Mean batch latency | 3.984 s |

These are **seconds per HTTP batch, not per individual example**. Keep these results separate from the Locust latency figures. The long tail warrants further investigation; retries allowed the full evaluation to finish, but do not prove the overload behavior has been eliminated.

## 4. Resource snapshot

A point-in-time `docker stats --no-stream` snapshot was collected after the tests; it was not captured throughout a load run and must not be interpreted as peak utilization.

| Component | CPU | Memory usage / limit | Memory % | Network I/O |
|---|---:|---:|---:|---:|
| `arabic-sentiment-bento` | 0.16% | 847.5 MiB / 3.388 GiB | 24.43% | 3.87 MB / 1.37 MB |
| `arabic-sentiment-demo-inference-proxy-1` | 0.11% | 40.79 MiB / 3.388 GiB | 1.18% | 5.44 MB / 6.45 MB |

## 5. Decision and operating boundary

- **Model quality:** PASS against the current configured quality thresholds.
- **Observed short-run behavior at 4–5 users:** PASS, with no failed requests in the captured runs.
- **Observed behavior at 10 users:** FAIL for a zero-error reliability criterion because a request failed with an overload-related 502/503.
- **Controlled demonstration:** May be run with a provisional cap of **5 concurrent users**, while monitoring errors and latency. This is a cautious operating boundary based on the tests available, not a capacity guarantee.
- **Unrestricted production release:** NO-GO. The 10-user failures and full-evaluation latency tail remain unresolved, and the available point-in-time resource snapshot does not measure peak load utilization.
- **Formal canary rollout:** Not yet proven by the evidence in this record. A real canary should also demonstrate traffic splitting between the current baseline and candidate, monitored acceptance criteria, and a tested rollback path. Do not claim that a canary rollout has occurred unless those actions are performed and recorded.

## 6. Release acceptance criteria

Before revisiting the production NO-GO decision, record evidence that:

1. The full 8,874-row test evaluation remains complete and the quality gate passes (`Macro-F1 ≥ 0.85`, `Neutral F1 ≥ 0.80`).
2. The declared target load completes without 5xx responses during the agreed test window. Until a higher capacity is demonstrated, use 5 concurrent users as the provisional demonstration cap.
3. Latency targets are explicitly defined and evaluated consistently at the request or batch level; do not compare the two units directly.
4. CPU and memory are measured during the target-load run, including peak or time-series values rather than a single idle snapshot.
5. The canary traffic split, monitoring checks, and rollback procedure are exercised and recorded before calling the release a qualified production canary.

## 7. Evidence and limitations

This record combines the completed full API evaluation and quality-gate result, the previously captured Locust runs, and the point-in-time Docker resource snapshot supplied on 2026-10-23. The available measurements are short and local. They do not demonstrate long-term availability, behavior under changing traffic, multi-instance resilience, or production SLO compliance. No additional load test or serving-configuration change is implied by this record.
