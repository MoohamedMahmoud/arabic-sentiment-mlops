# Local serving and monitoring runbook

This demo stack is intentionally separate from `docker/docker-compose.yml`, which runs PostgreSQL, RustFS, and MLflow. It does not stop, rebuild, or recreate those services. It also does not manage unrelated Docker workloads such as Nextcloud.

## Components and local URLs

- BentoML service already running on the host: `http://127.0.0.1:3001`
- Observability proxy and Prometheus metrics: `http://127.0.0.1:8000`
- Streamlit user interface: `http://127.0.0.1:8501`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3002` (username `admin`; set `GRAFANA_ADMIN_PASSWORD` before launch for a non-default local password)

The proxy records end-to-end HTTP latency and error counts, predicted label counts, confidence, input character-length statistics, and PSI for recent predicted-label and input-length distributions. It stores only aggregate per-prediction events (label, confidence, character count, timestamp), not the submitted text itself.

## Start the stack

1. Ensure the existing BentoML container is healthy and listens on host port 3001. In another terminal, `Invoke-RestMethod http://127.0.0.1:3001/readyz` should succeed.
2. From the repository root, build and start the separate UI/monitoring stack:

   ```powershell
   docker compose -f docker/docker-compose.observability.yml up -d --build
   ```

3. Check the health and metrics endpoints:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/health
   Invoke-WebRequest http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty StatusCode
   docker compose -f docker/docker-compose.observability.yml ps
   ```

4. Open the Streamlit page and Grafana dashboard using the URLs above.

If you change only `ui/app.py`, rebuild only the frontend so the existing BentoML, MLflow, Prometheus, Grafana, and stored monitoring history remain untouched:

```powershell
docker compose -f docker/docker-compose.observability.yml up -d --build --no-deps frontend
```

## Evaluate the real serving path

Run a smoke test first. This only tests a small subset and does not create a drift baseline:

```powershell
uv run python scripts/evaluate_served_model.py --limit 32 --batch-size 4
```

If the smoke test succeeds, evaluate the full held-out test dataset using safe batches of at most 8:

```powershell
uv run python scripts/evaluate_served_model.py --batch-size 8 --retries 5
uv run python scripts/quality_gate.py reports/serving/api_evaluation.json
```

The full successful run saves class metrics, confusion matrix, confidence mean, batch latency percentiles, and a predictions CSV without raw review text. It also writes `monitoring/reference_distribution.json` from predictions on the full held-out test set. As the file is bind-mounted into the proxy container read-only, restart the demo stack after creating/updating it so the new reference is copied/loaded reliably:

```powershell
docker compose -f docker/docker-compose.observability.yml up -d --build --force-recreate inference-proxy
```

Do not generate a reference distribution from a smoke-test subset. Regenerate it only when intentionally changing the reference model or reference dataset. Drift alerts are disabled in practice until this evaluated baseline exists.

## Load test

First run 1–10 users for a short test and inspect the failure rate. Only proceed toward 25, 50, and then 100 concurrent users if errors and latency stay stable. Do not treat a target of 100 users as proof the service can handle 100 users; the Locust report is the evidence.

```powershell
New-Item -ItemType Directory -Force reports/locust | Out-Null
uv run --with locust locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --users 10 --spawn-rate 2 --run-time 60s --headless --csv reports/locust/smoke
```

Record p50/p95/p99 latency, RPS, failure rate, number of users, request mixture, hardware, and the first concurrency level where p95 increases sharply. Do not run a 100-user test until the smoke test is clean.

## GPU and Docker resource checks

- `docker stats` shows CPU and memory use for active containers.
- `nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv` records host GPU status.
- The BentoML container may not use the laptop GPU unless Docker is configured for GPU passthrough and the image has compatible CUDA dependencies. Do not report GPU acceleration unless it is verified during inference.

## Troubleshooting

- Proxy `/health` returns 503: verify BentoML at `127.0.0.1:3001/readyz` and check `docker logs arabic-sentiment-bento`.
- Proxy returns 502 or 503: inspect `docker logs` for both the BentoML container and the observability Compose stack. Reduce the test batch size to 1–4 while diagnosing.
- Proxy rejects a batch over 8: expected by design; it protects the currently configured BentoML batch limit.
- PSI remains 0 and `sentiment_drift_reference_available` is 0: the complete API evaluation baseline has not been created or loaded yet.
- Prometheus has no series: confirm `/metrics` responds on port 8000 and the Prometheus target is `UP` at `http://127.0.0.1:9090/targets`.
- Grafana cannot log in: set `GRAFANA_ADMIN_PASSWORD` in the shell environment before recreating its container. Keep this local-only stack bound to loopback; do not expose it publicly with default credentials.

## Stop only this stack

```powershell
docker compose -f docker/docker-compose.observability.yml down
```

This command does not stop the separate MLflow/RustFS/PostgreSQL Compose project or other Compose projects. It preserves the demo's named volumes. Use `down -v` only if you intentionally want to delete the stored monitoring history and Grafana/Prometheus data.
