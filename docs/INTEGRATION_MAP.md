# Integration map — read before extracting

This patch is additive to the repository layout shown on 2026-10-17. Do not move or rename existing files.

## Existing files preserved

- `docker/docker-compose.yml` — existing MLflow + PostgreSQL + RustFS stack. Do not replace or edit it for this integration.
- `docker/mlflow.Dockerfile` — existing MLflow image build file.
- `serving/service.py`, `bentofile.yaml`, `wheels/`, `src/`, `tests/`, `data/`, `models/`, `.env`, `pyproject.toml`, `uv.lock`, `dvc.yaml`, `dvc.lock`, and the current root `README.md` remain untouched by this patch.

## New files and folders

- `docker/docker-compose.observability.yml`: separate Compose project for the UI, API metrics proxy, Prometheus, and Grafana. It intentionally does not repeat MLflow/RustFS/PostgreSQL services and does not own or restart the BentoML container.
- `ui/`: Streamlit application and its isolated Docker build.
- `monitoring/`: FastAPI metrics proxy, Prometheus rules/config, Grafana dashboard/provisioning, and initially empty drift reference.
- `scripts/`: evaluation of the *actual HTTP-served model* and a quality gate.
- `loadtest/`: Locust workload targeting the proxy API.
- `reports/serving/` and `reports/locust/`: locations for generated evaluation and load-test evidence.
- `docs/runbook.md`: run, verify, troubleshoot, and stop instructions.
- `.github/workflows/ci.yml`: CI checks.
- `docs/README-final-draft.md`: proposed README content for manual review; it does not replace the current root `README.md`.

## Compose boundaries

Existing stack (leave running):

```powershell
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . up -d
```

The existing BentoML service is separately run on host port 3001. The new observability stack connects to `http://host.docker.internal:3001` by default. New services listen only on host loopback: proxy `8000`, Streamlit `8501`, Prometheus `9090`, Grafana `3002`.

Start the additive stack from the repo root only after the BentoML `/readyz` endpoint is healthy:

```powershell
Invoke-RestMethod http://127.0.0.1:3001/readyz
docker compose -f docker/docker-compose.observability.yml config
docker compose -f docker/docker-compose.observability.yml up -d --build
```

Stop only the added stack with `docker compose -f docker/docker-compose.observability.yml down`. Do not use `down -v` unless intentionally deleting monitoring history. It does not stop the existing Compose project or unrelated Docker containers.

## Safe extraction

Extract into the repository root **without `-Force`**. The patch intentionally contains no root `README.md`, `.dockerignore`, `docker-compose.yml`, `.env`, model artifacts, DVC metadata replacements, or compiled Python files. If PowerShell reports a path collision, stop and inspect it rather than overwriting.
