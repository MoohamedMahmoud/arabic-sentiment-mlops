# Local MLflow tracking infrastructure (RustFS object store)

This Compose stack is separate from the existing Nextcloud Compose project. It runs MLflow, PostgreSQL, and RustFS locally.

## Why RustFS instead of MinIO?

The old `minio/minio` image reference in the prior bundle cannot currently be pulled from Docker Hub; the registry reports that the repository does not exist/access is denied. This configuration uses RustFS, an S3-compatible object store. MLflow's current upstream Compose example also uses RustFS. It is a substitute, not the MinIO product itself; confirm with the course reviewer if literal MinIO is mandatory in your rubric.

## Install / update

Extract this archive into the repository root. It updates `docker/docker-compose.yml`, `docker/mlflow.Dockerfile`, and this README, and supplies `.env.example`. Do not overwrite your existing `.env` blindly.

Edit the existing root `.env` so it contains these variable names (keep your actual secret values; never paste them into chat):

```dotenv
POSTGRES_PASSWORD=your_private_local_password
S3_ACCESS_KEY=your_private_access_key
S3_SECRET_KEY=your_private_object_store_secret
MLFLOW_BUCKET=mlflow-artifacts
```

If your `.env` currently has `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`, rename those variable names to `S3_ACCESS_KEY` and `S3_SECRET_KEY`, preserving the values. Use alphanumeric values for these local development credentials so the PostgreSQL URI does not need URL-escaping. Never commit `.env`.

Confirm Git ignores it:

```powershell
git check-ignore .env
```

## Validate and pull images before starting

From repository root:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . config --quiet
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . pull postgres object-store create-mlflow-bucket
```

If pulls succeed, build and start MLflow:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . up -d --build
```

Check status and logs:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . ps
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . logs --tail 100 mlflow postgres object-store create-mlflow-bucket
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
Invoke-RestMethod http://127.0.0.1:9000/health
```

- MLflow UI: http://127.0.0.1:5000
- RustFS Console: http://127.0.0.1:9001
- Object storage S3 API: http://127.0.0.1:9000

For the RustFS Console, use the S3 access key and secret from `.env`.

## Stop safely

Stop containers while retaining them and their data:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . stop
```

Remove containers and network while retaining data volumes:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml --project-directory . down
```

Do not add `-v` unless you deliberately intend to delete the PostgreSQL database and object-store artifacts. Do not stop or modify the existing Nextcloud stack.

## Project notes

- Host ports are bound to `127.0.0.1`, not all network interfaces. PostgreSQL is only reachable inside the Compose network.
- The MLflow container image is pinned to `v3.16.1`, matching the current project dependency version.
- The existing root `mlflow.db` is not migrated. This starts a new PostgreSQL-backed tracking store; runs from the old SQLite database will not automatically show up here.
- This is a local learning setup, not a production-hardened deployment.
