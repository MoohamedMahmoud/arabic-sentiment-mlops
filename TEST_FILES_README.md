# Arabic Sentiment MLOps test files

This archive adds a `tests/` directory only; it does not replace or edit `src/`.

## Install the test client dependency

```powershell
uv add --dev httpx
```

## Extract

Extract the archive into the repository root and merge its `tests/` directory with your empty `tests/` directory.

## Fix the Ruff issues currently reported

```powershell
uv run ruff check --fix src tests
uv run ruff format src tests
```

These commands automatically organize imports/remove unused imports and apply formatting. Review `git diff` afterward.

## Run tests and checks

```powershell
uv run pytest tests/ -v
uv run ruff check src tests
uv run ruff format --check src tests
```

The tests intentionally mock the model and MLflow for API/predictor checks, so they do not start training or download model weights.

Do not add a `--cov-fail-under=70` gate until the current measured coverage has been reviewed and expanded to the agreed threshold. Otherwise the first test suite may fail only because parts such as the expensive training orchestration are not yet covered.
