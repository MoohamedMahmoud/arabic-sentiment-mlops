"""Small MLflow helpers kept in one place."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the tracking server and select an experiment."""

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def get_git_commit(project_root: Path) -> str:
    """Return the current Git commit, or ``unknown`` outside a Git repo."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    """Return a SHA-256 file hash."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_dvc_data_hash(data_path: Path) -> str:
    """Return the DVC source hash when available.

    For a DVC-tracked file, the adjacent ``.dvc`` file normally contains the
    content hash. Until the file is DVC-tracked, use SHA-256 as a transparent
    fallback rather than inventing a version number.
    """

    dvc_file = Path(f"{data_path}.dvc")

    if dvc_file.exists():
        content = dvc_file.read_text(encoding="utf-8")
        match = re.search(r"\bmd5:\s*([0-9a-f]+)", content)
        if match:
            return match.group(1)

    if data_path.exists():
        return sha256_file(data_path)

    return "unknown"


def register_model(model_uri: str, model_name: str) -> str:
    """Register one already-logged MLflow model and return its version."""

    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    return str(version.version)


def set_model_alias(model_name: str, alias: str, version: str) -> None:
    """Point an MLflow alias to a model version."""

    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, int(version))


def get_model_version_by_alias(model_name: str, alias: str):
    """Return the registry metadata for an alias."""

    client = MlflowClient()
    return client.get_model_version_by_alias(model_name, alias)
