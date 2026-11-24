"""MLflow stores experiments and immutable model versions for the local release controller."""

import json
import os
import shutil
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL = "spiritops-weekly-volume"
EXPERIMENT = "SpiritOps"


def client(home: Path) -> MlflowClient:
    uri = (
        os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{home.resolve().as_posix()}/mlflow.sqlite3"
    )
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def start(home: Path, job_id: str, attempt: int, payload: dict[str, Any]) -> str:
    api = client(home)
    experiment = api.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        local = not os.environ.get("MLFLOW_TRACKING_URI", "").startswith(("https:", "http:"))
        location = (home.resolve() / "mlartifacts").as_uri() if local else None
        experiment_id = api.create_experiment(EXPERIMENT, artifact_location=location)
    else:
        experiment_id = experiment.experiment_id
    git_sha = os.environ.get("SPIRITOPS_GIT_SHA")
    if not git_sha:
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            git_sha = "unavailable"
    run = api.create_run(
        experiment_id,
        run_name=f"{payload['kind']}-{job_id[:8]}-{attempt}",
        tags={"spiritops.job_id": job_id, "spiritops.attempt": str(attempt), "git_sha": git_sha},
    )
    run_id = str(run.info.run_id)
    for key, value in payload.items():
        api.log_param(run_id, key, value)
    versions = {
        name: metadata.version(name) for name in ("lightgbm", "pandas", "mlflow", "duckdb", "numpy")
    }
    api.log_dict(run_id, versions, "dependencies.json")
    return run_id


def register(
    home: Path, run_id: str, directory: Path, meta: dict[str, Any], sha: dict[str, str]
) -> str:
    api = client(home)
    api.log_artifacts(run_id, str(directory), "native")
    api.log_dict(run_id, sha, "integrity.json")
    api.log_dict(
        run_id,
        {
            "inputs": [{"name": field, "type": "double"} for field in meta["feature_names"]],
            "outputs": [{"name": "litres", "type": "double"}],
        },
        "signature.json",
    )
    for key in ("schema_version", "dataset_sha256", "training_start", "training_end"):
        api.set_tag(run_id, key, str(meta[key]))
    for name, value in meta.get("parameters", {}).items():
        api.log_param(run_id, f"model.{name}", value)
    for name in ("duration_seconds", "rss_bytes", "training_rows"):
        if name in meta:
            api.log_metric(run_id, name, float(meta[name]))
    api.log_metric(run_id, "artifact_bytes", sum((directory / name).stat().st_size for name in sha))
    try:
        api.create_registered_model(MODEL)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    version = str(
        api.create_model_version(
            MODEL,
            f"runs:/{run_id}/native",
            run_id,
            tags={"schema_version": str(meta["schema_version"]), "sha256": sha["model.json"]},
        ).version
    )
    api.set_registered_model_alias(MODEL, "challenger", version)
    return version


def evaluated(home: Path, run_id: str, report: dict[str, Any]) -> None:
    api = client(home)
    api.log_dict(run_id, report, "evaluation.json")
    for group in ("candidate", "incumbent", "intervals"):
        for metric, value in (report.get(group) or {}).items():
            if isinstance(value, (int, float)):
                api.log_metric(run_id, f"{group}.{metric}", float(value))
    if report.get("latency_ms") is not None:
        api.log_metric(run_id, "latency_ms", float(report["latency_ms"]))
    api.set_tag(run_id, "validation_passed", str(bool(report["passed"])).lower())
    api.set_tag(run_id, "validation_reasons", json.dumps(report["reasons"])[:4900])


def retrieve(home: Path, run_id: str, target: Path, expected: dict[str, str]) -> None:
    """Materialize the logged native files, verifying against the original training digests."""
    from spiritops.deployment.artifacts import ALLOWED, verify

    if set(expected) not in ({"model.json"}, ALLOWED):
        raise ValueError("invalid registered native file set")
    api = client(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        directory = Path(temporary) / "prepared"
        directory.mkdir()
        for name in expected:
            fetched = Path(api.download_artifacts(run_id, f"native/{name}", temporary))
            if not fetched.is_file() or fetched.is_symlink():
                raise ValueError("invalid retrieved file")
            shutil.copyfile(fetched, directory / name)
        verify(directory, expected)
        directory.rename(target)


def champion(home: Path, version: str) -> None:
    client(home).set_registered_model_alias(MODEL, "champion", version)
