import json
from pathlib import Path

import pytest

from spiritops import registry
from spiritops.deployment.artifacts import checksums


def test_real_sqlite_tracking_registration_and_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    artifact = tmp_path / "native-source"
    artifact.mkdir()
    meta = {
        "kind": "baseline",
        "schema_version": "registry-fixture",
        "dataset_sha256": "f" * 64,
        "training_start": "2024-01-01",
        "training_end": "2024-12-02",
        "feature_names": ["seasonal_naive"],
        "parameters": {},
        "duration_seconds": 0.1,
        "training_rows": 48,
    }
    (artifact / "model.json").write_text(json.dumps(meta))
    run = registry.start(tmp_path, "registry-test", 1, {"kind": "baseline"})
    version = registry.register(tmp_path, run, artifact, meta, checksums(artifact))
    report = {
        "passed": True,
        "reasons": [],
        "candidate": {"wape": 0.2},
        "incumbent": {"wape": 0.2},
        "latency_ms": 1.0,
    }
    registry.evaluated(tmp_path, run, report)
    registry.champion(tmp_path, version)
    api = registry.client(tmp_path)
    api.set_terminated(run)
    persisted = registry.client(tmp_path)
    assert persisted.get_run(run).info.status == "FINISHED"
    assert persisted.get_run(run).data.metrics["candidate.wape"] == 0.2
    assert {item.path for item in persisted.list_artifacts(run)} >= {
        "native",
        "evaluation.json",
        "signature.json",
        "dependencies.json",
        "integrity.json",
    }
    for alias in ("champion", "challenger"):
        model = persisted.get_model_version_by_alias(registry.MODEL, alias)
        assert str(model.version) == version and model.run_id == run
