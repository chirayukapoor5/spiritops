import concurrent.futures
import json
import math
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from spiritops import deployment, worker
from spiritops.deployment.artifacts import checked_directory, checksums, verify
from spiritops.storage import database, encode


@pytest.fixture
def home(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "current.json").write_text(
        encode({"last_week": "2024-12-30", "product_selection_end": "2024-03-25", "sha256": "abc"})
    )
    return tmp_path


def test_concurrent_idempotency_timeout_and_recovery(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: worker.enqueue(home, "same event", {}), range(24)))
    assert len(set(ids)) == 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        claims = list(pool.map(lambda _: worker._claim(home), range(5)))
    assert sum(x is not None for x in claims) == 1
    assert worker.job(home, ids[0])["status"] == "running"
    with database(home, write=True) as db:
        db.execute("UPDATE jobs SET lease=0,timeout_s=0.001")
    assert worker.run_once(home)
    result = worker.job(home, ids[0])
    assert result["status"] == "failed" and result["attempt"] == 2
    assert "exceeded" in result["error"]
    assert "worker lease expired" in str(result["logs"])
    assert not worker.run_once(home)
    assert worker.status(home)["failed"] == 1
    assert worker.status(home)["available"]
    assert len(worker.jobs(home)) == 1 and worker.job(home, "missing") is None


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "pickle"},
        {"x": 1},
        {"validation_end": "2024-12-31"},
        {"validation_start": "2024-01-01"},
    ],
)
def test_invalid_jobs_cannot_enter_queue(home: Path, payload: dict) -> None:
    with pytest.raises(ValueError):
        worker.enqueue(home, "event", payload)
    assert worker.jobs(home) == []


@pytest.fixture
def predictors(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    hooks = SimpleNamespace(started=threading.Event(), resume=threading.Event())
    features = ModuleType("spiritops.features")
    monkeypatch.setattr(features, "SCHEMA_VERSION", "test-schema", raising=False)
    training = ModuleType("spiritops.training")

    def load(path: Path) -> Any:
        value = json.loads((path / "model.json").read_text())["value"]

        class Predictor:
            def predict(self, rows: list[dict[str, float]]) -> list[float]:
                if any(row.get("pause") for row in rows):
                    hooks.started.set()
                    assert hooks.resume.wait(15)
                return [math.nan if row.get("bad") and value == 2 else value for row in rows]

        return Predictor()

    monkeypatch.setattr(training, "load", load, raising=False)
    monkeypatch.setitem(sys.modules, "spiritops.features", features)
    monkeypatch.setitem(sys.modules, "spiritops.training", training)
    monkeypatch.setattr(deployment, "reconcile", lambda _: None)
    return hooks


def candidate(
    home: Path, version: str, gen: int, *, passed: bool = True, corrupt: bool = False
) -> dict:
    id = worker.enqueue(home, "model " + version, {})
    row = worker._claim(home)
    assert row is not None
    directory = home / "models" / version
    directory.mkdir(parents=True)
    meta = {"schema_version": "test-schema", "kind": "baseline", "value": int(version)}
    (directory / "model.json").write_text(encode(meta))
    sha = checksums(directory)
    if corrupt:
        (directory / "model.json").write_text("{}")
    try:
        return deployment.release(
            home,
            version,
            "run" + version,
            directory,
            sha,
            meta,
            {"passed": passed, "reasons": [] if passed else ["WAPE degraded"]},
            {"x": 1},
            gen,
            id,
            row["token"],
        )
    finally:
        with database(home, write=True) as db:
            db.execute("UPDATE jobs SET status='succeeded',token=NULL,lease=NULL WHERE id=?", (id,))


def test_batch_pins_version_while_promoting(home: Path, predictors: SimpleNamespace) -> None:
    candidate(home, "1", 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(deployment.predict, home, [{"pause": 1.0}, {"x": 2.0}])
        try:
            assert predictors.started.wait(15)
            assert candidate(home, "2", 1)["outcome"] == "PROMOTE"
            assert deployment.predict(home, [{"x": 0.0}]) == ("2", [2.0])
        finally:
            predictors.resume.set()
        assert slow.result() == ("1", [1.0, 1.0])
    assert deployment.current(home)["version"] == "2"
    assert deployment.predict(home, [{"bad": 1.0}]) == ("1", [1.0])
    assert deployment.history(home)[0]["outcome"] == "ROLLBACK"


def test_gates_corruption_and_restart(home: Path, predictors: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="no active"):
        deployment.current(home)
    assert deployment.incumbent(home) == (None, 0)
    candidate(home, "1", 0)
    assert candidate(home, "2", 1, passed=False)["outcome"] == "REJECT_CANDIDATE"
    assert candidate(home, "3", 0)["outcome"] == "REJECT_CANDIDATE"
    with pytest.raises(ValueError, match="checksum"):
        candidate(home, "4", 1, corrupt=True)
    assert deployment.current(home)["version"] == "1"
    candidate(home, "5", 1)
    assert deployment.incumbent(home)[1] == 2
    (home / "models/5/model.json").unlink()
    deployment._cache.clear()  # emulate a fresh API process
    assert deployment.predict(home, [{"foo": 1.0}]) == ("1", [1.0])
    assert deployment.history(home)[0]["previous_version"] == "5"
    assert deployment.history(home)[0]["outcome"] == "ROLLBACK"
    with pytest.raises(RuntimeError, match="no previous compatible"):
        deployment.rollback(home, "manual recovery requested")
    assert worker.status(home)["model_loading_failures"] >= 1


def test_reject_unexpected_native_files(home: Path, predictors: SimpleNamespace) -> None:
    d = home / "models" / "1"
    d.mkdir(parents=True)
    (d / "model.json").write_text('{"kind":"baseline","schema_version":"outdated"}')
    with pytest.raises(ValueError, match="incompatible"):
        verify(d, checksums(d))
    (d / "model.pkl").write_text("untrusted fixture")
    with pytest.raises(ValueError, match="unexpected"):
        checksums(d)
    with pytest.raises(ValueError, match="escapes"):
        checked_directory(home, "../../tmp/artifact")
