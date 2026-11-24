"""Isolated training process. Run only with a valid durable job lease."""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from spiritops import deployment, registry
from spiritops.deployment.artifacts import checksums, verify
from spiritops.storage import database, encode, log, now


def _stage(home: Path, id: str, token: str, message: str) -> None:
    with database(home, write=True) as db:
        if not db.execute(
            "SELECT 1 FROM jobs WHERE id=? AND token=? AND lease>?", (id, token, time.time())
        ).fetchone():
            raise RuntimeError("job lease lost")
        log(db, id, message)


def _snapshot(home: Path, view: Path, expected: str) -> dict[str, Any]:
    raw = (home / "data/current.json").read_bytes()
    manifest: dict[str, Any] = json.loads(raw)
    if manifest["sha256"] != expected:
        raise ValueError("queued dataset has changed; enqueue a job for the current snapshot")
    relative = Path(manifest["path"])
    source = (home / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not source.is_relative_to(home.resolve())
    ):
        raise ValueError("invalid curated dataset path")
    dest = view / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)
    (view / "data").mkdir(exist_ok=True)
    (view / "data/current.json").write_bytes(raw)
    return manifest


def execute(home: Path, id: str, token: str) -> None:
    from spiritops.evaluation import evaluate
    from spiritops.features import features_for
    from spiritops.training import fit

    with database(home) as db:
        row = db.execute(
            "SELECT * FROM jobs WHERE id=? AND token=? AND status='running' AND lease>?",
            (id, token, time.time()),
        ).fetchone()
    if row is None:
        raise RuntimeError("job is no longer owned by this worker")
    payload = json.loads(row["payload"])
    start, end = payload["validation_start"], payload["validation_end"]
    run = registry.start(home, id, row["attempt"], payload)
    try:
        _stage(home, id, token, f"MLflow run {run}")
        artifact = home / "models" / f"{id}-{row['attempt']}-{token[:8]}"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home, prefix="training-") as working:
            view = Path(working)
            snapshot = _snapshot(home, view, payload["dataset_sha256"])
            _stage(home, id, token, "snapshot pinned; fitting model")
            meta = fit(view, artifact, start, end, kind=payload["kind"])
            sha = checksums(artifact)
            verify(artifact, sha)
            if meta["dataset_sha256"] != snapshot["sha256"] or meta["training_end"] >= start:
                raise ValueError("training data version or chronological cutoff mismatch")
            version = registry.register(home, run, artifact, meta, sha)
            _stage(home, id, token, f"registered challenger {version}; evaluating release gates")
            incumbent, generation = deployment.incumbent(home)
            report = evaluate(view, artifact, incumbent, start, end)
            registry.evaluated(home, run, report)
            smoke = features_for(view, snapshot["products"][0], start)
            result = deployment.release(
                home, version, run, artifact, sha, meta, report, smoke, generation, id, token
            )
        api = registry.client(home)
        api.set_tag(run, "release_outcome", result["outcome"])
        api.set_terminated(run)
        with database(home, write=True) as db:
            changed = db.execute(
                "UPDATE jobs SET status='succeeded',result=?,error=NULL,updated_at=?,token=NULL,lease=NULL "
                "WHERE id=? AND token=?",
                (encode(result), now(), id, token),
            ).rowcount
            if changed:
                log(db, id, result["outcome"])
    except BaseException as exc:
        registry.client(home).set_tag(run, "failure", str(exc)[:4000])
        registry.client(home).set_terminated(run, status="FAILED")
        raise


if __name__ == "__main__":
    execute(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
