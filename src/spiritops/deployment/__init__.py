"""Atomic local releases. A request holds its predictor until the entire batch completes."""

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spiritops.deployment.artifacts import checked_directory, native_files, verify
from spiritops.storage import database, encode, now

LOGGER = logging.getLogger(__name__)


@dataclass
class Loaded:
    version: str
    predictor: Any
    signature: tuple[tuple[str, int, int, int], ...]
    info: dict[str, Any]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def predict(self, rows: list[dict[str, float]]) -> list[float]:
        with self.lock:
            values = [float(v) for v in self.predictor.predict(rows)]
        if len(rows) != len(values) or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid model predictions")
        return values


_cache: dict[Path, Loaded] = {}
_cache_lock = threading.Lock()


def _signature(directory: Path) -> tuple[tuple[str, int, int, int], ...]:
    result = []
    for path in native_files(directory):
        stat = path.stat()
        result.append((path.name, stat.st_size, stat.st_mtime_ns, stat.st_ino))
    return tuple(result)


def _load(home: Path, row: sqlite3.Row) -> Loaded:
    from spiritops.training import load

    directory = checked_directory(home, row["directory"])
    signature = _signature(directory)
    with _cache_lock:
        incumbent = _cache.get(home.resolve())
        if incumbent and incumbent.version == row["version"] and incumbent.signature == signature:
            return incumbent
    meta = verify(directory, json.loads(row["checksums"]))
    model = Loaded(
        row["version"],
        load(directory),
        signature,
        {
            **meta,
            "version": row["version"],
            "run_id": row["run_id"],
            "evaluation": json.loads(row["report"]),
        },
    )
    model.predict([json.loads(row["smoke_row"])])
    if signature != _signature(directory):
        raise ValueError("native artifact changed during preparation")
    return model


def _active(db: sqlite3.Connection) -> sqlite3.Row | None:
    return db.execute(
        "SELECT m.*,s.generation FROM manifest s JOIN models m ON s.version=m.version"
    ).fetchone()


def _event(
    db: sqlite3.Connection,
    version: str | None,
    previous: str | None,
    outcome: str,
    reason: str,
    job_id: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO deployments(created_at,version,previous_version,outcome,reason,job_id) VALUES(?,?,?,?,?,?)",
        (now(), version, previous, outcome, reason, job_id),
    )


def _switch(
    db: sqlite3.Connection,
    version: str,
    previous: str | None,
    outcome: str,
    reason: str,
    job_id: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO manifest VALUES(1,?,1) ON CONFLICT(id) DO UPDATE SET "
        "version=excluded.version,generation=generation+1",
        (version,),
    )
    db.execute("UPDATE models SET approved=1 WHERE version=?", (version,))
    _event(db, version, previous, outcome, reason, job_id)


def reconcile(home: Path) -> None:
    """Repair the alias after a committed promotion or process interruption."""
    from spiritops import registry

    # Compare after the external call so an overlapping alias write cannot win permanently.
    for _ in range(3):
        with database(home) as db:
            first = db.execute("SELECT * FROM manifest").fetchone()
        if first is None:
            return
        try:
            registry.champion(home, first["version"])
        except Exception:
            LOGGER.exception("champion alias reconciliation deferred")
            return
        with database(home) as db:
            second = db.execute("SELECT generation FROM manifest").fetchone()
        if second is not None and second[0] == first["generation"]:
            return


def history(home: Path) -> list[dict[str, Any]]:
    with database(home) as db:
        rows = db.execute(
            "SELECT d.*,m.run_id,m.metadata,m.report FROM deployments d LEFT JOIN models m "
            "ON d.version=m.version ORDER BY d.seq DESC LIMIT 500"
        )
        result = []
        for row in rows:
            item = dict(row)
            meta = json.loads(item.pop("metadata") or "{}")
            item["kind"] = meta.get("kind")
            item["evaluation"] = json.loads(item.pop("report") or "{}")
            result.append(item)
        return result


def _restore(home: Path, reason: str, failed_generation: int | None) -> Loaded:
    with database(home) as db:
        old = _active(db)
        if old is None:
            raise RuntimeError("no active deployment")
        candidates = db.execute(
            "SELECT m.* FROM models m JOIN deployments d ON m.version=d.version "
            "WHERE m.approved=1 AND m.version<>? AND d.outcome IN ('PROMOTE','ROLLBACK') "
            "GROUP BY m.version ORDER BY MAX(d.seq) DESC",
            (old["version"],),
        ).fetchall()
    if failed_generation is not None and old["generation"] != failed_generation:
        return _serving(home)
    failures = []
    for row in candidates:
        try:
            model = _load(home, row)
        except Exception as exc:
            failures.append(f"{row['version']}: {exc}")
            continue
        with database(home, write=True) as db:
            actual = _active(db)
            changed = actual is None or old["generation"] != actual["generation"]
            if not changed:
                _switch(db, model.version, old["version"], "ROLLBACK", reason)
        if changed:
            return _serving(home)
        with _cache_lock:
            _cache[home.resolve()] = model
        reconcile(home)
        return model
    message = f"{reason}; no previous compatible release"
    if failures:
        message += f" ({', '.join(failures)})"
    with database(home, write=True) as db:
        # Collapse identical repeated probes on an unhealthy deployment.
        last = db.execute("SELECT reason FROM deployments ORDER BY seq DESC LIMIT 1").fetchone()
        if not last or last["reason"] != message:
            _event(db, old["version"], old["version"], "ROLLBACK_FAILED", message)
    raise RuntimeError(message)


def rollback(home: Path, reason: str) -> dict[str, Any]:
    return dict(_restore(home, reason, None).info)


def _serving(home: Path) -> Loaded:
    with database(home) as db:
        row = _active(db)
    if row is None:
        raise RuntimeError("no active deployment")
    try:
        model = _load(home, row)
    except Exception as exc:
        return _restore(
            home, f"active model verification or smoke failed: {exc}", row["generation"]
        )
    with _cache_lock:
        _cache[home.resolve()] = model
    return model


def current(home: Path) -> dict[str, Any]:
    return dict(_serving(home).info)


def predict_with_metadata(
    home: Path, rows: list[dict[str, float]]
) -> tuple[dict[str, Any], list[float]]:
    model = _serving(home)
    try:
        return dict(model.info), model.predict(rows)
    except Exception as failure:
        with database(home) as db:
            state = _active(db)
        if state is not None and state["version"] == model.version:
            model = _restore(home, f"serving health check failed: {failure}", state["generation"])
        else:
            model = _serving(home)
        return dict(model.info), model.predict(rows)


def predict(home: Path, rows: list[dict[str, float]]) -> tuple[str, list[float]]:
    info, values = predict_with_metadata(home, rows)
    return str(info["version"]), values


def incumbent(home: Path) -> tuple[Path | None, int]:
    with database(home) as db:
        row = _active(db)
    if row is None:
        return None, 0
    directory = checked_directory(home, row["directory"])
    verify(directory, json.loads(row["checksums"]))
    return directory, int(row["generation"])


def _owned(db: sqlite3.Connection, job_id: str, token: str) -> None:
    if not db.execute(
        "SELECT 1 FROM jobs WHERE id=? AND status='running' AND token=? AND lease>?",
        (job_id, token, time.time()),
    ).fetchone():
        raise RuntimeError("worker lease expired before deployment")


def release(
    home: Path,
    version: str,
    run_id: str,
    directory: Path,
    sha: dict[str, str],
    meta: dict[str, Any],
    report: dict[str, Any],
    smoke_row: dict[str, float],
    evaluated_generation: int,
    job_id: str,
    token: str,
) -> dict[str, Any]:
    relative = directory.resolve().relative_to(home.resolve()).as_posix()
    with database(home, write=True) as db:
        _owned(db, job_id, token)
        db.execute(
            "INSERT INTO models(version,run_id,directory,checksums,metadata,report,smoke_row) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                version,
                run_id,
                relative,
                encode(sha),
                encode(meta),
                encode(report),
                encode(smoke_row),
            ),
        )
        row = db.execute("SELECT * FROM models WHERE version=?", (version,)).fetchone()
    try:
        _load(home, row)
    except Exception as exc:
        with database(home, write=True) as db:
            active = _active(db)
            _event(
                db,
                version,
                active["version"] if active else None,
                "DEPLOYMENT_FAILED",
                f"candidate verification or smoke failed: {exc}",
                job_id,
            )
        raise
    reasons = list(report["reasons"])
    with database(home, write=True) as db:
        _owned(db, job_id, token)
        active = _active(db)
        previous = active["version"] if active else None
        generation = active["generation"] if active else 0
        if generation != evaluated_generation:
            reasons.append("serving generation changed during evaluation")
        if report["passed"] and generation == evaluated_generation:
            outcome = "PROMOTE"
            _switch(db, version, previous, outcome, "release gates passed", job_id)
        else:
            outcome = "REJECT_CANDIDATE"
            _event(
                db, version, previous, outcome, "; ".join(reasons) or "release gate failed", job_id
            )
    if outcome == "PROMOTE":
        loaded = current(home)
        if loaded["version"] != version:
            raise RuntimeError("release failed post-activation health confirmation")
        reconcile(home)
    return {"outcome": outcome, "version": version, "run_id": run_id, "reasons": reasons}
