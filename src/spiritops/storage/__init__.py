"""Transactional operational state shared by the single-host worker and API."""

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL CHECK(max_attempts >= 1),
    timeout_s REAL NOT NULL CHECK(timeout_s > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    lease REAL,
    token TEXT,
    error TEXT,
    result TEXT
);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status,created_at);
CREATE TABLE IF NOT EXISTS tracking_recovery (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    attempt INTEGER NOT NULL,
    run_status TEXT NOT NULL CHECK(run_status IN ('FINISHED','KILLED')),
    reason TEXT NOT NULL,
    PRIMARY KEY(job_id,attempt)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running ON jobs(status) WHERE status='running';
CREATE TABLE IF NOT EXISTS job_log (
    seq INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    created_at TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS log_by_job ON job_log(job_id,seq);
CREATE TABLE IF NOT EXISTS worker_health (
    id INTEGER PRIMARY KEY CHECK(id=1), seen REAL NOT NULL, heartbeat_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    directory TEXT NOT NULL UNIQUE,
    checksums TEXT NOT NULL,
    metadata TEXT NOT NULL,
    report TEXT NOT NULL,
    smoke_row TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0,1))
);
CREATE TABLE IF NOT EXISTS manifest (
    id INTEGER PRIMARY KEY CHECK(id=1),
    version TEXT NOT NULL REFERENCES models(version),
    generation INTEGER NOT NULL CHECK(generation>=1)
);
CREATE TABLE IF NOT EXISTS deployments (
    seq INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    version TEXT,
    previous_version TEXT,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    job_id TEXT REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS releases_by_version ON deployments(version,seq);
CREATE UNIQUE INDEX IF NOT EXISTS decision_by_job ON deployments(job_id)
    WHERE job_id IS NOT NULL AND outcome IN ('PROMOTE','REJECT_CANDIDATE');
"""

_ready: set[Path] = set()
_lock = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat()


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


@contextmanager
def database(home: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    path = home.resolve() / "control.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        fresh = not path.exists()
        db = sqlite3.connect(path, timeout=15, isolation_level=None)
        try:
            if fresh or path not in _ready:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(_SCHEMA)
                _ready.add(path)
        except BaseException:
            db.close()
            raise
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def log(db: sqlite3.Connection, job_id: str, message: str) -> None:
    db.execute(
        "INSERT INTO job_log(job_id,created_at,message) VALUES(?,?,?)",
        (job_id, now(), message),
    )
