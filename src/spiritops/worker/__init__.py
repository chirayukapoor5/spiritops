"""Durable local training queue. A leased subprocess owns each bounded training attempt."""

import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from spiritops.storage import database, encode, log, now

LOGGER = logging.getLogger(__name__)
LEASE_SECONDS = 30


def enqueue(home: Path, key: str, payload: dict[str, Any]) -> str:
    from datetime import date, timedelta

    if not key.strip() or len(key) > 512:
        raise ValueError("idempotency key must contain 1–512 characters")
    with database(home, write=True) as db:
        exists = db.execute("SELECT id FROM jobs WHERE key=?", (key,)).fetchone()
        if exists:
            return str(exists["id"])
        try:
            data = json.loads((home / "data/current.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("ingest a validated data snapshot before enqueuing training") from exc
        if set(payload) - {"kind", "validation_start", "validation_end"}:
            raise ValueError("unknown training job parameter")
        kind = payload.get("kind", "lightgbm")
        if kind not in ("baseline", "lightgbm"):
            raise ValueError("unsupported model kind")
        last = date.fromisoformat(data["last_week"])
        end = date.fromisoformat(payload.get("validation_end") or last.isoformat())
        start = date.fromisoformat(
            payload.get("validation_start") or (end - timedelta(weeks=7)).isoformat()
        )
        if start.weekday() or end.weekday() or start > end or end > last:
            raise ValueError("validation bounds must be ordered, complete Mondays in the snapshot")
        if start <= date.fromisoformat(data["product_selection_end"]):
            raise ValueError("validation must follow training-period product selection")
        details = {
            "kind": kind,
            "validation_start": str(start),
            "validation_end": str(end),
            "dataset_sha256": data["sha256"],
        }
        attempts = int(os.environ.get("SPIRITOPS_JOB_MAX_ATTEMPTS", "2"))
        timeout = float(os.environ.get("SPIRITOPS_JOB_TIMEOUT_SECONDS", "600"))
        if not 1 <= attempts <= 20 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid worker retry or timeout configuration")
        identifier = uuid.uuid4().hex
        timestamp = now()
        db.execute(
            "INSERT INTO jobs(id,key,payload,status,max_attempts,timeout_s,created_at,updated_at) "
            "VALUES(?,?,?,'queued',?,?,?,?)",
            (identifier, key, encode(details), attempts, timeout, timestamp, timestamp),
        )
        log(db, identifier, "queued against validated snapshot")
        return identifier


def _public(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in ("payload", "result"):
        result[key] = json.loads(result[key]) if result[key] else None
    for key in ("token", "lease"):
        result.pop(key, None)
    return result


def job(home: Path, id: str) -> dict[str, Any] | None:
    with database(home) as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (id,)).fetchone()
        if row is None:
            return None
        result = _public(row)
        result["logs"] = [
            dict(r)
            for r in db.execute(
                "SELECT created_at,message FROM job_log WHERE job_id=? ORDER BY seq", (id,)
            )
        ]
        return result


def jobs(home: Path) -> list[dict[str, Any]]:
    with database(home) as db:
        return [
            _public(r) for r in db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 500")
        ]


def _beat(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO worker_health VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET "
        "seen=excluded.seen,heartbeat_at=excluded.heartbeat_at",
        (time.time(), now()),
    )


def status(home: Path) -> dict[str, Any]:
    with database(home) as db:
        counts = {
            r["status"]: r["n"]
            for r in db.execute("SELECT status,COUNT(*) n FROM jobs GROUP BY status")
        }
        beat = db.execute("SELECT * FROM worker_health").fetchone()
        load_failures = db.execute(
            "SELECT COUNT(*) FROM deployments WHERE outcome IN "
            "('DEPLOYMENT_FAILED','ROLLBACK_FAILED') OR "
            "(outcome='ROLLBACK' AND reason LIKE 'active model%')"
        ).fetchone()[0]
    return {
        "available": bool(beat and time.time() - beat["seen"] < 90),
        "heartbeat_at": beat["heartbeat_at"] if beat else None,
        **{s: counts.get(s, 0) for s in ("queued", "running", "failed", "succeeded")},
        "model_loading_failures": load_failures,
    }


def _settle(db: sqlite3.Connection, row: sqlite3.Row, reason: str) -> None:
    decision = db.execute(
        "SELECT d.outcome,d.version,d.reason,m.run_id FROM deployments d JOIN models m "
        "ON d.version=m.version WHERE d.job_id=? AND d.outcome IN ('PROMOTE','REJECT_CANDIDATE')",
        (row["id"],),
    ).fetchone()
    if decision:
        result = {
            "outcome": decision["outcome"],
            "version": decision["version"],
            "run_id": decision["run_id"],
            "reasons": [decision["reason"]],
        }
        db.execute(
            "UPDATE jobs SET status='succeeded',result=?,error=NULL,lease=NULL,token=NULL,updated_at=? "
            "WHERE id=?",
            (encode(result), now(), row["id"]),
        )
        log(db, row["id"], "recovered committed release decision")
        return
    state = "queued" if row["attempt"] < row["max_attempts"] else "failed"
    db.execute(
        "UPDATE jobs SET status=?,error=?,updated_at=?,token=NULL,lease=NULL WHERE id=?",
        (state, reason, now(), row["id"]),
    )
    log(db, row["id"], f"{state}: {reason}")


def _claim(home: Path) -> dict[str, Any] | None:
    with database(home, write=True) as db:
        _beat(db)
        running = db.execute("SELECT * FROM jobs WHERE status='running'").fetchone()
        if running is not None:
            if running["lease"] > time.time():
                return None
            _settle(db, running, "worker lease expired")
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        token = uuid.uuid4().hex
        db.execute(
            "UPDATE jobs SET status='running',attempt=attempt+1,lease=?,token=?,updated_at=? WHERE id=?",
            (time.time() + LEASE_SECONDS, token, now(), row["id"]),
        )
        log(db, row["id"], f"claimed attempt {row['attempt'] + 1}")
        return {**dict(row), "attempt": row["attempt"] + 1, "token": token}


def _invoke(home: Path, row: dict[str, Any]) -> None:
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{row['id']}-{row['attempt']}.log"
    deadline = time.monotonic() + row["timeout_s"]
    args = [
        sys.executable,
        "-m",
        "spiritops.worker.execute",
        str(home.resolve()),
        row["id"],
        row["token"],
    ]
    with (
        path.open("ab") as output,
        subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT) as proc,
    ):
        try:
            while True:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    raise TimeoutError(f"training exceeded {row['timeout_s']} seconds")
                try:
                    code = proc.wait(timeout=min(wait, LEASE_SECONDS / 3))
                except subprocess.TimeoutExpired:
                    with database(home, write=True) as db:
                        _beat(db)
                        count = db.execute(
                            "UPDATE jobs SET lease=? WHERE id=? AND token=?",
                            (time.time() + LEASE_SECONDS, row["id"], row["token"]),
                        ).rowcount
                        if count == 0:
                            return
                    continue
                if code:
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 1800))
                        tail = stream.read().decode(errors="replace")
                    raise RuntimeError(f"training exited {code}: {tail}")
                return
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def run_once(home: Path) -> bool:
    row = _claim(home)
    if row is None:
        return False
    try:
        _invoke(home, row)
        reason = "training subprocess exited without recording completion"
    except Exception as exc:
        reason = str(exc)
    with database(home, write=True) as db:
        _beat(db)
        running = db.execute(
            "SELECT * FROM jobs WHERE id=? AND token=?", (row["id"], row["token"])
        ).fetchone()
        if running:
            _settle(db, running, reason)
    return True
