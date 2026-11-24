"""Durable, transactionally paired predictions and weekly labels."""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from contextlib import contextmanager, closing
from functools import lru_cache
from pathlib import Path
from typing import Iterator

DDL = """
CREATE TABLE IF NOT EXISTS predictions(
 id TEXT PRIMARY KEY, product TEXT NOT NULL, week TEXT NOT NULL,
 version TEXT NOT NULL, prediction REAL NOT NULL CHECK(prediction>=0),
 seasonal REAL NOT NULL, lag1 REAL NOT NULL, segment TEXT NOT NULL,
 snapshot TEXT NOT NULL, source TEXT NOT NULL, created REAL NOT NULL,
 UNIQUE(product,week,version,source)
);
CREATE INDEX IF NOT EXISTS forecast_window ON predictions(version,week,product);
CREATE TABLE IF NOT EXISTS labels(product TEXT NOT NULL, week TEXT NOT NULL,
 actual REAL NOT NULL CHECK(actual>=0), received REAL NOT NULL,
 PRIMARY KEY(product,week));
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, at REAL NOT NULL,
 action TEXT NOT NULL, reason TEXT NOT NULL, event TEXT NOT NULL, job_id TEXT);
CREATE INDEX IF NOT EXISTS decision_window ON decisions(action,at);
CREATE TABLE IF NOT EXISTS pending(version TEXT PRIMARY KEY, trigger_week TEXT NOT NULL,
 validation_end TEXT NOT NULL, job_id TEXT);
CREATE TABLE IF NOT EXISTS traffic(id INTEGER PRIMARY KEY, at REAL NOT NULL,
 status INTEGER NOT NULL, latency_ms REAL NOT NULL);
CREATE INDEX IF NOT EXISTS recent_traffic ON traffic(at);
CREATE TABLE IF NOT EXISTS counters(singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 requests INTEGER NOT NULL, errors INTEGER NOT NULL);
INSERT OR IGNORE INTO counters VALUES(1,0,0);
"""


_schema_lock = threading.Lock()


@lru_cache(maxsize=32)
def _init(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=20)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(DDL)


@contextmanager
def db(home: Path) -> Iterator[sqlite3.Connection]:
    path = home / "observations.sqlite3"
    with _schema_lock:
        _init(path)
    with closing(sqlite3.connect(path, timeout=20)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            yield conn


def save_batch(home: Path, rows: list[dict]) -> list[dict]:
    result = []
    with db(home) as conn:
        for row in rows:
            conn.execute(
                """INSERT INTO predictions VALUES(:id,:product,:week,:version,:prediction,
                 :seasonal,:lag1,:segment,:snapshot,:source,:created)
                 ON CONFLICT(product,week,version,source) DO NOTHING""",
                row,
            )
            stored = conn.execute(
                """SELECT id AS prediction_id,product,week,version,prediction
                FROM predictions WHERE product=:product AND week=:week AND version=:version
                AND source=:source""",
                row,
            ).fetchone()
            result.append(dict(stored))
    return result


def label(home: Path, product: str, week: str, actual: float) -> bool:
    if not math.isfinite(actual) or actual < 0:
        raise ValueError("Actual litres must be finite and nonnegative")
    with db(home) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO labels VALUES(?,?,?,?)", (product, week, actual, time.time())
        )
        if cursor.rowcount == 1:
            return True
        stored = conn.execute(
            "SELECT actual FROM labels WHERE product=? AND week=?", (product, week)
        ).fetchone()[0]
        if stored != actual:
            raise ValueError("A different actual is already recorded for this product and week")
        return False


def traffic(home: Path, status: int, latency: float) -> None:
    now = time.time()
    with db(home) as conn:
        conn.execute(
            "INSERT INTO traffic(at,status,latency_ms) VALUES(?,?,?)", (now, status, latency)
        )
        conn.execute(
            "UPDATE counters SET requests=requests+1,errors=errors+?", (int(status >= 400),)
        )
        conn.execute("DELETE FROM traffic WHERE at<?", (now - 600,))


def recent(home: Path, table: str) -> list[dict]:
    queries = {
        "decisions": "SELECT * FROM decisions ORDER BY id DESC LIMIT 100",
        "observations": """SELECT p.product,p.week,p.version,p.prediction,p.source,l.actual,p.created
            FROM predictions p LEFT JOIN labels l USING(product,week)
            ORDER BY p.week DESC,p.created DESC LIMIT 2000""",
    }
    with db(home) as conn:
        return [dict(r) for r in conn.execute(queries[table])]


def note(home: Path, action: str, reason: str, event: str, job_id: str | None = None) -> None:
    with db(home) as conn:
        conn.execute(
            "INSERT INTO decisions(at,action,reason,event,job_id) VALUES(?,?,?,?,?)",
            (time.time(), action, reason, event, job_id),
        )
