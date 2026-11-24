"""Monitoring compares immutable issued forecasts with independently matured labels."""

from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, time as wall_time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq
import psutil

from spiritops.config import snapshot, shift
from spiritops.features import panel
from spiritops.ingestion.sources import checksum
from spiritops.lifecycle import Policy, decide
from .observations import db, note

IOWA_TIME = ZoneInfo("America/Chicago")


def policy(home: Path) -> Policy:
    path = home / "policy.json"
    return Policy(**json.loads(path.read_text())) if path.exists() else Policy()


def as_of(home: Path) -> date:
    clock = home / "replay-clock.json"
    return (
        date.fromisoformat(json.loads(clock.read_text())["as_of"])
        if clock.exists()
        else datetime.now(IOWA_TIME).date()
    )


def latest_closed(home: Path) -> str:
    today = as_of(home)
    return (today - timedelta(days=today.weekday() + 7)).isoformat()


def _scope(home: Path) -> str:
    return "historical_replay" if (home / "replay-clock.json").is_file() else "snapshot_forecast"


# Old databases may contain snapshots scored after the reporting week closed.
_PROSPECTIVE = """(p.source='historical_replay'
    OR p.created<reporting_close(p.week))"""


@lru_cache(maxsize=512)
def reporting_close(week: str) -> float:
    closed = date.fromisoformat(week) + timedelta(weeks=1)
    return datetime.combine(closed, wall_time.min, IOWA_TIME).timestamp()


def _rows(home: Path, version: str) -> list[dict]:
    source = _scope(home)
    with db(home) as conn:
        conn.create_function("reporting_close", 1, reporting_close, deterministic=True)
        end = conn.execute(
            f"""SELECT max(p.week) FROM predictions p WHERE p.version=? AND p.week<=?
            AND p.source=? AND {_PROSPECTIVE}""",
            (version, latest_closed(home), source),
        ).fetchone()[0]
        if not end:
            return []
        return [
            dict(r)
            for r in conn.execute(
                f"""
            SELECT p.product,p.week,p.prediction,p.seasonal,p.lag1,p.segment,l.actual,i.lower,i.upper
            FROM predictions p LEFT JOIN intervals i USING(id) LEFT JOIN labels l USING(product,week)
            WHERE p.version=? AND p.week BETWEEN ? AND ? AND p.source=? AND {_PROSPECTIVE}
            ORDER BY p.week,p.product
            """,
                (version, shift(end, -7), end, source),
            )
        ]


def measures(rows: list[dict]) -> dict:
    observed = [r for r in rows if r["actual"] is not None]
    n = len(observed)
    total = sum(r["actual"] for r in observed)
    residual = [r["prediction"] - r["actual"] for r in observed]
    absolute = sum(abs(x) for x in residual)
    intervals = [r for r in observed if r.get("lower") is not None and r.get("upper") is not None]
    scale = sum(max(1, r["seasonal"], r["actual"]) for r in observed)
    return dict(
        samples=n,
        coverage=n / len(rows) if rows else 0,
        mae=absolute / n if n else None,
        wape=absolute / total if total else None,
        bias=sum(residual) / total if total else None,
        interval_samples=len(intervals),
        interval_coverage=sum(r["lower"] <= r["actual"] <= r["upper"] for r in intervals)
        / len(intervals)
        if intervals
        else None,
        seasonal_scaled_error=absolute / scale if scale else None,
    )


def performance(home: Path, model: dict) -> dict:
    rows = _rows(home, str(model["version"]))
    weeks = sorted({r["week"] for r in rows})
    groups = sorted({r["segment"] for r in rows})
    metrics = measures(rows)
    return {
        "version": model["version"],
        "source": _scope(home),
        "status": "available" if metrics["samples"] else "waiting_for_labels",
        "metrics": metrics,
        "weekly": [dict(week=w, **measures([r for r in rows if r["week"] == w])) for w in weeks],
        "segments": {g: measures([r for r in rows if r["segment"] == g]) for g in groups},
    }


@lru_cache(maxsize=8)
def reference_profile(path: Path, sha: str, end: str) -> dict[str, tuple[float, float]]:
    if checksum(path) != sha:
        raise ValueError("Drift reference checksum mismatch")
    frame = pq.read_table(path, columns=["product", "week", "volume"]).to_pandas()
    weeks = frame.week.astype(str).str.slice(0, 10)
    past = frame[(weeks <= end) & (weeks > shift(end, -12))].copy()
    past["logged"] = np.log1p(past.volume)
    return {
        str(product): (float(group.logged.mean()), float(group.logged.std(ddof=0)))
        for product, group in past.groupby("product")
    }


def drift(home: Path, model: dict) -> dict:
    manifest = snapshot(home)
    digest = model["dataset_sha256"]
    path = (
        home / manifest["path"]
        if digest == manifest["sha256"]
        else home / "data" / f"weekly-{digest}.parquet"
    )
    rules = policy(home)
    try:
        if not path.resolve().is_relative_to((home / "data").resolve()):
            raise ValueError("Invalid drift reference path")
        profile = reference_profile(path, digest, model["training_end"])
    except (ValueError, OSError):
        return {
            "feature": "log1p(lag1)",
            "samples": 0,
            "mean_standardized_shift": None,
            "threshold": rules.drift_z,
            "alert": False,
            "status": "reference_unavailable",
        }
    source = _scope(home)
    with db(home) as conn:
        conn.create_function("reporting_close", 1, reporting_close, deterministic=True)
        rows = conn.execute(
            f"""WITH cohort AS (
              SELECT product,lag1,week FROM predictions p WHERE p.version=? AND p.source=?
                AND {_PROSPECTIVE}
            ) SELECT product,lag1 FROM cohort
            WHERE week >= (SELECT date(MAX(week),'-21 days') FROM cohort)
            ORDER BY week DESC LIMIT 1000""",
            (model["version"], source),
        ).fetchall()
    scores = [
        (math.log1p(r["lag1"]) - profile[r["product"]][0]) / max(0.25, profile[r["product"]][1])
        for r in rows
        if r["product"] in profile
    ]
    z = sum(scores) / len(scores) if scores else None
    return {
        "feature": "log1p(lag1)",
        "samples": len(scores),
        "mean_standardized_shift": z,
        "threshold": rules.drift_z,
        "alert": len(scores) >= rules.min_samples and z is not None and abs(z) > rules.drift_z,
        "status": "available" if scores else "waiting_for_predictions",
    }


def operations(home: Path) -> dict:
    from spiritops.worker import status

    with db(home) as conn:
        totals = dict(conn.execute("SELECT requests,errors FROM counters").fetchone())
        latencies = [
            r[0]
            for r in conn.execute("SELECT latency_ms FROM traffic WHERE at>=?", (time.time() - 60,))
        ]
    return {
        **totals,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "throughput_per_second": len(latencies) / 60,
        "rss_mb": psutil.Process().memory_info().rss / 1024**2,
        "worker": status(home),
    }


def mature(home: Path, manifest: dict) -> int:
    """Attach published outcomes only after both reporting and replay clocks have closed."""
    cutoff = min(latest_closed(home), manifest["last_week"])
    with db(home) as conn:
        keys = conn.execute(
            """SELECT DISTINCT p.product,p.week FROM predictions p
            LEFT JOIN labels l USING(product,week) WHERE l.week IS NULL AND p.week<=?
            ORDER BY p.week,p.product LIMIT 4000""",
            (cutoff,),
        ).fetchall()
    if not keys:
        return 0
    history = panel(home, manifest)
    records = []
    for row in keys:
        series = history.series.get(row["product"])
        week = date.fromisoformat(row["week"])
        if series is not None and history.first <= week <= history.last:
            index = (week - history.first).days // 7
            records.append((row["product"], row["week"], float(series.volume[index]), time.time()))
    with db(home) as conn:
        conn.executemany("INSERT OR IGNORE INTO labels VALUES(?,?,?,?)", records)
    return len(records)


def tick(home: Path) -> dict:
    from spiritops import deployment, worker
    from spiritops.features import schema_features

    try:
        model = deployment.current(home)
        manifest = snapshot(home)
    except RuntimeError as exc:
        note(home, "HOLD", str(exc), "bootstrap")
        return {"action": "HOLD", "reason": str(exc)}
    rules = policy(home)
    try:
        panel(home, manifest)
        mature(home, manifest)
    except (ValueError, OSError) as exc:
        reason = f"Published observations failed snapshot verification: {exc}"
        note(home, "QUARANTINE", reason, f"snapshot:{manifest['sha256']}")
        return {"action": "QUARANTINE", "reason": reason}
    perf = performance(home, model)
    windows = perf["weekly"]
    last = windows[-1]["week"] if windows else None
    event = f"monitor:{model['version']}:{last}"
    quality = None
    try:
        schema_features(model["schema_version"])
    except ValueError:
        quality = "The active feature schema has changed"
    if (as_of(home) - date.fromisoformat(shift(manifest["last_week"]))).days > rules.max_lag_days:
        quality = "The weekly dataset is stale"
    recent = windows[-rules.consecutive_weeks :]
    if any(shift(a["week"]) != b["week"] for a, b in zip(recent, recent[1:])):
        quality = "Monitoring periods are discontinuous"
    state = worker.status(home)
    with db(home) as conn:
        previous = conn.execute(
            "SELECT COUNT(*),MAX(at) FROM decisions WHERE action='RETRAIN' AND at>?",
            (time.time() - 86400,),
        ).fetchone()
        last_retrain = conn.execute(
            "SELECT MAX(at) FROM decisions WHERE action='RETRAIN'"
        ).fetchone()[0]
        existing = conn.execute(
            "SELECT job_id FROM decisions WHERE action='RETRAIN' AND event=?", (event,)
        ).fetchone()
    busy = bool(state.get("running") or state.get("queued") or existing)
    spent = previous[0] >= rules.daily_budget
    cooldown = bool(last_retrain and time.time() - last_retrain < 3600 * rules.cooldown_hours)
    action, reason = decide(
        rules,
        windows,
        model.get("evaluation", {}).get("candidate", {}).get("seasonal_scaled_error"),
        quality=quality,
        drift=drift(home, model)["alert"],
        busy=busy,
        spent=spent,
        cooldown=cooldown,
        trainable=True,
    )
    job_id = None
    with db(home) as conn:
        if action == "RETRAIN" and last:
            conn.execute(
                "INSERT OR IGNORE INTO pending(version,trigger_week,validation_end) VALUES(?,?,?) ON CONFLICT(version) DO UPDATE SET trigger_week=excluded.trigger_week,validation_end=excluded.validation_end,job_id=NULL WHERE pending.job_id IS NOT NULL AND excluded.trigger_week>pending.validation_end",
                (model["version"], last, shift(last, rules.validation_weeks)),
            )
        pending = conn.execute(
            "SELECT * FROM pending WHERE version=? AND job_id IS NULL", (model["version"],)
        ).fetchone()
    if pending and not quality and not (busy or cooldown or spent):
        start = shift(pending["trigger_week"])
        end = pending["validation_end"]
        event = f"monitor:{model['version']}:{pending['trigger_week']}"
        if min(latest_closed(home), manifest["last_week"]) < end:
            action, reason = (
                "WAIT_FOR_LABELS",
                f"Reserved independent validation period matures after {end}",
            )
        else:
            job_id = worker.enqueue(
                home, event, {"kind": "lightgbm", "validation_start": start, "validation_end": end}
            )
            with db(home) as conn:
                conn.execute(
                    "UPDATE pending SET job_id=? WHERE version=?", (job_id, model["version"])
                )
            action, reason = (
                "RETRAIN",
                "Persistent degradation confirmed; reserved chronological holdout matured",
            )
    elif action == "RETRAIN":
        action, reason = "HOLD", "A job already evaluated this model and degradation event"
    note(home, action, reason, event, job_id)
    return {
        "action": action,
        "reason": reason,
        "event": event,
        "job_id": job_id,
        "performance": perf,
    }
