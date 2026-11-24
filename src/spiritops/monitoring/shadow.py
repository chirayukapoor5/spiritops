"""Issue shadow forecasts before labels arrive, using persisted as-of inputs."""

import json
import time
from datetime import datetime
from pathlib import Path

from spiritops.deployment.artifacts import checked_directory, verify
from spiritops.monitoring import latest_closed
from spiritops.storage import database
from spiritops.training import load
from .observations import db


def collect(home: Path) -> None:
    with database(home) as control:
        release = control.execute("""SELECT m.*,d.created_at FROM models m JOIN deployments d
            USING(version) WHERE d.outcome='REJECT_CANDIDATE' ORDER BY d.seq DESC LIMIT 1""").fetchone()
    if release is None:
        return
    info = json.loads(release["metadata"])
    version = release["version"]
    published = datetime.fromisoformat(release["created_at"]).timestamp()
    with db(home) as conn:
        rows = conn.execute(
            """SELECT p.id,f.features FROM predictions p JOIN inputs f USING(id)
            LEFT JOIN prospective_shadow s ON s.id=p.id AND s.challenger=?
            LEFT JOIN labels l USING(product,week)
            WHERE s.id IS NULL AND l.week IS NULL AND p.created>=? AND p.week>? AND p.version<>?
            ORDER BY p.created LIMIT 500""",
            (version, published, max(info["validation_end"], latest_closed(home)), version),
        ).fetchall()
    if rows:
        directory = checked_directory(home, release["directory"])
        verify(directory, json.loads(release["checksums"]))
        values = load(directory).predict([json.loads(r["features"]) for r in rows])
        with db(home) as conn:
            cutoff = latest_closed(home)
            conn.executemany(
                """INSERT OR IGNORE INTO prospective_shadow SELECT p.id,?,?,?
                FROM predictions p LEFT JOIN labels l USING(product,week)
                WHERE p.id=? AND p.week>? AND l.week IS NULL""",
                [
                    (version, v, time.time(), r["id"], cutoff)
                    for r, v in zip(rows, values, strict=True)
                ],
            )


def summary(home: Path) -> list[dict]:
    with db(home) as conn:
        return [
            dict(r)
            for r in conn.execute("""SELECT s.challenger,p.version AS champion,p.source,
            COUNT(*) AS issued,COUNT(l.actual) AS matured, MIN(p.week) AS first_week,
            MAX(p.week) AS last_week,
            AVG(ABS(s.value-l.actual)) AS challenger_mae,
            AVG(ABS(p.prediction-l.actual)) AS champion_mae,
            SUM(ABS(s.value-l.actual))/NULLIF(SUM(l.actual),0) AS challenger_wape,
            SUM(ABS(p.prediction-l.actual))/NULLIF(SUM(l.actual),0) AS champion_wape
            FROM prospective_shadow s JOIN predictions p USING(id)
            LEFT JOIN labels l ON l.product=p.product AND l.week=p.week AND l.received>=s.scored_at
            GROUP BY s.challenger,p.version,p.source
            ORDER BY s.challenger,p.version,p.source""")
        ]
