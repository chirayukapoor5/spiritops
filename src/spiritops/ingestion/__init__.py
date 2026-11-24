"""Build complete-week immutable product panels from native Iowa transactions."""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import psutil

from spiritops.ingestion.sources import (
    FIELDS,
    YEARS,
    atomic_json,
    checksum,
    fetch,
    project_archive,
)
from spiritops.validation import REQUIRED, validate

RESOURCES = Path(__file__).parent / "resources"
PIPELINE = "weekly-gross-purchases-v1"


def ingest(home: Path, mode: str = "demo") -> dict[str, Any]:
    if mode == "demo":
        provenance = json.loads((RESOURCES / "demo.json").read_text())
        part = RESOURCES / "demo.csv.gz"
        if checksum(part) != provenance["sha256"]:
            raise ValueError("bundled Iowa data failed checksum verification")
        return ingest_files(home, [part], date(2023, 1, 1), date(2024, 12, 31), provenance)
    if mode != "full":
        raise ValueError("use demo or full")
    raw = home / "data" / "raw"
    parts: list[Path] = []
    records = []
    for year in YEARS:
        path, info = fetch(raw, year)
        records.append(info)
        parts.extend(project_archive(path, raw / info["sha256"]))
    return ingest_files(
        home, parts, date(2023, 1, 1), date(2025, 12, 31), {"mode": "full", "upstream": records}
    )


def _build(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    first: date,
    last: date,
    selection_end: date,
    workspace: Path,
) -> dict[str, Any]:
    con.execute(
        "CREATE TABLE raw AS SELECT "
        + ", ".join(f"NULL::VARCHAR AS {f}" for f in sorted(REQUIRED))
        + " WHERE false"
    )
    for path in files:
        rel = (
            con.read_parquet(str(path))
            if path.suffix == ".parquet"
            else con.read_csv(str(path), all_varchar=True, header=True)
        )
        if missing := REQUIRED.difference(rel.columns):
            raise ValueError(f"missing mandatory Iowa fields: {sorted(missing)}")
        rel.project(FIELDS).insert_into("raw")
    validate(con)
    report = dict(
        con.execute("SELECT outcome, count(*) FROM classified GROUP BY outcome").fetchall()
    )
    con.sql("SELECT * FROM classified WHERE outcome != 'accepted'").write_parquet(
        str(workspace / "quarantine.parquet")
    )
    con.execute(
        """
        CREATE TEMP TABLE eligible AS
        SELECT * FROM classified WHERE outcome='accepted'
            AND day BETWEEN ? AND ? + INTERVAL 6 DAY
    """,
        [first, last],
    )
    con.execute(
        """
        CREATE TEMP TABLE selected AS SELECT item_no AS product FROM eligible
        WHERE day < ? + INTERVAL 7 DAY GROUP BY item_no HAVING sum(greatest(liters,0))>0
        ORDER BY sum(greatest(liters,0)) DESC, item_no LIMIT 100
    """,
        [selection_end],
    )
    products = [
        r[0] for r in con.execute("SELECT product FROM selected ORDER BY product").fetchall()
    ]
    if not products:
        raise ValueError("initial training period contains no purchasable products")
    con.execute(
        """
        CREATE TEMP TABLE panel AS
        WITH sums AS (
            SELECT item_no AS product, date_trunc('week', day)::DATE AS week,
                round(sum(greatest(liters,0)), 6) AS volume,
                round(sum(greatest(-liters,0)), 6) AS returns,
                arg_max(category_code, (day, invoice_id)) AS category,
                sum(dollars) FILTER (WHERE dollars>=0 AND liters>0) /
                nullif(sum(liters) FILTER (WHERE dollars>=0 AND liters>0),0) AS price,
                count(*) FILTER (WHERE liters>0) AS purchases
            FROM eligible JOIN selected ON item_no=product GROUP BY ALL
        )
        SELECT s.product, d.week::DATE AS week, coalesce(a.volume,0.0)::DOUBLE AS volume,
            a.category, a.price, coalesce(a.purchases,0) AS purchases,
            coalesce(a.returns,0.0)::DOUBLE AS returns
        FROM selected s CROSS JOIN generate_series(?::DATE, ?::DATE, INTERVAL 7 DAY) d(week)
        LEFT JOIN sums a ON a.product=s.product AND a.week=d.week
        ORDER BY d.week, s.product
    """,
        [first, last],
    )
    con.table("panel").write_parquet(str(workspace / "weekly.parquet"))
    returns, litres, unknown_category, missing_amount = con.execute("""
        SELECT count(*) FILTER (WHERE liters<0),coalesce(sum(-liters) FILTER (WHERE liters<0),0),
          count(*) FILTER (WHERE category_code IS NULL), count(*) FILTER (WHERE dollars IS NULL)
        FROM eligible""").fetchone() or (0, 0, 0, 0)
    return {
        "products": products,
        "outcomes": report,
        "return_lines": returns,
        "returned_litres": litres,
        "missing_category_lines": unknown_category,
        "missing_amount_lines": missing_amount,
        "source_rows": sum(report.values()),
    }


def ingest_files(
    home: Path,
    files: list[Path],
    first_day: date,
    last_day: date,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Input coverage dates describe the source contract, including days with no orders."""
    if not files or last_day < first_day:
        raise ValueError("provide at least one file and an ordered coverage interval")
    first = first_day + timedelta(days=(-first_day.weekday()) % 7)
    last = last_day - timedelta(days=(last_day.weekday() + 1) % 7 + 6)
    selection_end = first + timedelta(weeks=7)
    if last <= selection_end:
        raise ValueError("need at least nine complete weeks")
    data = home / "data"
    data.mkdir(parents=True, exist_ok=True)
    inputs = [{"name": p.name, "sha256": checksum(p)} for p in files]
    identity = json.dumps(
        [PIPELINE, first.isoformat(), last.isoformat(), inputs, provenance or {}], sort_keys=True
    )
    key = hashlib.sha256(identity.encode()).hexdigest()
    with closing(sqlite3.connect(data / "ingestion.sqlite3", timeout=300)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, manifest TEXT NOT NULL)"
        )
        db.execute("BEGIN IMMEDIATE")
        cached = db.execute("SELECT manifest FROM batches WHERE id=?", (key,)).fetchone()
        if cached:
            manifest = json.loads(cached[0])
            curated = home / manifest["path"]
            quarantine = home / manifest["quarantine"]
            if (
                curated.is_file()
                and checksum(curated) == manifest["sha256"]
                and quarantine.is_file()
                and checksum(quarantine) == manifest["quarantine_sha256"]
            ):
                atomic_json(data / "current.json", manifest)
                return dict(manifest)
        started = time.perf_counter()
        process = psutil.Process()
        peak = [process.memory_info().rss]
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(0.05):
                peak[0] = max(peak[0], process.memory_info().rss)

        meter = threading.Thread(target=sample, daemon=True)
        meter.start()
        try:
            with tempfile.TemporaryDirectory(dir=data) as tmp:
                workspace = Path(tmp)
                with duckdb.connect(
                    str(workspace / "working.duckdb"),
                    config={"memory_limit": "512MB", "threads": "2"},
                ) as con:
                    report = _build(con, files, first, last, selection_end, workspace)
                digest = checksum(workspace / "weekly.parquet")
                weekly = data / f"weekly-{digest}.parquet"
                quarantine = data / f"quarantine-{key}.parquet"
                os.replace(workspace / "weekly.parquet", weekly)
                os.replace(workspace / "quarantine.parquet", quarantine)
        finally:
            stop.set()
            meter.join()
        manifest = {
            **report,
            "path": weekly.relative_to(home).as_posix(),
            "sha256": digest,
            "first_week": first.isoformat(),
            "last_week": last.isoformat(),
            "product_selection_end": selection_end.isoformat(),
            "inputs": inputs,
            "pipeline": PIPELINE,
            "provenance": provenance or {"mode": "local"},
            "quarantine": quarantine.relative_to(home).as_posix(),
            "quarantine_sha256": checksum(quarantine),
            "batch_id": key,
            "curation_seconds": time.perf_counter() - started,
            "sampled_peak_rss_bytes": peak[0],
        }
        db.execute("INSERT OR REPLACE INTO batches VALUES (?,?)", (key, json.dumps(manifest)))
        atomic_json(data / "current.json", manifest)
        return manifest
