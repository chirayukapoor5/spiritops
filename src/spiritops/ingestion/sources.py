"""Resolve DCAT distributions and stream immutable official exports."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

import duckdb

from spiritops.validation import REQUIRED

YEARS = (2023, 2024, 2025)
DEMO_STORES = {"2622", "4161"}
COLUMNS = sorted(REQUIRED)
FIELDS = ", ".join(COLUMNS)


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def resolve(year: int) -> dict[str, str]:
    if year not in YEARS:
        raise ValueError("unsupported Iowa release")
    catalog = f"https://catalog.data.gov/dataset/iowa-liquor-sales-{year}"
    request = Request(catalog, headers={"User-Agent": "SpiritOps Iowa data ingestion"})
    with urlopen(request, timeout=60) as response:
        html = response.read(2_000_000).decode()
    urls = set(re.findall(r"https://idh-be\.iowa\.gov/api/v1/datasets/\d+/rows\.csv", html))
    if len(urls) != 1:
        raise ValueError(f"could not resolve a unique official CSV from {catalog}")
    modified = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00)", html)
    return {
        "year": str(year),
        "catalog": catalog,
        "url": urls.pop(),
        "metadata_sha256": hashlib.sha256(html.encode()).hexdigest(),
        "retrieved": datetime.now(UTC).isoformat(),
        "catalog_timestamp": modified.group(1) if modified else "",
    }


def fetch(raw: Path, year: int) -> tuple[Path, dict[str, str]]:
    raw.mkdir(exist_ok=True, parents=True)
    info = resolve(year)
    record = raw / f"{year}.json"
    if record.exists():
        previous = json.loads(record.read_text())
        path = raw / previous["filename"]
        if (
            previous["url"] == info["url"]
            and previous["catalog_timestamp"]
            and previous["catalog_timestamp"] == info["catalog_timestamp"]
            and path.is_file()
            and checksum(path) == previous["sha256"]
        ):
            return path, previous
    with tempfile.NamedTemporaryFile(dir=raw, suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urlopen(info["url"], timeout=120) as response:
                shutil.copyfileobj(response, tmp, length=1 << 20)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        return register_archive(raw, tmp_path, info)
    finally:
        tmp_path.unlink(missing_ok=True)


def register_archive(raw: Path, path: Path, info: dict[str, str]) -> tuple[Path, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        if not csv_members(archive):
            raise ValueError("Iowa archive contains no CSV members")
    digest = checksum(path)
    name = f"{info['year']}-{digest}.zip"
    dest = raw / name
    os.replace(path, dest)
    info = {**info, "filename": name, "sha256": digest}
    atomic_json(raw / f"{info['year']}.json", info)
    return dest, info


def csv_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))


def project_archive(archive_path: Path, folder: Path) -> Iterator[Path]:
    """Project one member at a time. DuckDB spills to disk on full releases."""
    folder.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=folder) as tmp:
        con = duckdb.connect(config={"memory_limit": "512MB", "threads": "2"})
        try:
            for index, member in enumerate(csv_members(archive)):
                dest = folder / f"{index:04d}.parquet"
                if not dest.exists():
                    part = Path(tmp) / "member.csv"
                    with archive.open(member) as src, part.open("wb") as out:
                        shutil.copyfileobj(src, out, length=1 << 20)
                    projected = con.read_csv(str(part), all_varchar=True).project(FIELDS)
                    temp = Path(tmp) / "member.parquet"
                    projected.write_parquet(str(temp))
                    os.replace(temp, dest)
                yield dest
        finally:
            con.close()


def extract_demo(archives: list[Path], target: Path, stores: set[str]) -> int:
    """Rebuild the redistributable row subset using a fixed geographic predicate."""
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
    ):
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(COLUMNS)
            for path in archives:
                with zipfile.ZipFile(path) as archive:
                    for member in csv_members(archive):
                        with (
                            archive.open(member) as stream,
                            io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as lines,
                        ):
                            rows = csv.reader(lines)
                            header = next(rows)
                            store_idx = header.index("store_no")
                            positions = [header.index(key) for key in COLUMNS]
                            for row in rows:
                                if row[store_idx] in stores:
                                    writer.writerow([row[i] for i in positions])
                                    count += 1
    return count
