"""Allowlist and verify locally trained model bytes before parsing native formats."""

import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED = {"model.json", "model.txt"}


def native_files(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("artifact directory is missing or linked")
    files = sorted(directory.iterdir())
    if {p.name for p in files} not in ({"model.json"}, ALLOWED):
        raise ValueError("unexpected or missing native artifact files")
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("linked or irregular native artifact")
    return files


def checksums(directory: Path) -> dict[str, str]:
    result = {}
    for path in native_files(directory):
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def checked_directory(home: Path, relative: str) -> Path:
    path = home.resolve() / relative
    if path.resolve().parent != (home.resolve() / "models") or path.is_symlink():
        raise ValueError("artifact path escapes trusted model directory")
    return path


def verify(directory: Path, expected: dict[str, str]) -> dict[str, Any]:
    from spiritops.features import SCHEMA_VERSION

    if checksums(directory) != expected:
        raise ValueError("native artifact checksum mismatch")
    meta: dict[str, Any] = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    kind = meta.get("kind")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("incompatible feature schema")
    if kind not in {"baseline", "lightgbm"} or ("model.txt" in expected) != (kind == "lightgbm"):
        raise ValueError("unsupported or incomplete native model")
    return meta
