from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def home_dir() -> Path:
    return Path(os.getenv("SPIRITOPS_HOME", ".spiritops")).resolve()


def snapshot(home: Path) -> dict[str, Any]:
    try:
        return json.loads((home / "data" / "current.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("A validated data snapshot is required") from exc


def shift(week: str, weeks: int = 1) -> str:
    return (date.fromisoformat(week) + timedelta(weeks=weeks)).isoformat()
