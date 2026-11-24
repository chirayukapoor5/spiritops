"""Replay the cached statewide baseline with the default controller thresholds."""

import argparse
import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from spiritops.config import shift, snapshot
from spiritops.demonstration import live_api, reuse_statewide, write_json
from spiritops.monitoring import policy, tick
from spiritops.worker import jobs


def clone_state(source: Path, state: Path) -> None:
    reuse_statewide(state, source)
    shutil.copytree(source / "models", state / "models")
    with (
        closing(sqlite3.connect(source / "control.sqlite3")) as original,
        closing(sqlite3.connect(state / "control.sqlite3")) as copied,
    ):
        original.backup(copied)


def replay(source: Path, state: Path) -> dict:
    clone_state(source, state)
    manifest = snapshot(state)
    settings = asdict(policy(state))
    decisions = []
    with live_api(state) as api:
        active = api.get("/v1/models/current").raise_for_status().json()
        if active["kind"] != "baseline":
            raise ValueError("Expected the restored statewide baseline")
        target = shift(active["validation_end"])
        while target <= manifest["last_week"]:
            write_json(state / "replay-clock.json", {"as_of": target})
            forecasts = [{"product": product, "week": target} for product in manifest["products"]]
            api.post("/v1/predict/batch", json={"items": forecasts}).raise_for_status()
            next_week = shift(target)
            write_json(state / "replay-clock.json", {"as_of": next_week})
            result = tick(state)
            decisions.append(
                {
                    "week": target,
                    "action": result["action"],
                    "reason": result["reason"],
                    "job_id": result["job_id"],
                    "latest_week_metrics": result["performance"]["weekly"][-1],
                }
            )
            target = next_week
    enqueued = [job for job in jobs(state) if job["status"] == "queued"]
    return {
        "scenario": "Historical replay with default thresholds and cached artifacts. Jobs remain queued.",
        "dataset_sha256": manifest["sha256"],
        "serving_version": active["version"],
        "policy": settings,
        "reference_seasonal_scaled_error": active["evaluation"]["candidate"][
            "seasonal_scaled_error"
        ],
        "decisions": decisions,
        "queued_jobs": enqueued,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        result = replay(args.cached_run, Path(directory))
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
