"""Measure durable HTTP serving against an isolated copy of a cached statewide replay."""

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import numpy as np
import uvicorn
from check_monitoring import clone_state

from spiritops import deployment, features
from spiritops.api import create_app
from spiritops.config import shift, snapshot
from spiritops.demonstration import load_requests, write_json
from spiritops.monitoring import observations


def serve(port: int) -> None:
    stages: dict[str, list[float]] = {}

    def timed(fn, samples):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                samples.append(1000 * (time.perf_counter() - start))

        return wrapper

    for module, name in (
        (features, "panel"),
        (deployment, "predict_with_metadata"),
        (observations, "save_batch"),
        (observations, "traffic"),
    ):
        stages[name] = []
        setattr(module, name, timed(getattr(module, name), stages[name]))
    app = create_app()

    @app.get("/profile")
    def profile():
        return stages

    @app.post("/profile/clear")
    def clear():
        for entries in stages.values():
            entries.clear()

    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False, log_level="warning")


def scenarios(api: httpx.Client) -> list[dict]:
    info = api.get("/v1/products").raise_for_status().json()
    target = {"product": info["products"][0], "week": info["next_week"]}
    api.post("/v1/predict", json=target).raise_for_status()

    def finish(result: dict) -> dict:
        marks = api.get("/profile").raise_for_status().json()
        result["stage_ms"] = {
            name: {
                "p50": float(np.percentile(values[-result["requests"] :], 50)),
                "p95": float(np.percentile(values[-result["requests"] :], 95)),
            }
            for name, values in marks.items()
            if values
        }
        return result

    trials = []
    for concurrency in (1, 8, 8, 8):
        api.post("/profile/clear").raise_for_status()
        trials.append(finish(load_requests(api, target, 160, concurrency)))

    def issue(product: str) -> tuple[float, bool]:
        start = time.perf_counter()
        response = api.post("/v1/predict", json={**target, "product": product})
        return 1000 * (time.perf_counter() - start), response.is_success

    api.post("/profile/clear").raise_for_status()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        fresh = list(pool.map(issue, info["products"][1:]))
    elapsed = time.perf_counter() - start
    trials.append(
        finish(
            {
                "workload": "distinct products",
                "concurrency": 8,
                "requests": len(fresh),
                "errors": sum(not ok for _, ok in fresh),
                "p95_ms": float(np.percentile([ms for ms, _ in fresh], 95)),
                "requests_per_second": len(fresh) / elapsed,
            }
        )
    )
    return trials


def run(cached_run: Path, state: Path) -> list[dict]:
    clone_state(cached_run, state)
    # Select the already approved LightGBM in the throwaway copy after the replay's rollback.
    with closing(sqlite3.connect(state / "control.sqlite3")) as conn:
        conn.execute("UPDATE manifest SET version='2'")
        conn.commit()
    write_json(state / "replay-clock.json", {"as_of": shift(snapshot(state)["last_week"])})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    child = subprocess.Popen(
        [sys.executable, __file__, "--serve", str(port)],
        env=dict(os.environ, SPIRITOPS_HOME=str(state)),
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=25, trust_env=False) as api:
            for _ in range(200):
                try:
                    if api.get("/health/ready").is_success:
                        return scenarios(api)
                except httpx.TransportError:
                    if child.poll() is not None:
                        raise RuntimeError("Profile server exited")
                time.sleep(0.15)
            raise TimeoutError("Profile server readiness failed")
    finally:
        child.terminate()
        child.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cached-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
    elif args.cached_run and args.output:
        with TemporaryDirectory() as directory:
            samples = run(args.cached_run, Path(directory))
            args.output.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    else:
        parser.error("Provide --cached-run and --output")
