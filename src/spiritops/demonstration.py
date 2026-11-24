"""Run an isolated historical replay over real HTTP and a persistent model registry."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
import psutil

from spiritops import deployment, monitoring
from spiritops.cli import fixture, finish, train
from spiritops.config import shift, snapshot
from spiritops.ingestion import ingest


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def live_api(home: Path) -> Iterator[httpx.Client]:
    env = dict(os.environ, SPIRITOPS_HOME=str(home), SPIRITOPS_ADMIN_TOKEN=uuid.uuid4().hex)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (home / "api.log").open("ab") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "spiritops.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=30,
                trust_env=False,
                headers={"X-Admin-Token": env["SPIRITOPS_ADMIN_TOKEN"]},
            ) as client:
                for _ in range(300):
                    try:
                        if client.get("/health/ready").is_success:
                            break
                    except httpx.TransportError:
                        pass
                    if proc.poll() is not None:
                        raise RuntimeError(f"API exited: {(home / 'api.log').read_text()[-2000:]}")
                    time.sleep(0.2)
                else:
                    raise TimeoutError("API readiness failed")
                yield client
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def load_requests(client: httpx.Client, target: dict, count: int, workers: int) -> dict:
    def request(_: int) -> tuple[float, bool, str]:
        start = time.perf_counter()
        r = client.post("/v1/predict", json=target)
        elapsed = (time.perf_counter() - start) * 1000
        value = r.json()
        ok = r.is_success and np.isfinite(value["prediction"]) and value["prediction"] >= 0
        return elapsed, bool(ok), str(value.get("version", ""))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(request, range(count)))
    return {
        "requests": count,
        "concurrency": workers,
        "errors": sum(not r[1] for r in results),
        "p95_ms": float(np.percentile([r[0] for r in results], 95)),
        "requests_per_second": count / (time.perf_counter() - start),
        "versions": sorted({r[2] for r in results}),
    }


def _measure_job(
    client: httpx.Client, home: Path, identifier: str, target: dict
) -> tuple[dict, dict]:
    samples: list[tuple[float, bool, str]] = []
    done = threading.Event()

    def probe() -> None:
        while not done.is_set():
            start = time.perf_counter()
            try:
                r = client.post("/v1/predict", json=target)
                value = r.json()
                valid = (
                    r.is_success and np.isfinite(value["prediction"]) and value["prediction"] >= 0
                )
                version = str(value.get("version", ""))
            except (httpx.HTTPError, ValueError, KeyError):
                valid, version = False, ""
            samples.append(((time.perf_counter() - start) * 1000, bool(valid), version))
            done.wait(0.1)

    thread = threading.Thread(target=probe)
    thread.start()
    try:
        result = finish(home, identifier)
    finally:
        done.set()
        thread.join()
    return result, {
        "requests": len(samples),
        "errors": sum(not s[1] for s in samples),
        "p95_ms": float(np.percentile([s[0] for s in samples], 95)),
        "versions": sorted({s[2] for s in samples}),
    }


def demonstrate(root: Path, mode: str, benchmark: bool = False, source: Path | None = None) -> dict:
    parent = root / "demonstrations"
    parent.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="replay-", dir=parent)).resolve()
    # Avoid a demonstration alias changing an existing tracking server's champion.
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (home / "mlflow.sqlite3").as_posix()
    try:
        return _replay(home, mode, benchmark, source)
    finally:
        if tracking is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = tracking


def reuse_statewide(home: Path, source: Path) -> None:
    source = source.resolve()
    manifest = snapshot(source)
    if manifest.get("provenance", {}).get("mode") != "full":
        raise ValueError("Replay requires a statewide snapshot")
    for key in ("path", "quarantine"):
        relative = Path(manifest[key])
        original = (source / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not original.is_relative_to(source / "data")
        ):
            raise ValueError("Invalid snapshot path")
        target = home / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    write_json(home / "data/current.json", manifest)


def _replay(home: Path, mode: str, benchmark: bool, source: Path | None) -> dict:
    report: dict = {
        "mode": mode,
        "scenario": "Historical as-of replay",
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
    }
    start = time.perf_counter()
    peak = 0
    stop = threading.Event()

    def memory() -> None:
        nonlocal peak
        process = psutil.Process()
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            stop.wait(0.025)

    sampler = threading.Thread(target=memory)
    sampler.start()
    try:
        if source is not None:
            reuse_statewide(home, source)
        elif mode == "fixture":
            fixture(home)
        else:
            ingest(home, mode)
    finally:
        stop.set()
        sampler.join()
    m = snapshot(home)
    statewide = mode == "full"
    report.update(
        snapshot_reused=source is not None,
        ingestion_seconds=time.perf_counter() - start if source is None else None,
        ingestion_sampled_peak_rss_bytes=peak if source is None else None,
        data_sha256=m["sha256"],
        products=len(m["products"]),
        first_week=m["first_week"],
        last_week=m["last_week"],
    )
    last = m["last_week"]
    boot_end = shift(last, -18)
    initial = train(home, "baseline", shift(boot_end, -5), boot_end)
    report["bootstrap"] = initial["result"]
    clock = home / "replay-clock.json"
    replay_end = shift(last, -8) if statewide else last
    if statewide:
        write_json(home / "policy.json", {"excess_error": -0.99})
        report["controlled_policy_change"] = (
            "99% lower alert threshold for historical demonstration"
        )
    with live_api(home) as client:
        outcomes = []
        for offset in range(-17, -8):
            week = shift(last, offset)
            write_json(clock, {"as_of": week})
            targets = [{"product": p, "week": week} for p in m["products"][:100]]
            response = client.post("/v1/predict/batch", json={"items": targets})
            response.raise_for_status()
            write_json(clock, {"as_of": shift(week)})
            decision = monitoring.tick(home)
            outcomes.append({k: decision.get(k) for k in ("action", "reason", "event", "job_id")})
            if "Reserved independent" in decision["reason"]:
                break
        if not any("Reserved independent" in str(d["reason"]) for d in outcomes):
            # A controlled alert sensitivity change leaves measured forecasts and labels intact.
            write_json(home / "policy.json", {"excess_error": -0.99})
            decision = monitoring.tick(home)
            outcomes.append({k: decision.get(k) for k in ("action", "reason", "event", "job_id")})
            report["controlled_policy_change"] = (
                "99% lower alert threshold for historical demonstration"
            )
        write_json(clock, {"as_of": shift(replay_end)})
        decision = monitoring.tick(home)
        outcomes.append({k: decision.get(k) for k in ("action", "reason", "event", "job_id")})
        report["monitoring"] = outcomes
        report["performance_before_retraining"] = decision.get("performance")
        if decision["action"] != "RETRAIN":
            raise RuntimeError(f"Retraining did not enqueue: {outcomes}")
        target = {"product": m["products"][0], "week": shift(replay_end)}
        result, load = _measure_job(client, home, decision["job_id"], target)
        report["automatic_job"] = result["result"]
        if statewide and result["result"]["outcome"] != "PROMOTE":
            raise RuntimeError(f"Statewide development release did not qualify: {result['result']}")
        report["requests_during_training_and_activation"] = load
        report["warm_serial"] = load_requests(client, target, 100 if benchmark else 12, 1)
        report["concurrent"] = load_requests(client, target, 160 if benchmark else 24, 8)
        state = deployment.current(home)
        from spiritops.monitoring.shadow import collect, summary

        if statewide:
            # The confirmatory holiday window follows both the fit and the previous release.
            rejection_start, rejection_end = shift(last, -7), shift(last, -2)
            write_json(clock, {"as_of": shift(rejection_end)})
            weaker = train(home, "lightgbm", rejection_start, rejection_end)
            report["holdout_challenger"] = weaker["result"]
            if weaker["result"]["outcome"] != "REJECT_CANDIDATE":
                raise RuntimeError("Expected the holiday holdout to reject the challenger")
            for week in (shift(last, -1), last):
                write_json(clock, {"as_of": week})
                batch = [{"product": p, "week": week} for p in m["products"]]
                client.post("/v1/predict/batch", json={"items": batch}).raise_for_status()
                collect(home)
                write_json(clock, {"as_of": shift(week)})
                monitoring.mature(home, m)
            target["week"] = shift(last)
        else:
            weaker = train(home, "baseline", state["validation_start"], state["validation_end"])
            report["baseline_challenger"] = weaker["result"]
        if str(state["version"]) != str(initial["result"]["version"]):
            with contextlib.closing(sqlite3.connect(home / "control.sqlite3")) as conn:
                directory = conn.execute(
                    "SELECT directory FROM models WHERE version=?", (state["version"],)
                ).fetchone()[0]
            artifact = home / directory / "model.json"
            original = artifact.with_suffix(".fault-injection")
            artifact.rename(original)
            try:
                start = time.perf_counter()
                restored = client.post("/v1/predict", json=target)
                restored.raise_for_status()
                report["rollback_seconds"] = time.perf_counter() - start
                report["rollback"] = {
                    "injected": "active model.json temporarily removed",
                    "from": state["version"],
                    "restored_version": restored.json()["version"],
                }
            finally:
                original.rename(artifact)
        if not statewide:
            # The fixture has no outcome for its next reporting week.
            client.post(
                "/v1/predict", json={**target, "product": m["products"][-1]}
            ).raise_for_status()
            collect(home)
        report["shadow"] = summary(home)
        if statewide and not any(s["matured"] == 2 * len(m["products"]) for s in report["shadow"]):
            raise RuntimeError("The statewide replay did not mature its prospective forecasts")
        report["history"] = deployment.history(home)
        report["operations"] = client.get("/v1/operations").raise_for_status().json()
    with live_api(home) as restarted:
        report["after_restart"] = (
            restarted.post("/v1/predict", json=target).raise_for_status().json()
        )
    write_json(home / "report.json", report)
    write_json(home.parent / "latest.json", {"directory": home.name})
    return report
