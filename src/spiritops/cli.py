from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from spiritops.config import home_dir, snapshot


def fixture(home: Path) -> dict:
    from spiritops.ingestion import ingest_files

    path = Path("tests/fixtures/iowa.csv.gz")
    if not path.is_file():
        raise FileNotFoundError(
            "Run fixture mode from the repository root containing tests/fixtures/iowa.csv.gz"
        )
    return ingest_files(home, [path], date(2023, 1, 1), date(2023, 12, 31))


def train(home: Path, kind: str, start: str | None = None, end: str | None = None) -> dict:
    from spiritops import worker

    m = snapshot(home)
    params = {"kind": kind}
    if start and end:
        params.update(validation_start=start, validation_end=end)
    key = f"cli:{m['sha256']}:{kind}:{start}:{end}"
    identifier = worker.enqueue(home, key, params)
    return finish(home, identifier)


def finish(home: Path, identifier: str) -> dict:
    from spiritops import worker

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        record = worker.job(home, identifier)
        if record is None:
            raise RuntimeError("The worker lost a persistent job")
        if record["status"] == "succeeded":
            return record
        if record["status"] == "failed":
            raise RuntimeError(f"Training job failed: {record}")
        if not worker.run_once(home):
            time.sleep(0.3)
    raise TimeoutError(f"Timed out awaiting job {identifier}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="spiritops")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "demo", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("--mode", default="demo", choices=("demo", "full", "fixture"))
        if name != "data":
            sub.add_argument("--report", type=Path)
    commands.add_parser("fixture")
    commands.add_parser("monitor")
    sub = commands.add_parser("train")
    sub.add_argument("--kind", default="lightgbm", choices=("lightgbm", "baseline"))
    sub.add_argument("--validation-start")
    sub.add_argument("--validation-end")
    args = parser.parse_args()
    root = home_dir()
    result: Any
    if args.command == "fixture" or args.command == "data" and args.mode == "fixture":
        result = fixture(root)
    elif args.command == "data":
        from spiritops.ingestion import ingest

        result = ingest(root, args.mode)
    elif args.command == "train":
        if bool(args.validation_start) != bool(args.validation_end):
            parser.error("Provide both validation boundaries")
        result = train(root, args.kind, args.validation_start, args.validation_end)
    elif args.command == "monitor":
        from spiritops.monitoring import tick

        result = tick(root)
    else:
        from spiritops.demonstration import demonstrate

        result = demonstrate(root, args.mode, benchmark=args.command == "benchmark")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
