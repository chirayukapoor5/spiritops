"""Exercise Compose over HTTP and check serving-version persistence across restarts."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def fetch(url: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def smoke(base: str, dashboard: str, previous: dict[str, Any] | None) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    while True:
        try:
            fetch(base + "/health/ready")
            if fetch(base + "/v1/operations")["worker"]["available"]:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Serving API and worker did not become ready")
        time.sleep(1)
    with urlopen(dashboard + "/_stcore/health", timeout=10) as response:
        if response.read().strip() != b"ok":
            raise RuntimeError("Dashboard health check failed")
    products = fetch(base + "/v1/products")
    item = {"product": products["products"][0], "week": products["next_week"]}
    active = str(fetch(base + "/v1/models/current")["version"])
    prediction = fetch(base + "/v1/predict", item)
    if (
        str(prediction["version"]) != active
        or prediction["week"] != item["week"]
        or not prediction["prediction_id"]
        or not math.isfinite(prediction["prediction"])
        or prediction["prediction"] < 0
    ):
        raise RuntimeError(f"Invalid prediction: {prediction!r}")
    if previous and (
        active != previous["version"]
        or item != previous["input"]
        or prediction["prediction"] != previous["prediction"]
    ):
        raise RuntimeError("Serving version or prediction changed across restart")
    return {"version": active, "input": item, "prediction": prediction["prediction"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8501")
    parser.add_argument("--expect", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    previous = json.loads(args.expect.read_text()) if args.expect else None
    evidence = json.dumps(smoke(args.base, args.dashboard, previous), indent=2)
    if args.output:
        args.output.write_text(evidence + "\n", encoding="utf-8")
    print(evidence)


if __name__ == "__main__":
    main()
