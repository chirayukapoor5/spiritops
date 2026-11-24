"""Local Compose entry points. Each process shares the same durable volume."""

import os
import secrets
import subprocess
import sys
from pathlib import Path

from spiritops import deployment, worker


def bootstrap(home: Path) -> None:
    if not (home / "data" / "current.json").is_file():
        mode = os.environ.get("SPIRITOPS_MODE", "demo")
        if mode not in {"demo", "full", "fixture"}:
            raise ValueError("SPIRITOPS_MODE must be demo, full or fixture")
        command = ["fixture"] if mode == "fixture" else ["data", "--mode", mode]
        subprocess.run(["spiritops", *command], check=True)

    try:
        active = deployment.current(home)
    except RuntimeError:
        if any(event.get("outcome") == "PROMOTE" for event in deployment.history(home)):
            raise
        active = {}
    if not active.get("version"):
        subprocess.run(["spiritops", "train", "--kind", "baseline"], check=True)
    print(f"Bootstrapped serving version {deployment.current(home)['version']}", flush=True)


def api(home: Path) -> None:
    if not os.environ.get("SPIRITOPS_ADMIN_TOKEN"):
        secret = home / "admin-token"
        if not secret.exists():
            # Atomic publication also prevents reading a token truncated by a crash.
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(dir=home, delete=False, mode="w") as handle:
                handle.write(secrets.token_urlsafe(32))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, secret)
        token = secret.read_text().strip()
        if not token:
            raise RuntimeError("The stored administrator token is empty")
        os.environ["SPIRITOPS_ADMIN_TOKEN"] = token
    os.execvp("uvicorn", ["uvicorn", "spiritops.api:app", "--host", "0.0.0.0", "--port", "8000"])


def main() -> None:
    home = Path(os.environ["SPIRITOPS_HOME"])
    match sys.argv[1]:
        case "bootstrap":
            bootstrap(home)
        case "api":
            api(home)
        case "worker-health":
            raise SystemExit(0 if worker.status(home)["available"] else 1)
        case _:
            raise ValueError("Unknown container command")


if __name__ == "__main__":
    main()
