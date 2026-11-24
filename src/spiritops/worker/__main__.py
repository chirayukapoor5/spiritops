import logging
import os
import time
from pathlib import Path

from spiritops.deployment import reconcile
from spiritops.monitoring import tick
from spiritops.worker import recover_tracking, run_once


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    home = Path(os.environ.get("SPIRITOPS_HOME", ".spiritops")).resolve()
    next_tick = 0.0
    while True:
        if time.monotonic() >= next_tick:
            try:
                recover_tracking(home)
                reconcile(home)
                tick(home)
            except Exception:
                logging.exception("monitoring cycle failed")
            next_tick = time.monotonic() + float(os.environ.get("SPIRITOPS_MONITOR_SECONDS", "30"))
        if not run_once(home):
            time.sleep(1)


if __name__ == "__main__":
    main()
