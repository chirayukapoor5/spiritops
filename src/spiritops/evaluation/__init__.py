"""Release gates on the same independent rolling-origin target weeks."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from spiritops.features import monday, panel
from spiritops.training import load


def metrics(
    actual: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    seasonal: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    y, yhat, naive = (np.asarray(v, dtype=float) for v in (actual, prediction, seasonal))
    if y.ndim != 1 or not len(y) or y.shape != yhat.shape or y.shape != naive.shape:
        raise ValueError("metrics require paired observations")
    if not all(np.isfinite(v).all() and (v >= 0).all() for v in (y, yhat, naive)):
        raise ValueError("invalid volume or prediction")
    err = yhat - y
    denominator = float(np.abs(y).sum())
    absolute = float(np.abs(err).sum())
    return {
        "wape": absolute / denominator if denominator else None,
        "mae": float(np.abs(err).mean()),
        "bias": float(err.sum()) / denominator if denominator else None,
        "seasonal_scaled_error": absolute / float(np.maximum(np.maximum(y, naive), 1).sum()),
    }


def _threshold(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid release threshold {name}")
    return value


def evaluate(
    home: Path, candidate: Path, incumbent: Path | None, start: str, end: str
) -> dict[str, Any]:
    new = load(candidate)
    old = load(incumbent) if incumbent is not None else None
    first, last = monday(start), monday(end)
    for model in (new, old):
        if model is not None and monday(model.metadata["training_end"]) >= first:
            raise ValueError("evaluation overlaps fitted observations")
    p = panel(home)
    if new.metadata["dataset_sha256"] != p.sha256:
        raise ValueError("candidate trained on another snapshot")
    rows, y, weeks = p.examples(first, last)
    naive = np.array([r["seasonal_naive"] for r in rows])
    y_new = np.asarray(new.predict(rows))
    y_old = np.asarray(old.predict(rows)) if old else naive
    candidate_scores, incumbent_scores = metrics(y, y_new, naive), metrics(y, y_old, naive)
    gain = _threshold("SPIRITOPS_MIN_WAPE_GAIN", 0.02)
    tolerance = _threshold("SPIRITOPS_GROUP_WAPE_TOLERANCE", 0.15)
    budget = _threshold("SPIRITOPS_MODEL_LATENCY_MS", 200)
    runs = []
    for row in rows[: min(80, len(rows))]:
        before = time.perf_counter()
        new.predict([row])
        runs.append(1000 * (time.perf_counter() - before))
    latency = float(np.percentile(runs, 95))
    cw, iw = candidate_scores["wape"], incumbent_scores["wape"]
    bootstrap = old is None and new.metadata["kind"] == "baseline"
    tests = {
        "native_integrity": True,
        "schema_compatible": True,
        "finite_predictions": True,
        "independent_window": True,
        "defined_wape": cw is not None and iw is not None,
        "latency": latency <= budget,
        "relative_wape_gain": bootstrap
        or (cw is not None and iw is not None and iw > 0 and cw <= (1 - gain) * iw),
    }
    categories = np.array([str(int(row["category"])) for row in rows])
    segments = {}
    critical = (old or new).metadata["critical_categories"]
    for group in sorted(set(categories)):
        mask = categories == group
        sc, si = (
            metrics(y[mask], y_new[mask], naive[mask]),
            metrics(y[mask], y_old[mask], naive[mask]),
        )
        a, b = sc["wape"], si["wape"]
        segments[str(group)] = {"candidate": sc, "incumbent": si, "rows": int(mask.sum())}
        if group in critical:
            tests[f"category_{group}"] = bootstrap or (
                a is not None and b is not None and a <= b + tolerance
            )
    for group in set(critical) - set(categories):
        tests[f"category_{group}"] = bootstrap
    all_weeks = np.asarray(weeks)
    weekly = {}
    for week in sorted(set(weeks)):
        mask = all_weeks == week
        weekly[week] = {
            "candidate": metrics(y[mask], y_new[mask], naive[mask]),
            "incumbent": metrics(y[mask], y_old[mask], naive[mask]),
        }
    reasons = [name for name, passed in tests.items() if not passed]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "tests": tests,
        "candidate": candidate_scores,
        "incumbent": incumbent_scores,
        "latency_ms": latency,
        "validation_start": start,
        "validation_end": end,
        "rows": len(y),
        "weeks": weekly,
        "segments": segments,
        "dataset_sha256": p.sha256,
        "baseline_bootstrap": bootstrap,
        "policy": {
            "min_relative_wape_gain": gain,
            "group_wape_tolerance": tolerance,
            "latency_budget_ms": budget,
            "critical_categories": critical,
        },
    }
