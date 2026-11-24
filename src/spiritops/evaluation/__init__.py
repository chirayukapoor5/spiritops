"""Release gates on the same independent rolling-origin target weeks."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from spiritops.config import snapshot
from spiritops.features import monday, panel
from spiritops.training import load, prediction_bounds


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


def quality_limits(home: Path) -> dict[str, float | str]:
    mode = snapshot(home).get("provenance", {}).get("mode")
    sandbox = mode in {"demo", "fixture"}
    return {
        "profile": "sparse_showcase" if sandbox else "statewide",
        "max_wape": _threshold("SPIRITOPS_MAX_WAPE", 1.80 if sandbox else 0.40),
        "max_abs_bias": _threshold("SPIRITOPS_MAX_ABS_BIAS", 0.65 if sandbox else 0.15),
        "max_segment_wape": _threshold("SPIRITOPS_MAX_SEGMENT_WAPE", 2.40 if sandbox else 0.50),
        "max_segment_abs_bias": _threshold(
            "SPIRITOPS_MAX_SEGMENT_ABS_BIAS", 0.80 if sandbox else 0.25
        ),
        "group_wape_tolerance": _threshold("SPIRITOPS_GROUP_WAPE_TOLERANCE", 0.10),
    }


def meets_limit(value: float | None, limit: float) -> bool:
    return value is not None and math.isfinite(value) and abs(value) <= limit


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
    limits = quality_limits(home)
    tolerance = float(limits["group_wape_tolerance"])
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
        "fresh_release_window": old is None or first > monday(old.metadata["validation_end"]),
        "defined_wape": cw is not None and iw is not None,
        "minimum_validation_sample": len(set(weeks)) >= 6 and len(y) >= 100,
        "absolute_wape": meets_limit(cw, float(limits["max_wape"])),
        "absolute_bias": meets_limit(candidate_scores["bias"], float(limits["max_abs_bias"])),
        "latency": latency <= budget,
        "relative_wape_gain": bootstrap
        or (cw is not None and iw is not None and iw > 0 and cw <= (1 - gain) * iw),
    }
    categories = np.array([str(int(row["category"])) for row in rows])
    segments = {}
    critical = set(new.metadata["critical_categories"])
    if old is not None:
        critical.update(old.metadata["critical_categories"])
    for group in sorted(set(categories)):
        mask = categories == group
        sc, si = (
            metrics(y[mask], y_new[mask], naive[mask]),
            metrics(y[mask], y_old[mask], naive[mask]),
        )
        a, b = sc["wape"], si["wape"]
        count = int(mask.sum())
        segments[str(group)] = {
            "candidate": sc,
            "incumbent": si,
            "rows": count,
            "actual_litres": float(y[mask].sum()),
        }
        if group in critical:
            tests[f"category_{group}_sample"] = count >= 6 and float(y[mask].sum()) >= 1.0
            tests[f"category_{group}_wape"] = meets_limit(a, float(limits["max_segment_wape"]))
            tests[f"category_{group}_bias"] = meets_limit(
                sc["bias"], float(limits["max_segment_abs_bias"])
            )
            tests[f"category_{group}_relative"] = bootstrap or (
                a is not None and b is not None and a <= b + tolerance
            )
    for group in set(critical) - set(categories):
        tests[f"category_{group}_sample"] = False
    intervals = [
        (lo, hi, target)
        for row, prediction, target in zip(rows, y_new, y, strict=True)
        for lo, hi in [prediction_bounds(new.metadata, row, float(prediction))]
        if lo is not None and hi is not None
    ]
    interval_metrics = None
    if intervals:
        interval_metrics = {
            "samples": len(intervals),
            "coverage": float(np.mean([lo <= target <= hi for lo, hi, target in intervals])),
            "mean_width_litres": float(np.mean([hi - lo for lo, hi, _ in intervals])),
        }
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
        "intervals": interval_metrics,
        "weeks": weekly,
        "segments": segments,
        "dataset_sha256": p.sha256,
        "baseline_bootstrap": bootstrap,
        "policy": {
            "min_relative_wape_gain": gain,
            **limits,
            "latency_budget_ms": budget,
            "critical_categories": sorted(critical),
        },
    }
