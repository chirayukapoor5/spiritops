"""Global native LightGBM and seasonal naive with an explicitly versioned schema."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import psutil

from spiritops.features import FEATURE_NAMES, SCHEMA_VERSION, WEEK, monday, panel, schema_features
from spiritops.ingestion.sources import atomic_json, checksum


@dataclass
class Predictor:
    metadata: dict[str, Any]
    booster: lgb.Booster | None

    def predict(self, rows: list[dict[str, float]]) -> list[float]:
        if not rows:
            return []
        names = self.metadata["feature_names"]
        if any(not row.keys() >= set(names) for row in rows):
            raise ValueError("incompatible input features")
        x = np.array([[row[f] for f in names] for row in rows], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("non-finite input")
        if self.booster is None:
            y = x[:, names.index("seasonal_naive")]
        else:
            y = np.asarray(self.booster.predict(x, num_threads=1))
        result = np.maximum(np.asarray(y), 0)
        if not np.isfinite(result).all() or result.shape != (len(rows),):
            raise ValueError("invalid forecast")
        return [float(value) for value in result]


def prediction_bounds(
    metadata: dict[str, Any], row: dict[str, float], estimate: float
) -> tuple[float | None, float | None]:
    quantiles = metadata.get("residual_quantiles")
    if quantiles is None:
        return None, None
    scale = max(1.0, row["mean13"]) if quantiles.get("scale") == "mean13" else 1.0
    return (
        max(0.0, estimate + scale * quantiles["lower"]),
        estimate + scale * quantiles["upper"],
    )


def fit(
    home: Path, out: Path, validation_start: str, validation_end: str, kind: str = "lightgbm"
) -> dict[str, Any]:
    started = time.perf_counter()
    if kind not in {"baseline", "lightgbm"}:
        raise ValueError("unknown model kind")
    p = panel(home)
    start, end = monday(validation_start), monday(validation_end)
    train_start, train_end = p.selection_end + WEEK, start - WEEK
    if train_start > train_end or not start <= end <= p.last:
        raise ValueError("validation must follow initial selection and training")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("training requires an empty output directory")
    rows, y, weeks = p.examples(train_start, train_end)
    x = np.array([[row[f] for f in FEATURE_NAMES] for row in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite training observations")
    parameters: dict[str, Any] = {}
    model_sha: str | None = None
    if kind == "lightgbm":
        parameters = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "tweedie",
            "learning_rate": 0.06,
            "num_leaves": 15,
            "min_data_in_leaf": 18,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": 31,
            "verbosity": -1,
        }
        booster = lgb.train(
            parameters,
            lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES)),
            num_boost_round=65,
        )
        booster.save_model(str(out / "model.txt"))
        model_sha = checksum(out / "model.txt")
    # Calibrate on the final eight development weeks with a frozen earlier fit.
    # Scale errors by recent purchase volume to avoid a common width for all products.
    residual_quantiles = None
    cutoff = max(train_start + WEEK, train_end - 7 * WEEK).isoformat()
    cal = np.asarray([w >= cutoff for w in weeks])
    if cal.any() and (~cal).any():
        center = x[cal, FEATURE_NAMES.index("seasonal_naive")]
        if kind == "lightgbm":
            prior = lgb.train(parameters, lgb.Dataset(x[~cal], label=y[~cal]), num_boost_round=65)
            center = np.maximum(np.asarray(prior.predict(x[cal], num_threads=1)), 0)
        scale = np.maximum(x[cal, FEATURE_NAMES.index("mean13")], 1.0)
        delta = (y[cal] - center) / scale
        lower, upper = np.quantile(delta, [0.1, 0.9])
        residual_quantiles = {
            "lower": min(0.0, float(lower)),
            "upper": max(0.0, float(upper)),
            "first_week": cutoff,
            "samples": int(cal.sum()),
            "scale": "mean13",
            "nominal_coverage": 0.8,
        }
    profile = {
        f: {"mean": float(x[:, i].mean()), "std": float(x[:, i].std())}
        for i, f in enumerate(FEATURE_NAMES)
    }
    # Critical categories are fixed using the fitted period only.
    category_volumes: dict[str, float] = {}
    for row, target in zip(rows, y, strict=True):
        group = str(int(row["category"]))
        category_volumes[group] = category_volumes.get(group, 0) + float(target)
    critical = sorted(category_volumes, key=lambda k: (-category_volumes[k], k))[:3]
    metadata = {
        "kind": kind,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "dataset_sha256": p.sha256,
        "training_start": train_start.isoformat(),
        "training_end": train_end.isoformat(),
        "validation_start": validation_start,
        "validation_end": validation_end,
        "residual_quantiles": residual_quantiles,
        "training_rows": len(y),
        "parameters": parameters,
        "boost_rounds": 65 if kind == "lightgbm" else 0,
        "profile": profile,
        "critical_categories": critical,
        "model_sha256": model_sha,
        "duration_seconds": time.perf_counter() - started,
        "rss_bytes": psutil.Process().memory_info().rss,
    }
    atomic_json(out / "model.json", metadata)
    return metadata


def load(out: Path) -> Predictor:
    metadata = json.loads((out / "model.json").read_text())
    names = list(schema_features(metadata["schema_version"]))
    if metadata.get("feature_names") != names:
        raise ValueError("incompatible artifact feature schema")
    kind = metadata.get("kind")
    if kind not in {"baseline", "lightgbm"}:
        raise ValueError("unknown model artifact")
    booster = None
    if kind == "lightgbm":
        model = out / "model.txt"
        if model.is_symlink() or checksum(model) != metadata.get("model_sha256"):
            raise ValueError("native model integrity check failed")
        booster = lgb.Booster(model_file=str(model))
        if booster.feature_name() != names:
            raise ValueError("native model feature schema mismatch")
    elif metadata.get("model_sha256") is not None:
        raise ValueError("unexpected baseline native artifact")
    return Predictor(metadata, booster)
