"""Compare frozen forecast specifications on cached, verified weekly panels."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np

from spiritops.evaluation import evaluate, metrics
from spiritops.features import LEGACY_FEATURE_NAMES, WEEK, monday, panel
from spiritops.training import fit


def previous_accuracy(home: Path, start: str, end: str) -> dict:
    history = panel(home)
    first, last = monday(start), monday(end)
    train, y_train, _ = history.examples(history.selection_end + WEEK, first - WEEK)
    validation, y, _ = history.examples(first, last)
    x, xv = (
        np.asarray([[r[n] for n in LEGACY_FEATURE_NAMES] for r in rows])
        for rows in (train, validation)
    )
    prior = lgb.train(
        {
            "objective": "regression_l1",
            "learning_rate": 0.06,
            "num_leaves": 15,
            "min_data_in_leaf": 18,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": 31,
            "verbosity": -1,
        },
        lgb.Dataset(x, label=y_train, feature_name=list(LEGACY_FEATURE_NAMES)),
        num_boost_round=65,
    )
    prediction = np.maximum(prior.predict(xv, num_threads=1), 0)
    naive = [r["seasonal_naive"] for r in validation]
    positive = y > 0
    return {
        "metrics": metrics(y, prediction, naive),
        "rows": len(y),
        "training_zero_rate": float(np.mean(y_train == 0)),
        "validation_zero_rate": float(np.mean(~positive)),
        "predicted_litres": float(prediction.sum()),
        "actual_litres": float(y.sum()),
        "underforecast_litres_on_nonzero_weeks": float(
            np.maximum(y[positive] - prediction[positive], 0).sum()
        ),
    }


def run(home: Path, demo: Path) -> dict:
    p = panel(home)
    results = []
    for name, offset, length in [
        ("development", -31, 8),
        ("development", -23, 8),
        ("development", -15, 8),
        ("holdout", -7, 6),
    ]:
        start = (p.last + offset * WEEK).isoformat()
        end = (p.last + (offset + length - 1) * WEEK).isoformat()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            fit(home, path, start, end)
            report = evaluate(home, path, None, start, end)
        results.append(
            {
                "split": name,
                "start": start,
                "end": end,
                "seasonal_naive": report["incumbent"],
                "tweedie_lightgbm": report["candidate"],
                "prior_lightgbm": previous_accuracy(home, start, end),
                "intervals": report["intervals"],
                "gate_results": report["tests"],
                "critical_segments": {
                    group: report["segments"].get(group)
                    for group in report["policy"]["critical_categories"]
                },
                "policy": report["policy"],
            }
        )
    return {
        "dataset_sha256": p.sha256,
        "windows": results,
        "previous_two_store_release": previous_accuracy(demo, "2024-10-28", "2024-12-16"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statewide", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(run(args.statewide, args.demo), indent=2) + "\n", encoding="utf-8"
    )
