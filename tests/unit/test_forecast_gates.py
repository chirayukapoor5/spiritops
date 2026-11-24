from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spiritops import evaluation


@pytest.mark.parametrize(
    ("major", "protected", "failures"),
    [
        (55, 55, {"absolute_wape", "absolute_bias"}),
        (100, 35, {"category_200_wape", "category_200_bias"}),
    ],
)
def test_absolute_gates_reject_improvement_over_inadequate_incumbent(
    monkeypatch, tmp_path, major, protected, failures
):
    start = date(2025, 9, 8)
    weeks = [(start + timedelta(weeks=i)).isoformat() for i in range(6) for _ in range(20)]
    rows = [{"category": 200 if i % 20 < 2 else 100, "seasonal_naive": 20} for i in range(120)]
    metadata = {
        "kind": "lightgbm",
        "training_end": "2025-08-18",
        "validation_end": "2025-09-01",
        "dataset_sha256": "snapshot",
        "critical_categories": ["100", "200"],
    }
    monkeypatch.setattr(evaluation, "snapshot", lambda _: {"provenance": {"mode": "full"}})
    monkeypatch.setattr(
        evaluation,
        "panel",
        lambda _: SimpleNamespace(
            sha256="snapshot", examples=lambda *_: (rows, np.full(120, 100), weeks)
        ),
    )

    def load(path: Path) -> SimpleNamespace:
        def predict(inputs):
            if path.name == "previous":
                return [20] * len(inputs)
            return [protected if row["category"] == 200 else major for row in inputs]

        return SimpleNamespace(metadata=metadata, predict=predict)

    monkeypatch.setattr(evaluation, "load", load)
    report = evaluation.evaluate(
        tmp_path, tmp_path / "new", tmp_path / "previous", weeks[0], weeks[-1]
    )
    assert report["tests"]["relative_wape_gain"]
    assert not report["passed"]
    assert failures <= set(report["reasons"])
