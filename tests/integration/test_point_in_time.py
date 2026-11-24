from datetime import timedelta

import numpy as np
import pytest

from spiritops.cli import fixture
from spiritops.features import panel
from spiritops.training import fit, load


def test_forecasts_invariant_to_future_observations(tmp_path):
    fixture(tmp_path)
    history = panel(tmp_path)
    week = history.last - timedelta(weeks=3)
    output = tmp_path / "candidate"
    fit(tmp_path, output, week.isoformat(), history.last.isoformat())
    model = load(output)
    product = next(iter(history.series))
    row = history.features(product, week)
    forecast = model.predict([row])
    i = (week - history.first).days // 7
    series = history.series[product]
    for array in (series.volume, series.category, series.price, series.priced, series.purchases):
        array[i:] = np.arange(len(array) - i) + 1000
    assert history.features(product, week) == row
    assert model.predict([history.features(product, week)]) == forecast
    assert history.features(product, week + timedelta(weeks=1))["lag1"] == 1000
    with pytest.raises(ValueError):
        history.features(product, history.last + timedelta(weeks=2))


def test_snapshot_ingestion_idempotent(tmp_path):
    first = fixture(tmp_path)
    second = fixture(tmp_path)
    assert first["sha256"] == second["sha256"]
    assert first["source_rows"] == second["source_rows"]
    assert sum(first["outcomes"].values()) == first["source_rows"]
