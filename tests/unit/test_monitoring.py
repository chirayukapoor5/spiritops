from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from spiritops.lifecycle import Policy, decide
from spiritops.monitoring import measures, reporting_close
from spiritops.monitoring.observations import db, label, note, recent, save_batch, traffic
from spiritops.monitoring.shadow import summary


def test_error_metrics_and_unavailable_labels():
    missing = {"actual": None, "prediction": 100, "seasonal": 10}
    rows = [
        missing,
        {"actual": 10, "prediction": 5, "seasonal": 20},
        {"actual": 0, "prediction": 1, "seasonal": 0},
    ]
    result = measures(rows)
    assert result["mae"] == 3
    assert result["bias"] == pytest.approx(-0.4)
    assert result["wape"] == pytest.approx(0.6)
    assert result["seasonal_scaled_error"] == pytest.approx(6 / 21)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert measures([missing])["wape"] is None
    assert measures([rows[-1]])["wape"] is None


def test_reporting_week_closes_at_iowa_midnight():
    utc = timezone.utc
    assert reporting_close("2025-12-15") == datetime(2025, 12, 22, 6, tzinfo=utc).timestamp()
    assert reporting_close("2025-08-25") == datetime(2025, 9, 1, 5, tzinfo=utc).timestamp()


def test_shadow_summary_uses_only_labels_received_after_scoring(tmp_path):
    rows = [
        ("early", "a", "2025-12-15", "2", 8, 10, 10, "group", "data", "snapshot_forecast", 1),
        ("later", "b", "2025-12-15", "2", 8, 10, 10, "group", "data", "snapshot_forecast", 1),
    ]
    with db(tmp_path) as conn:
        conn.executemany("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany(
            "INSERT INTO prospective_shadow VALUES(?,?,?,?)",
            [("early", "3", 9, 4), ("later", "3", 9, 4)],
        )
        conn.executemany(
            "INSERT INTO labels VALUES(?,?,?,?)",
            [("a", "2025-12-15", 100, 2), ("b", "2025-12-15", 10, 5)],
        )
    report = summary(tmp_path)[0]
    assert report["issued"] == 2
    assert report["matured"] == 1
    assert report["champion_wape"] == pytest.approx(0.2)
    assert report["challenger_wape"] == pytest.approx(0.1)


def test_parallel_forecast_retries_preserve_first_issuance(tmp_path):
    row = dict(
        product="a",
        week="2025-12-15",
        version="2",
        prediction=12,
        seasonal=10,
        lag1=8,
        segment="1",
        snapshot="snapshot",
        source="snapshot_forecast",
        created=1,
        lower=1,
        upper=24,
        features={"lag1": 8},
    )

    def issue(i: int) -> dict:
        return save_batch(tmp_path, [{**row, "id": str(i)}])[0]

    with ThreadPoolExecutor(8) as pool:
        forecasts = list(pool.map(issue, range(32)))
    assert all(forecast == forecasts[0] for forecast in forecasts)
    with db(tmp_path) as conn:
        for table in ("predictions", "intervals", "inputs"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, "RETRAIN"),
        ({"busy": True}, "HOLD"),
        ({"cooldown": True}, "HOLD"),
        ({"spent": True}, "HOLD"),
        ({"quality": "incomplete week"}, "QUARANTINE"),
        ({"trainable": False}, "WAIT_FOR_LABELS"),
        ({"reference": None}, "INVESTIGATE"),
        ({"windows": []}, "WAIT_FOR_LABELS"),
        ({"reference": 0.6}, "HOLD"),
        ({"reference": 0.6, "drift": True}, "INVESTIGATE"),
    ],
)
def test_policy(change, expected):
    inputs = dict(
        windows=[dict(samples=30, coverage=1, seasonal_scaled_error=0.5)] * 2,
        reference=0.3,
        drift=False,
        busy=False,
        cooldown=False,
        spent=False,
        trainable=True,
        quality=None,
    )
    inputs.update(change)
    assert decide(Policy(), **inputs)[0] == expected


@pytest.mark.parametrize(
    "params",
    [
        {"min_samples": 0},
        {"excess_error": float("nan")},
        {"cooldown_hours": -1},
        {"min_coverage": 2},
    ],
)
def test_policy_rejects_invalid_configuration(params):
    with pytest.raises(ValueError):
        Policy(**params)


def test_concurrent_feedback_and_audit_persistence(tmp_path):
    with ThreadPoolExecutor(8) as pool:
        writes = list(pool.map(lambda _: label(tmp_path, "sku", "2023-01-02", 42), range(24)))
    assert sum(writes) == 1
    with pytest.raises(ValueError, match="different"):
        label(tmp_path, "sku", "2023-01-02", 43)
    with pytest.raises(ValueError, match="finite"):
        label(tmp_path, "sku", "2023-01-09", float("inf"))
    note(tmp_path, "HOLD", "same champion", "v1")
    traffic(tmp_path, 422, 10)
    assert recent(tmp_path, "decisions")[0]["reason"] == "same champion"
    with db(tmp_path) as conn:
        assert conn.execute("SELECT count(*) FROM labels").fetchone()[0] == 1
        assert conn.execute("SELECT errors FROM counters").fetchone()[0] == 1
