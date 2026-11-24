import pytest
from fastapi.testclient import TestClient

from spiritops.api import create_app
from spiritops.cli import fixture, train
from spiritops.config import shift
from spiritops.demonstration import write_json


@pytest.fixture(scope="module")
def serving(tmp_path_factory):
    root = tmp_path_factory.mktemp("serving")
    fixture(root)
    train(root, "baseline")
    return root


def test_empty_service_is_live_and_not_ready(tmp_path):
    with TestClient(create_app(tmp_path)) as api:
        assert api.get("/health/live").status_code == 200
        assert api.get("/health/ready").status_code == 503
        assert api.get("/v1/jobs/unknown").status_code == 404
        assert api.post("/v1/admin/retrain").status_code == 401


def test_api_data_contracts_and_restart(serving, monkeypatch):
    monkeypatch.setenv("SPIRITOPS_ADMIN_TOKEN", "local-test")
    headers = {"X-Admin-Token": "local-test", "Idempotency-Key": "example"}
    with TestClient(create_app(serving)) as api:
        products = api.get("/v1/products").json()
        target = dict(product=products["products"][0], week=products["next_week"])
        assert api.post("/v1/predict", json=target).is_success
        assert api.get("/v1/observations").json()["items"][0]["source"] == "retrospective_snapshot"
        assert api.get("/v1/monitoring/drift").json()["samples"] == 0
        write_json(serving / "replay-clock.json", {"as_of": target["week"]})
        expected = api.get("/v1/models/current").json()["version"]
        first = api.post("/v1/predict", json=target, headers={"X-Request-ID": "sample-42"})
        assert first.status_code == 200 and first.headers["X-Request-ID"] == "sample-42"
        body = first.json()
        assert 0 <= body["lower"] <= body["prediction"] <= body["upper"]
        assert body["version"] == expected and body["prediction"] >= 0
        assert api.post("/v1/predict/batch", json={"items": [target, target]}).json()["items"] == [
            body,
            body,
        ]
        assert api.post("/v1/predict", json={**target, "product": "unknown"}).status_code == 404
        assert api.post("/v1/predict", json={**target, "lag1": 78}).status_code == 422
        assert (
            api.post("/v1/predict", json={**target, "week": shift(target["week"], -1)}).status_code
            == 422
        )
        assert api.post("/v1/predict/batch", json={"items": [target] * 101}).status_code == 422
        assert api.get("/v1/monitoring/performance").json()["metrics"]["wape"] is None
        assert api.get("/v1/monitoring/drift").json()["samples"] == 1
        assert "spiritops_requests_total" in api.get("/metrics").text
        feedback = {**target, "actual": 32.0}
        assert api.post("/v1/feedback", json=feedback).status_code == 401
        assert api.post("/v1/feedback", json=feedback, headers=headers).status_code == 422
        write_json(serving / "replay-clock.json", {"as_of": shift(target["week"])})
        assert api.post("/v1/feedback", json=feedback, headers=headers).json()["created"]
        assert api.post("/v1/feedback", json=feedback, headers=headers).json() == {"created": False}
        assert (
            api.post("/v1/feedback", json={**feedback, "actual": 0}, headers=headers).status_code
            == 409
        )
        assert (
            api.post("/v1/feedback", json={**feedback, "actual": -10}, headers=headers).status_code
            == 422
        )
        metrics = api.get("/v1/monitoring/performance").json()
        assert metrics["source"] == "historical_replay" and metrics["metrics"]["samples"] == 1
        (serving / "replay-clock.json").unlink()
        assert api.get("/v1/monitoring/performance").json()["metrics"]["samples"] == 0
        identifier = api.post("/v1/admin/retrain", json={}, headers=headers).json()["job_id"]
        assert (
            api.post("/v1/admin/retrain", json={}, headers=headers).json()["job_id"] == identifier
        )
        assert (
            api.post("/v1/admin/retrain", json={"kind": "baseline"}, headers=headers).status_code
            == 409
        )
    with TestClient(create_app(serving)) as restarted:
        assert restarted.get("/health/ready").json()["version"] == expected
        assert restarted.get(f"/v1/jobs/{identifier}").json()["status"] == "queued"
        assert restarted.get("/v1/observations").json()["items"][0]["actual"] == 32
