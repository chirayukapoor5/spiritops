"""Local HTTP interface. The service never accepts feature vectors or filesystem paths."""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from spiritops import deployment, features, monitoring, worker
from spiritops.config import home_dir, shift, snapshot
from spiritops.monitoring import observations as obs


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Target(Payload):
    product: str = Field(min_length=1, max_length=80)
    week: date

    @field_validator("week")
    @classmethod
    def monday(cls, value: date) -> date:
        if value.weekday():
            raise ValueError("Forecast periods start on Monday")
        return value


class Feedback(Target):
    actual: float = Field(ge=0, allow_inf_nan=False)


class Batch(Payload):
    items: list[Target] = Field(min_length=1, max_length=100)


class Forecast(Target):
    lower: float | None = None
    upper: float | None = None
    prediction_id: str
    version: str
    prediction: float = Field(ge=0, allow_inf_nan=False)


class ForecastBatch(BaseModel):
    items: list[Forecast]


class TrainRequest(Payload):
    kind: Literal["lightgbm", "baseline"] = "lightgbm"
    validation_start: date | None = None
    validation_end: date | None = None


class RollbackRequest(Payload):
    reason: str = Field(min_length=1, max_length=300, default="Operator requested rollback")


def admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("SPIRITOPS_ADMIN_TOKEN", "")
    if not expected or not hmac.compare_digest((x_admin_token or "").encode(), expected.encode()):
        raise HTTPException(401, "A valid administrator token is required")


def create_app(home: Path | None = None) -> FastAPI:
    state = home or home_dir()
    app = FastAPI(
        title="SpiritOps",
        version="0.1.0",
        description="One-week wholesale purchase forecasts in litres.",
    )
    log = logging.getLogger("uvicorn.error")

    @app.middleware("http")
    async def instrument(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        rid = supplied if re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", supplied) else uuid.uuid4().hex
        request.state.rid = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Unhandled request failure %s", rid)
            response = JSONResponse(
                {"error": "Internal service failure", "request_id": rid}, status_code=500
            )
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = rid
        obs.traffic(state, response.status_code, elapsed)
        log.info(
            json.dumps(
                dict(
                    request_id=rid,
                    path=request.url.path,
                    status=response.status_code,
                    latency_ms=round(elapsed, 2),
                )
            )
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            {"error": exc.detail, "request_id": request.state.rid}, status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        return JSONResponse(
            {
                "error": "Invalid request",
                "fields": [{"at": e["loc"], "message": e["msg"]} for e in exc.errors()],
                "request_id": request.state.rid,
            },
            status_code=422,
        )

    @app.exception_handler(RuntimeError)
    async def unavailable(request: Request, exc: RuntimeError):
        return JSONResponse({"error": str(exc), "request_id": request.state.rid}, status_code=503)

    @app.get("/health/live")
    def live() -> dict:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready() -> dict:
        manifest = snapshot(state)
        current = deployment.current(state)
        if not (state / manifest["path"]).is_file():
            raise RuntimeError("Validated feature store is unavailable")
        return {"status": "ready", "version": str(current["version"])}

    @app.get("/v1/products")
    def products() -> dict:
        m = snapshot(state)
        replay = (state / "replay-clock.json").exists()
        return {
            **m,
            "next_week": monitoring.as_of(state).isoformat() if replay else shift(m["last_week"]),
            "historical_replay": replay,
        }

    @app.get("/v1/ingestion")
    def ingestion() -> dict:
        return snapshot(state)

    def forecast(items: list[Target]) -> list[dict]:
        m = products()
        for item in items:
            if item.product not in m["products"]:
                raise HTTPException(404, "Unknown product")
            when = item.week.isoformat()
            if when != m["next_week"]:
                raise HTTPException(
                    422, f"The feature store supports the forecast period {m['next_week']}"
                )
        rows = [features.features_for(state, i.product, i.week.isoformat()) for i in items]
        model, predictions = deployment.predict_with_metadata(state, rows)
        version = model["version"]
        if any(i.week.isoformat() <= model["validation_end"] for i in items):
            raise HTTPException(409, "Forecast period overlaps model development")
        if snapshot(state)["sha256"] != m["sha256"]:
            raise HTTPException(
                409, "Serving state changed during feature preparation. Retry the request"
            )
        if len(predictions) != len(items) or not all(
            math.isfinite(p) and p >= 0 for p in predictions
        ):
            raise RuntimeError("Serving model returned invalid predictions")
        source = "historical_replay" if m["historical_replay"] else "snapshot_forecast"
        from spiritops.training import prediction_bounds

        return obs.save_batch(
            state,
            [
                dict(
                    id=uuid.uuid4().hex,
                    product=i.product,
                    week=i.week.isoformat(),
                    version=str(version),
                    prediction=float(p),
                    seasonal=row["seasonal_naive"],
                    lag1=row["lag1"],
                    segment=str(row.get("category", 0)),
                    snapshot=m["sha256"],
                    lower=lo,
                    upper=hi,
                    features=row,
                    source=source,
                    created=time.time(),
                )
                for i, row, p in zip(items, rows, predictions, strict=True)
                for lo, hi in [prediction_bounds(model, row, p)]
            ],
        )

    @app.post("/v1/predict", response_model=Forecast)
    def predict(item: Target) -> dict:
        return forecast([item])[0]

    @app.post("/v1/predict/batch", response_model=ForecastBatch)
    def batch(payload: Batch) -> dict:
        return {"items": forecast(payload.items)}

    @app.post("/v1/feedback")
    def feedback(item: Feedback) -> dict:
        if item.product not in snapshot(state)["products"]:
            raise HTTPException(404, "Unknown product")
        if item.week.isoformat() > monitoring.latest_closed(state):
            raise HTTPException(422, "Feedback requires a completed reporting week")
        try:
            created = obs.label(state, item.product, item.week.isoformat(), item.actual)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"created": created}

    @app.get("/v1/models/current")
    def current() -> dict:
        return deployment.current(state)

    @app.get("/v1/models/history")
    def history() -> dict:
        return {"items": deployment.history(state)}

    @app.get("/v1/monitoring/shadow")
    def shadow_report() -> dict:
        from spiritops.monitoring.shadow import summary

        return {"items": summary(state)}

    @app.get("/v1/monitoring/drift")
    def drift() -> dict:
        return monitoring.drift(state, deployment.current(state))

    @app.get("/v1/monitoring/performance")
    def performance() -> dict:
        return monitoring.performance(state, deployment.current(state))

    @app.get("/v1/monitoring/decisions")
    def decisions() -> dict:
        return {"items": obs.recent(state, "decisions")}

    @app.get("/v1/observations")
    def observations() -> dict:
        return {"items": obs.recent(state, "observations")}

    @app.get("/v1/operations")
    def operations() -> dict:
        return monitoring.operations(state)

    @app.get("/v1/jobs")
    def all_jobs() -> dict:
        return {"items": worker.jobs(state)}

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str) -> dict:
        result = worker.job(state, job_id)
        if result is None:
            raise HTTPException(404, "Unknown job")
        return result

    @app.post("/v1/admin/retrain", status_code=202, dependencies=[Depends(admin)])
    def retrain(
        payload: TrainRequest = TrainRequest(),
        idempotency_key: Annotated[str | None, Header(max_length=120)] = None,
    ) -> dict:
        m = snapshot(state)
        params = payload.model_dump(mode="json", exclude_none=True)
        start, end = params.get("validation_start"), params.get("validation_end")
        if bool(start) != bool(end) or (
            start
            and end
            and (
                start > end
                or end > m["last_week"]
                or start <= m["product_selection_end"]
                or date.fromisoformat(start).weekday()
                or date.fromisoformat(end).weekday()
            )
        ):
            raise HTTPException(422, "Provide a complete valid chronological validation window")
        fingerprint = json.dumps([m["sha256"], params], sort_keys=True)
        key = f"manual:{idempotency_key or fingerprint}"
        try:
            job_id = worker.enqueue(state, key, params)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"job_id": job_id}

    @app.post("/v1/admin/rollback", dependencies=[Depends(admin)])
    def rollback(payload: RollbackRequest = RollbackRequest()) -> dict:
        return deployment.rollback(state, payload.reason)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        data = monitoring.operations(state)
        values = {
            "requests_total": data["requests"],
            "errors_total": data["errors"],
            "p95_ms": data["p95_ms"],
            "rss_mb": data["rss_mb"],
            "worker_available": int(bool(data["worker"].get("available"))),
            "throughput_per_second": data["throughput_per_second"],
        }
        return "".join(
            f"spiritops_{name} {value}\n" for name, value in values.items() if value is not None
        )

    return app


app = create_app()
