"""Dashboard contract tests with explicitly synthetic API replies."""

from pathlib import Path
from typing import Any

import httpx
from pytest import MonkeyPatch
from streamlit.testing.v1 import AppTest


DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"


def test_forecast_uses_selected_product_and_displays_returned_version(
    monkeypatch: MonkeyPatch,
) -> None:
    payloads: list[dict] = []

    def reply(method: str, url: str, **kwargs: Any) -> httpx.Response:
        req = httpx.Request(method, url)
        if url.endswith("/products"):
            result: dict = {"products": ["a", "b"], "next_week": "2026-10-05"}
        elif url.endswith("/observations"):
            result = {"items": []}
        else:
            assert method == "POST" and url.endswith("/predict")
            payloads.append(kwargs["json"])
            result = {**kwargs["json"], "prediction": 47.5, "version": "7", "prediction_id": "p1"}
        return httpx.Response(200, request=req, json=result)

    monkeypatch.setattr(httpx, "request", reply)
    view = AppTest.from_file(str(DASHBOARD), default_timeout=20).run()
    view.selectbox[0].select("b").run()
    view.button[0].click().run()
    assert not view.exception
    assert payloads == [{"product": "b", "week": "2026-10-05"}]
    assert any(metric.value == "47.50 L" for metric in view.metric)
    assert any("Serving version 7" in caption.value for caption in view.caption)


def test_failed_api_request_displays_status(monkeypatch: MonkeyPatch) -> None:
    def unavailable(method: str, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(503, request=httpx.Request(method, url), json={"detail": "booting"})

    monkeypatch.setattr(httpx, "request", unavailable)
    view = AppTest.from_file(str(DASHBOARD), default_timeout=20).run()
    assert not view.exception
    assert "503" in view.error[0].value
    assert "booting" in view.code[0].value
