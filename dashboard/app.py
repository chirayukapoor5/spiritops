"""SpiritOps operational dashboard. All values come through the running API."""

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("SPIRITOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")


def request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    method = "GET" if payload is None else "POST"
    response = httpx.request(method, API + path, json=payload, timeout=15)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected response for {path}")
    return result


def display_table(rows: list[dict[str, Any]], empty: str) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info(empty)


def number(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}{unit}"


def percent(value: float | None) -> str:
    return number(None if value is None else value * 100, 1, "%")


def forecasts() -> None:
    manifest = request("/v1/products")
    products = manifest.get("products", [])
    if not products:
        st.info("Ingest a weekly snapshot to make products available for forecasting.")
        return
    with st.container(border=True):
        product_col, week_col = st.columns([2, 1])
        product = product_col.selectbox("Product", products)
        week = manifest["next_week"]
        week_col.metric("Week beginning", week)
        if st.button("Generate forecast", type="primary"):
            st.session_state.generated = request("/v1/predict", {"product": product, "week": week})
        generated = st.session_state.get("generated", {})
        if generated.get("product") == product and generated.get("week") == week:
            st.metric("Forecast purchase volume", number(generated["prediction"], 2, " L"))
            st.caption(f"Serving version {generated['version']} · {generated['prediction_id']}")
            if generated.get("lower") is not None and generated.get("upper") is not None:
                st.write(
                    f"Prediction interval: {generated['lower']:,.2f}–{generated['upper']:,.2f} L"
                )
            else:
                st.caption("This model produces a point forecast.")
    st.subheader("Forecasts and reported purchases")
    observed = [
        row for row in request("/v1/observations")["items"] if str(row["product"]) == str(product)
    ]
    if observed:
        frame = pd.DataFrame(observed).sort_values("week")
        # Multiple serving versions may forecast the same week. Preserve separate series.
        series = frame.pivot_table(
            index="week", columns="version", values="prediction", aggfunc="last"
        )
        series.columns = [f"Forecast · v{version}" for version in series.columns]
        if "actual" in frame and frame["actual"].notna().any():
            series["Reported purchases"] = frame.groupby("week")["actual"].last()
        st.line_chart(series, x_label="Week beginning", y_label="Litres")
    display_table(
        observed, "No recorded forecasts for this product. Generate its first forecast above."
    )


def models() -> None:
    active = request("/v1/models/current")
    left, right = st.columns(2)
    left.metric("Serving version", str(active.get("version", "—")))
    right.metric("Validated through", str(active.get("validation_end") or "—"))
    with st.expander("Deployment manifest"):
        st.json(active)
    st.subheader("Candidate evaluations")
    releases = request("/v1/models/history")["items"]
    candidates = [row for row in releases if row["outcome"] in {"PROMOTE", "REJECT_CANDIDATE"}]
    if not candidates:
        st.info("Completed training jobs will appear here with their release decisions.")
        return
    selected = st.selectbox(
        "Candidate",
        range(len(candidates)),
        format_func=lambda i: (
            f"v{candidates[i]['version']} · {candidates[i]['kind']} · {candidates[i]['outcome']}"
        ),
    )
    event = candidates[selected]
    report = event["evaluation"]
    st.write(event["reason"])
    display_table(
        [
            {
                "Model": role.title(),
                "WAPE": percent(report[role].get("wape")),
                "MAE (L)": report[role].get("mae"),
                "Bias": percent(report[role].get("bias")),
            }
            for role in ("candidate", "incumbent")
            if isinstance(report.get(role), dict)
        ],
        "This release has no paired accuracy measurements.",
    )
    with st.expander("Full evaluation and gate results"):
        st.json(report)


def monitoring() -> None:
    performance = request("/v1/monitoring/performance")
    drift = request("/v1/monitoring/drift")
    metrics = performance.get("metrics") or performance
    cols = st.columns(3)
    cols[0].metric(
        "Rolling WAPE",
        percent(metrics.get("wape")),
    )
    cols[1].metric("Rolling MAE", number(metrics.get("mae"), 2, " L"))
    cols[2].metric("Forecast bias", percent(metrics.get("bias")))
    st.metric("Prediction interval coverage", percent(metrics.get("interval_coverage")))
    if metrics.get("wape") is None:
        st.info(
            "Forecast accuracy is available after sufficient actual purchases have been matched to predictions."
        )
    st.subheader("Segment performance")
    segments = performance.get("segments") or {}
    if isinstance(segments, dict):
        segments = [{"segment": segment, **values} for segment, values in segments.items()]
    display_table(segments, "No product segment has measured performance yet.")
    st.subheader("Prospective shadow forecasts")
    display_table(
        request("/v1/monitoring/shadow")["items"],
        "The worker has yet to issue a paired shadow forecast.",
    )
    st.subheader("Feature drift")
    if drift.get("alert"):
        st.warning(f"Drift threshold exceeded for {drift['feature']}.")
    left, right = st.columns(2)
    left.metric("Mean standardized shift", number(drift.get("mean_standardized_shift"), 2))
    right.metric("Threshold", number(drift.get("threshold"), 2))
    st.caption(f"{drift['feature']} · {drift['samples']} observed predictions")
    st.subheader("Controller decisions and alerts")
    display_table(
        request("/v1/monitoring/decisions")["items"],
        "The controller has not recorded a decision yet.",
    )
    with st.expander("Full performance report"):
        st.json(performance)


def pipeline() -> None:
    ingestion = request("/v1/ingestion")
    st.subheader("Ingestion and validation")
    summary = st.columns(3)
    summary[0].metric("Products", len(ingestion.get("products", [])))
    summary[1].metric("First complete week", str(ingestion.get("first_week", "—")))
    summary[2].metric("Last complete week", str(ingestion.get("last_week", "—")))
    with st.expander("Current snapshot and data quality"):
        st.json(ingestion)
    st.subheader("Training workflows")
    jobs = request("/v1/jobs")["items"]
    if not jobs:
        st.info("Queued, running and completed jobs will appear here.")
        return
    preview = [
        {key: row[key] for key in ("id", "status", "attempt", "created_at", "error") if key in row}
        for row in jobs
    ]
    display_table(preview, "")
    selected = st.selectbox("Job detail and execution log", [str(row["id"]) for row in jobs])
    detail = request(f"/v1/jobs/{selected}")
    display_table(detail.get("logs", []), "No execution messages yet.")
    with st.expander("Full job detail"):
        st.json(detail)


def releases() -> None:
    events = request("/v1/models/history")["items"]
    if not events:
        st.info(
            "Promotions, rejections and rollbacks will appear here as candidates are evaluated."
        )
        return
    display_table(
        [
            {
                key: row.get(key)
                for key in ("created_at", "outcome", "version", "previous_version", "reason")
            }
            for row in events
        ],
        "",
    )
    chosen = st.selectbox(
        "Inspect event",
        range(len(events)),
        format_func=lambda i: (
            " · ".join(
                str(events[i][k])
                for k in ("at", "created_at", "outcome", "version")
                if k in events[i]
            )
            or f"Event {i + 1}"
        ),
    )
    selected = events[chosen]
    left, right = st.columns(2)
    left.metric("Outcome", str(selected["outcome"]))
    right.metric("Version", str(selected["version"]))
    st.write(selected["reason"])
    with st.expander("Full deployment record"):
        st.json(selected)


def operations() -> None:
    report = request("/v1/operations")
    top = st.columns(3)
    top[0].metric("P95 API latency", number(report.get("p95_ms"), 1, " ms"))
    top[1].metric("Request throughput", number(report.get("throughput_per_second"), 2, " req/s"))
    top[2].metric("Resident memory", number(report.get("rss_mb"), 1, " MB"))
    bottom = st.columns(3)
    requests, errors = report.get("requests"), report.get("errors")
    bottom[0].metric("Requests", number(requests, 0))
    bottom[1].metric("Errors", number(errors, 0))
    bottom[2].metric(
        "Error rate",
        number(100 * errors / requests if requests and errors is not None else None, 2, "%"),
    )
    st.subheader("Lifecycle worker")
    st.json(report.get("worker", {}))
    with st.expander("Full operational report"):
        st.json(report)


def main() -> None:
    st.set_page_config(page_title="SpiritOps · Operations", page_icon="📈", layout="wide")
    st.sidebar.title("SpiritOps")
    st.sidebar.caption("Wholesale purchase forecasting")
    views = {
        "Forecasts": forecasts,
        "Model management": models,
        "Model monitoring": monitoring,
        "Pipeline management": pipeline,
        "Deployment history": releases,
        "Operational metrics": operations,
    }
    selected = st.sidebar.radio("Workspace", list(views), label_visibility="hidden")
    st.sidebar.button("Refresh data")
    st.title(selected)
    try:
        views[selected]()
    except httpx.HTTPStatusError as exc:
        st.error(f"API returned HTTP {exc.response.status_code}.")
        st.code(exc.response.text[:800])
    except httpx.RequestError:
        st.error(f"The API at {API} is unreachable. Check the API service, then refresh.")
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to read this API response: {exc}")


if __name__ == "__main__":
    main()
