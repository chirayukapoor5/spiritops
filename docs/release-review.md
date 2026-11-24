# Release review

- Feedback requires the administrator credential because labels affect monitoring and retraining. Live monitoring excludes historical replay and forecasts issued after a week closed at midnight in Iowa. Drift uses the verified training snapshot. Ground truth and readiness check the curated snapshot's integrity.
- Forecast batches pin a single checked feature panel. Parallel retries return the first persisted prediction, interval and feature vector. SQLite writes stay durable and queue within the API process. Telemetry writes leave the ASGI event loop free.
- The review exercised chronological features, model gates, idempotent jobs and predictions, artifact checks, and recovery after a failed activation or restart through the regression suite. The controlled statewide replay used cached models and data as evidence for this release.

## Concurrent HTTP requests

On Windows, one Uvicorn process served the cached statewide LightGBM through loopback HTTP with SQLite durability enabled. Each repeat run sent 160 requests for one warmed product. Eight-client repeat P95 is the median of three run P95s. The distinct-product run issued 99 new forecasts. All requests succeeded.

| Workload | Before P95 | After P95 |
| --- | ---: | ---: |
| Serial repeat | 26.1 ms | 26.6 ms |
| Eight-client repeat | 303.3 ms | 141.8 ms |
| Eight-client distinct | 431.1 ms | 307.3 ms |

The largest original repeat-path cost was the prediction write stage, with P95 between 231 and 262 ms. The read-only retry path reduced that range to approximately 20–24 ms. Durable telemetry and first issuance remain serialized on one database. Distinct-product throughput was 30.5 requests/s after the change versus 38.9 before in one run per revision. These measurements describe this workload and host.

## Monitoring validation

The default-policy replay covered August 25 through December 22, 2025. It recorded 12 HOLD and six WAIT_FOR_LABELS decisions. The controller reserved a validation window after the November 24 reporting week. That window ends January 19, 2026, beyond the cached data, so no job was enqueued.

The separate controlled replay lowered the alert threshold and injected a missing artifact. It verified promotion, later candidate rejection, 200 predictions scored before their labels, and rollback in 2.26 seconds with no errors across 64 requests during retraining and activation. The dashboard recording reads those persisted replay events.

The holiday challenger lost to the seasonal baseline. Holiday interval coverage reached 76.7% against a nominal 80%. The evidence covers archived transactions and a single host. Live operations require an up-to-date reporting feed and prospective forecasts.

To repeat these checks, pass the directory of the saved statewide lifecycle run as `--cached-run` to `scripts/profile_serving.py` or `scripts/check_monitoring.py`. Both commands accept an `--output` JSON path and operate on temporary copies of the recorded snapshot and artifacts.

[HTTP profile](../reports/serving-profile.json) · [Default controller replay](../reports/monitoring-defaults.json) · [Controlled statewide lifecycle](../reports/statewide-lifecycle.json)
