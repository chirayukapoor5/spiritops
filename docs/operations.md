# Local operations

Use a Linux container engine with Docker Compose v2. On Windows, enable Docker Desktop’s WSL2 Linux engine. The commands below use a checkout of the repository as the working directory. Native Python development uses CPython 3.11 and uv 0.9.18. Run `make setup` from a shell with GNU Make, or use `uv sync --locked --python 3.11`.

## Start and stop

```sh
docker compose up --build --detach --wait --wait-timeout 1800
docker compose ps -a
```

The first start needs Internet access for images, locked Python dependencies and the official Iowa demo data. The bootstrap service publishes a weekly snapshot, trains and registers the initial seasonal baseline, then exits with code 0. Read its output with `docker compose logs bootstrap`. API and worker startup wait for bootstrap. The service reuses an existing snapshot and active version on subsequent starts. The default demo is a bounded historical subset. Use a fresh Compose project with `SPIRITOPS_MODE=full` for the full ingest, or `SPIRITOPS_MODE=fixture` for the offline CI workload after building the image. A full ingest requires more disk space, time and memory. On Windows PowerShell, set the mode using `$env:SPIRITOPS_MODE = 'fixture'` before running Compose. Port overrides are listed in [.env.example](../.env.example).

| Service | Local address | What to inspect |
| --- | --- | --- |
| Dashboard | [localhost:8501](http://localhost:8501) | Forecasts, quality, jobs, releases and serving health |
| API | [localhost:8000/docs](http://localhost:8000/docs) | OpenAPI schemas, forecasts and feedback |
| MLflow | [localhost:5000](http://localhost:5000) | Candidate runs, native artifacts and registered versions |
| Prometheus | [localhost:9090](http://localhost:9090) | API and worker time series (monitoring profile) |
| Grafana | [localhost:3000](http://localhost:3000) | Provisioned read-only serving-health dashboard (monitoring profile) |

Enable the optional stack with `docker compose --profile monitoring up --build -d --wait --wait-timeout 1800`. Prometheus scrapes every 15 seconds and retains seven days. Grafana’s five-minute error-rate panel needs traffic and consecutive scrapes. Gauges and counters come from the running API. Throughput and P95 are computed over the latest minute. Counts cover the lifetime of the operational database. Resident memory refers to the API process.

`docker compose down` stops the core services and preserves state. If monitoring is enabled, use `docker compose --profile monitoring down`. The named `state` volume holds curated data, operational databases, local models and MLflow artifacts. Keep the same Compose project name when restarting. For an offline backup, stop the stack before copying the **whole** volume, including SQLite WAL files and model artifacts, then restart. A new project name gets separate volumes. Set different host ports when running two projects simultaneously.

## Admin authentication and troubleshooting

On first startup the API generates a random admin token in the shared local state, readable by the service account. To use write-protected operations, run `docker compose exec api cat /var/lib/spiritops/admin-token` in a private terminal and supply the value as the `X-Admin-Token` header. For a managed credential, set `SPIRITOPS_ADMIN_TOKEN` in your local environment before starting Compose. A configured token takes precedence over the file. Local ports bind to loopback. Anyone with Docker access on the host can inspect application state.

```sh
docker compose ps -a
docker compose logs --tail=100 bootstrap api worker mlflow
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:8000/v1/operations
curl --fail http://127.0.0.1:8000/v1/models/history
```

- **Bootstrap exits nonzero.** Inspect its most recent error and MLflow health. Fix the download, storage or configuration error, then rerun the startup command with the same named volume. A failed initialization prevents serving.
- **Model fails readiness.** Inspect release history for a rollback or failed artifact verification and inspect worker logs. Keep the release and data files for diagnosis. A trusted compatible predecessor is required for rollback. If all validated artifacts are lost, restore the state volume from an offline backup.
- **Worker unavailable.** Inspect `docker compose logs worker` and `GET /v1/jobs`. Restart with `docker compose restart worker`. Observe previous job IDs, attempts and outcomes through the API.
- **Accuracy unavailable.** Match a recorded forecast to matured weekly purchases using the feedback API. The dashboard shows missing measures as a dash. Feature drift and forecast error have separate reports.
- **Out-of-memory exit.** Inspect the affected service using `docker inspect` and Docker Desktop resource settings. Bootstrap and worker have 3 GiB limits in Compose; configure resources for the chosen data mode. Restart with existing state.

## Verification and release

`make test` runs lint, type checks, focused unit tests, integration tests and end-to-end tests. The CI workflow additionally builds and starts the locked container stack on the offline fixture, checks readiness and an actual HTTP prediction, restarts the API and worker, and verifies the same model version and result. It also runs the automated lifecycle demonstration in separate state. JUnit reports, coverage, Compose service status, logs and smoke results are attached to the workflow run.

For a local container check, run `python scripts/smoke.py --output before.json`, `docker compose restart api worker`, then `python scripts/smoke.py --expect before.json --output after.json`. Set `--base` and `--dashboard` when using non-default ports. The check issues a real forecast and records it in the selected deployment.

A release tag must exactly match the project version, for example `v0.1.0`. The release workflow validates the tag revision through CI before pushing its Linux/amd64 image to GHCR. BuildKit attaches build provenance and a software bill of materials. The release's `release.json` records the image digest, commit, lockfile checksum and validation URL. Follow the release’s digest-based Compose instructions from a checkout of that tag. Code release leaves model selection to the local controller.

Publish the static showcase by selecting **GitHub Actions** as the repository’s Pages source and pushing `docs/showcase/` to the default branch. The Pages workflow only uploads files from that directory.
