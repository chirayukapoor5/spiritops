# SpiritOps

Forecast weekly wholesale spirits purchases in litres and follow a fitted model from ingestion to rollback. The local platform joins issued forecasts to matured observations, queues training in a separate process, compares releases on chronological holdouts and serves the model identified by its durable deployment manifest.

[Project showcase](https://chirayukapoor5.github.io/spiritops/) · [CI](https://github.com/chirayukapoor5/spiritops/actions/workflows/ci.yml) · [Benchmark evidence](reports/benchmark.json)

## Start locally

From the repository root on Linux or Windows with WSL2 and Docker Compose v2:

```sh
docker compose up --build -d --wait --wait-timeout 1800
```

Open the [dashboard](http://localhost:8501), [API documentation](http://localhost:8000/docs) or [MLflow](http://localhost:5000). Startup ingests the bundled historical sample and registers the baseline. Use `docker compose down` to stop services and retain local volumes.

## Reproduce the experiment

Install uv 0.9.18, then run these commands from the clone. The locked environment uses Python 3.11. On Windows the same commands work in PowerShell.

```sh
uv sync --locked --python 3.11
uv run spiritops demo --report reports/local-replay.json
uv run spiritops benchmark --report reports/local-benchmark.json
uv run pytest
```

Each replay creates an isolated registry, operational databases, worker and HTTP server. The explicit historical clock allows observations to mature. The demo records any controlled policy change and the actual release-gate decision. Use `uv run spiritops data --mode full` for the statewide 2023–25 pipeline. Hardware and workload details accompany each benchmark.

## Results and scope

The recorded two-store replay covered 100 selected products. The LightGBM challenger improved holdout WAPE from 147.76% to 97.79%. Its aggregate bias was −88.55%, a substantial underforecast on this sparse panel. Treat it as a demonstration of release engineering. The local API served 91 error-free requests during retraining and activation. The benchmark records the workload, alert sensitivity change, latency under concurrency, fault injection and restored version. [Measurements](reports/benchmark.json)

## Engineering guide

- [Data and attribution](docs/data.md)
- [Architecture and interfaces](ARCHITECTURE.md)
- [Lifecycle decisions](DECISIONS.md)
- [Deployment and failure recovery](docs/operations.md)
- [Verification and remaining work](STATUS.md)
- [Internship skills and evidence](docs/portfolio.md)

The application code uses the MIT license. The Iowa source extract retains its publisher's CC BY 4.0 attribution. OpenAI Codex assisted implementation, testing, debugging and documentation.

