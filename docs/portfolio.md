# Internship alignment

| Responsibility | Implementation and evidence |
| --- | --- |
| Maintain deployed models | Durable release manifest, registry aliases, degradation decisions and automatic rollback in the historical replay |
| Build efficient ML pipelines | DuckDB ingestion, immutable Parquet snapshots and a shared point-in-time feature module |
| Develop production services | Typed FastAPI forecasts, asynchronous administrative jobs and concurrent serving tests |
| Improve delivery and quality | Locked Python environment, Docker Compose, automated checks and GitHub Actions |
| Support the development lifecycle | Requirements, architecture decisions, operational runbook and measured release evidence |

[Employer's role description, JR-055047](https://pernodricard.wd3.myworkdayjobs.com/fr-FR/pernod-ricard/job/Machine-Learning-Engineer-Intern---Paris---January-2027_JR-055047-1)

## Résumé bullets

- Processed 8.24 million Iowa wholesale transaction rows into a reproducible weekly panel covering 100 products across 156 weeks.
- Implemented chronological LightGBM release gates in a local MLflow and FastAPI platform. Promoted a 28.22% WAPE development candidate, rejected the later holiday challenger, and verified 200 shadow forecasts scored before their labels arrived in historical replay.
- Served 64 valid API responses during controlled retraining and activation, then restored the previous model in 2.26 seconds after injecting an artifact failure.

[Data evidence](../reports/statewide-ingestion.json) · [Statewide lifecycle](../reports/statewide-lifecycle.json) · [Final release review](release-review.md)

