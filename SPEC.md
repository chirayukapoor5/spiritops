# SPIRITOPS — PRODUCTION MLOPS ENGINEERING SPECIFICATION

## 1. Engineering mandate

Act as a principal MLOps engineer, senior Python engineer, data engineer, DevOps engineer and software quality engineer.

Build a complete, functional and independently reproducible MLOps platform named SpiritOps, specifically designed to demonstrate the technical skills required by the Pernod Ricard Machine Learning/MLOps internship.

Execute the complete software development lifecycle autonomously: architecture, implementation, testing, integration, deployment, benchmarking, debugging and documentation.

This is an engineering portfolio project, not an academic research project.

Do not produce research papers, publication plans or unnecessary academic experiments.

Do not stop after generating code. Execute the application, inspect results, resolve failures and verify the implementation.

## 2. Mandatory constraints

- Entirely zero-cost software infrastructure.
- No credit-card-dependent services.
- No paid cloud subscriptions.
- No paid APIs.
- No LLM dependency.
- No GPU requirement.
- No proprietary datasets.
- All mandatory application services run locally.
- Use standard free GitHub Actions runners for a public repository.
- Support Windows with WSL2.
- Prefer simple, maintainable solutions over unnecessary architectural complexity.
- Never fabricate performance metrics or test results.
- Record external code attribution and AI-assisted development accurately.

## 3. Product objective

Build an automated model lifecycle platform that forecasts weekly wholesale spirits purchase volumes.

The platform must continuously monitor the deployed model, identify degradation, initiate retraining, evaluate candidate models, automatically promote qualified models and safely restore previous versions when deployment fails.

The system must continue serving valid predictions while retraining takes place.

The principal deliverable is a dependable MLOps platform rather than an unusually complex forecasting model.

## 4. Authentic data ingestion pipeline

Use the official Iowa Liquor Sales dataset.

Official dataset sources:

- https://catalog.data.gov/dataset/iowa-liquor-sales-2023
- https://catalog.data.gov/dataset/iowa-liquor-sales-2024
- https://catalog.data.gov/dataset/iowa-liquor-sales-2025

Additional historical years may be included if necessary.

Resolve and verify actual download URLs from official metadata.

The dataset represents wholesale purchase transactions rather than direct consumer demand.

Implement a reproducible pipeline that:

- Downloads authentic source data.
- Records dataset provenance.
- Generates SHA-256 checksums.
- Supports incremental ingestion.
- Validates data types and required fields.
- Detects duplicate records.
- Detects invalid or missing values.
- Handles returns and transaction corrections explicitly.
- Quarantines invalid records.
- Produces data quality reports.
- Aggregates transactions into weekly product-level purchase volumes.
- Stores curated datasets in Parquet.
- Uses DuckDB for efficient processing.

Select approximately 100 products using initial training-period data only.

Start with a manageable real-data subset before processing the full dataset.

Use bounded-memory processing throughout.

Implement idempotent ingestion so rerunning the same batch does not duplicate records.

Do not include full raw datasets in Git.

## 5. Feature engineering pipeline

Build a shared feature engineering module used by both training and inference.

Include:

- Historical purchase-volume lags.
- Rolling averages.
- Rolling standard deviations.
- Calendar features.
- Seasonal patterns.
- Product category.
- Historical pricing where available.
- Purchase-frequency features.
- Historical trend indicators.

Ensure that only information available at the prediction timestamp is used.

Prevent target leakage and future feature leakage.

Create versioned feature schemas.

Implement automatic compatibility checks between trained models and inference features.

Write automated tests verifying that modifying future observations does not affect earlier predictions.

## 6. Model training pipeline

Implement a seasonal naive forecasting baseline and a global LightGBM forecasting model.

Add quantile forecasting to provide lower, median and upper purchase-volume predictions if the core lifecycle is already functional.

Use chronological training and validation splits.

Do not randomly shuffle time-series observations.

The training pipeline must:

1. Load a validated data version.
2. Generate features.
3. Validate feature schemas.
4. Train baseline and candidate models.
5. Evaluate forecasting performance.
6. Save model artifacts.
7. Register the candidate in MLflow.
8. Produce a machine-readable evaluation report.

Track WAPE, MAE, forecast bias, training duration and resource consumption.

Do not pursue unnecessary hyperparameter optimization.

A model that is simple, dependable and inexpensive to retrain is preferable to a model with marginally better accuracy but poor operational characteristics.

## 7. MLflow integration

Deploy a local MLflow tracking server with persistent SQLite metadata storage and local artifact storage.

Track every training run.

Record:

- Model parameters.
- Evaluation metrics.
- Training dataset version.
- Dataset checksum.
- Feature schema version.
- Training duration.
- Model artifact size.
- Git commit SHA.
- Dependency versions.
- Model signature.
- Validation results.

Implement registered model versions.

Use champion and challenger aliases for model lifecycle management.

Preserve previous validated versions.

Do not overwrite trained artifacts.

Ensure that the inference service reports the model version actually loaded rather than simply reporting the current MLflow alias.

## 8. Automated pipeline orchestration

Implement the following workflows:

- Data ingestion.
- Data validation.
- Feature generation.
- Model training.
- Candidate evaluation.
- Model registration.
- Model deployment.
- Monitoring.
- Automatic retraining.

Use Prefect's open-source local orchestration capabilities or a simpler durable Python worker if Prefect introduces unnecessary operational overhead.

Workflows must support:

- Explicit dependencies.
- Configurable retries.
- Timeouts.
- Persistent job status.
- Failure reporting.
- Execution logs.
- Idempotency.
- Duplicate job prevention.

A failed upstream step must prevent dependent steps from executing incorrectly.

The inference service must remain independent of long-running training operations.

## 9. Continuous model monitoring

Implement a monitoring service using recorded predictions and actual observations.

Monitor three categories.

### Data quality

- Missing values.
- Feature schema changes.
- Invalid ranges.
- Dataset freshness.
- Unknown products.
- Incomplete reporting periods.

### Model quality

- Rolling WAPE.
- Rolling MAE.
- Forecast bias.
- Feature distribution drift.
- Seasonality-adjusted performance degradation.
- Product-segment performance.
- Prediction interval coverage if quantile forecasting is implemented.

### System quality

- API error rates.
- Request throughput.
- Inference latency.
- Model loading failures.
- Training failures.
- Memory consumption.
- Worker availability.

Use Evidently OSS where appropriate and Prometheus-compatible metrics for operational monitoring.

Use actual model predictions and actual outcomes for forecasting-performance calculations.

When ground truth has not arrived, report performance as unavailable rather than inventing an accuracy estimate.

Distinguish feature drift from confirmed predictive degradation.

## 10. Automatic retraining controller

Implement a configurable lifecycle controller.

The controller must decide between:

- HOLD
- INVESTIGATE
- WAIT_FOR_LABELS
- QUARANTINE
- RETRAIN
- REJECT_CANDIDATE
- PROMOTE
- ROLLBACK

Retraining must consider:

- Persistent error degradation.
- Drift alerts.
- Available ground truth.
- Data quality.
- Minimum evaluation sample size.
- Retraining cooldown.
- Available computational budget.
- Existing training jobs.

Do not trigger unlimited retraining because one monitoring threshold is exceeded.

Prevent repeated retraining jobs for the same event.

Log every decision and its reason.

The controller must automatically enqueue retraining when its configured conditions are satisfied.

Make the thresholds configurable and validate them against historical development data.

## 11. Champion–challenger evaluation

Every newly trained candidate must be compared against the currently deployed model.

Evaluate both models on the same independent chronological validation window.

Implement automatic release gates:

- Forecasting performance meets the configured improvement requirement.
- Critical product groups do not degrade beyond their tolerance.
- Feature schema remains compatible.
- Predictions are valid and finite.
- Model artifacts pass integrity verification.
- Inference latency meets the configured budget.
- Required automated tests pass.

Initial proposed accuracy gate: at least 2% relative WAPE improvement, subject to validation and an explicitly documented alternative acceptance policy.

Where sufficient newly matured observations are available, implement shadow evaluation.

Reject inferior candidates automatically.

Retain rejected candidates and their evaluation reports in MLflow.

Never promote a candidate simply because training completed successfully.

## 12. Safe model deployment

Implement a release controller that:

1. Retrieves a validated model artifact.
2. Verifies its checksum.
3. Verifies its feature schema.
4. Loads the candidate separately from the incumbent.
5. Executes smoke predictions.
6. Atomically activates the model.
7. Confirms the active serving version.
8. Records the deployment event.
9. Preserves the previous validated version.

The inference service must continue serving its incumbent model during candidate preparation.

Each request must use a consistent model version.

Persist the active deployment manifest.

Recover the active model correctly after application restart.

Treat the deployment manifest as the authoritative serving state and reconcile MLflow aliases with it.

Do not claim distributed deployment guarantees for a single-host application.

## 13. Automatic rollback

Implement automatic rollback when:

- Model loading fails.
- An artifact is corrupted.
- Feature compatibility checks fail.
- Smoke predictions fail.
- A newly activated release violates operational health checks.

Restore the last validated compatible model.

Persist rollback events and failure reasons.

Test interrupted deployments and application restarts.

Do not activate untrusted serialized model artifacts.

Verify through integration tests that the previous model remains available when candidate deployment fails.

## 14. Production-style REST API

Implement FastAPI with typed request and response schemas.

Required endpoints:

GET /health/live

GET /health/ready

POST /v1/predict

POST /v1/predict/batch

POST /v1/feedback

GET /v1/models/current

GET /v1/models/history

GET /v1/monitoring/drift

GET /v1/monitoring/performance

GET /v1/jobs/{job_id}

POST /v1/admin/retrain

POST /v1/admin/rollback

GET /metrics

Implement:

- Input validation.
- Batch size limits.
- Consistent error responses.
- HTTP status codes.
- Request correlation identifiers.
- Structured logging.
- Thread-safe model access.
- Authentication for administrative endpoints.
- Idempotent feedback ingestion.
- Health and readiness checks.
- Persistent job tracking.
- OpenAPI documentation.

Prediction responses must contain the actual model version, forecast period and prediction values.

Compute features using the validated feature store.

Do not trust arbitrary client-provided features.

Administrative retraining requests must return an asynchronous job identifier rather than blocking during training.

Bind administrative and infrastructure interfaces to localhost or an isolated internal container network.

## 15. Persistent operational storage

Use SQLite for:

- Prediction records.
- Ground-truth feedback.
- Model deployment history.
- Lifecycle decisions.
- Training jobs.
- Monitoring alerts.
- Data ingestion manifests.
- Audit events.

Use Parquet and DuckDB for analytical datasets.

Implement database constraints, transactions, indexes and idempotent writes.

Ensure state survives process restarts.

Use explicit job recovery and bounded retries.

Avoid in-memory-only storage for critical operational state.

## 16. Functional web dashboard

Create a Streamlit operational control centre connected to the actual API and operational database through supported interfaces.

Dashboard sections:

### Forecasts

Show predicted and actual purchase volumes, product selection, forecast periods and prediction intervals.

### Model management

Show the deployed model, previous models, challenger candidates, evaluation results and promotion history.

### Model monitoring

Show forecasting errors, drift indicators, product-segment performance and degradation alerts.

### Pipeline management

Show ingestion jobs, training jobs, validation status, execution logs and failed workflows.

### Deployment history

Show model promotions, rejected candidates, rollback events and activation timestamps.

### Operational metrics

Show actual latency, throughput, resource utilization and error rates.

Include meaningful empty states.

All metrics must come from actual application data.

Do not hardcode dashboard metrics.

## 17. Docker deployment

Containerize the application.

Create Docker Compose services for:

- FastAPI.
- Lifecycle worker.
- MLflow.
- Streamlit.

Add optional Prometheus and Grafana monitoring services.

If using Prefect, configure its local server and worker explicitly.

Implement:

- Persistent volumes.
- Internal service networking.
- Health checks.
- Readiness dependencies.
- Restart behavior.
- Environment-based configuration.
- Non-root containers where practical.
- Reproducible dependency versions.
- Local resource limits where supported.

The complete application must start using a documented Docker Compose command.

No paid hosting platform is required.

## 18. Software quality standards

Use a packaged Python application with pyproject.toml and locked dependencies.

Implement:

- Type annotations.
- Ruff linting and formatting.
- Static type checking.
- Clear domain interfaces.
- Modular configuration.
- Structured exception handling.
- Dependency injection where useful.
- Small, testable functions.
- Separation between API, training, monitoring and deployment.
- Reusable pipeline components.

Avoid excessive abstractions, unnecessary microservices and duplicated code.

## 19. Automated testing

Implement meaningful unit tests covering:

- Data validation.
- Feature engineering.
- Forecasting metrics.
- Monitoring thresholds.
- Retraining decisions.
- Model promotion gates.
- Artifact integrity.
- Rollback selection.

Implement integration tests covering:

- Database persistence.
- MLflow model registration.
- Model loading.
- FastAPI predictions.
- Feedback ingestion.
- Automatic training jobs.
- Candidate evaluation.
- Model activation.
- Application restart.

Implement end-to-end tests that execute:

INGEST → VALIDATE → TRAIN → REGISTER → SERVE → MONITOR → RETRAIN → EVALUATE → PROMOTE OR REJECT → ROLLBACK.

Implement fault-injection tests for:

- Invalid data.
- Missing model artifacts.
- Corrupted artifacts.
- Failed training.
- Duplicate jobs.
- Invalid feedback.
- Service restart.
- Concurrent predictions during model activation.

Add property-based tests for critical lifecycle invariants where useful.

Target at least 85% test coverage for core lifecycle, policy and deployment modules.

Measure actual coverage rather than adding superficial tests to reach the target.

## 20. CI/CD pipeline

Configure GitHub Actions workflows.

On pull requests and pushes:

1. Install reproducibly locked dependencies.
2. Run lint checks.
3. Run static type checks.
4. Run unit tests.
5. Run integration tests.
6. Generate coverage reports.
7. Build Docker images.
8. Start a lightweight containerized environment.
9. Execute API smoke tests.
10. Execute a complete lifecycle integration test.
11. Upload test reports.

On validated release tags:

- Build versioned application images.
- Generate release metadata.
- Publish release artifacts using free GitHub capabilities where applicable.
- Provide a reproducible local deployment command.
- Generate a software bill of materials where practical.

Use a small deterministic fixture or appropriately sized real-data subset in CI.

Do not download the complete historical dataset on every commit.

Separate code-release automation from model-release automation.

Never automatically deploy a model that has failed its model quality gates.

## 21. Engineering benchmarks

Execute actual benchmarks on the declared local hardware.

Measure:

- Data processing time.
- Peak ingestion memory.
- Model training duration.
- Model artifact size.
- Prediction accuracy.
- Warm inference latency.
- Concurrent request throughput.
- API error rate.
- Inference latency during retraining.
- Model activation duration.
- Rollback duration.

Initial engineering targets:

- Single-request warm P95 latency below 200 milliseconds.
- Rollback below 10 seconds.
- Zero invalid predictions during controlled model activation.
- No duplicated jobs or feedback records during defined concurrency tests.

Do not fabricate results if these targets are missed.

Investigate bottlenecks and document actual limitations.

Use Locust or an equivalent load-testing tool.

## 22. End-to-end demonstration

Create an automated demonstration script that runs the actual platform.

The demonstration must show:

1. Authentic data ingestion.
2. Data validation.
3. Initial model training.
4. MLflow experiment tracking.
5. Registration of the first champion.
6. Predictions through FastAPI.
7. Historical observation replay.
8. Model degradation detection.
9. Automatic retraining.
10. Candidate evaluation.
11. Rejection of a weaker candidate.
12. Activation of a qualified candidate.
13. Continued inference during deployment.
14. Controlled deployment failure.
15. Restoration of the previous model.
16. Inspection of audit logs and monitoring metrics.

Use explicitly labelled controlled fault injection where naturally occurring historical events cannot reliably demonstrate a particular failure scenario.

No manually edited screenshots or simulated metrics.

## 23. Repository structure

Create a standalone repository named spiritops.

Organize it into:

- src/spiritops/ingestion/
- src/spiritops/validation/
- src/spiritops/features/
- src/spiritops/training/
- src/spiritops/evaluation/
- src/spiritops/registry/
- src/spiritops/monitoring/
- src/spiritops/lifecycle/
- src/spiritops/deployment/
- src/spiritops/api/
- src/spiritops/storage/
- src/spiritops/worker/
- dashboard/
- configs/
- tests/unit/
- tests/integration/
- tests/end_to_end/
- tests/load/
- scripts/
- reports/
- docs/
- monitoring/
- .github/workflows/

Include:

- README.md
- pyproject.toml
- Dependency lockfile
- Dockerfile
- compose.yaml
- Makefile
- LICENSE
- .env.example
- .gitignore

Exclude credentials, raw datasets and oversized generated artifacts.

## 24. Developer onboarding

Provide simple documented commands equivalent to:

make setup

make data

make train

make up

make test

make demo

make benchmark

make down

Implement a lightweight demonstration mode and a full real-data mode.

Both modes must use the same core application components.

The lightweight mode must not replace actual training or API serving with fake results.

Verify setup from a clean environment.

Ensure a new engineer can run the project without undocumented configuration steps.

## 25. Engineering documentation

Create concise practical documentation:

- System architecture.
- Data pipeline and dataset provenance.
- API usage.
- Model lifecycle.
- Deployment procedures.
- Monitoring and alerting.
- Testing instructions.
- Failure recovery runbook.
- Benchmarks.
- Known limitations.

Include architecture decision records for significant technical choices.

Do not create a research paper.

Do not create lengthy academic documents.

Prioritize documentation that helps another engineer understand, run, maintain and troubleshoot the application.

## 26. Public portfolio presentation

Create a static GitHub Pages showcase with:

- Architecture diagram.
- Verified engineering achievements.
- Actual dashboard screenshots.
- CI evidence.
- Model lifecycle demonstration.
- Measured performance benchmarks.
- Links to repository code.
- Reproduction instructions.

Clearly identify the site as a static project showcase rather than a live hosted inference service.

Record a concise video showing the actual platform running locally.

The video should prioritize retraining, model evaluation, deployment and rollback rather than general dashboard navigation.

## 27. Final acceptance criteria

The project is complete only when:

- Real historical data has been processed.
- An actual forecasting model has been trained.
- MLflow records real experiments and model versions.
- FastAPI serves predictions from a fitted model.
- Monitoring consumes recorded predictions and actual observations.
- Automated retraining functions.
- A weaker candidate is rejected.
- A qualifying candidate is activated.
- Rollback is demonstrated.
- Inference remains valid during controlled model changes.
- Persistent state survives restart.
- Docker Compose startup succeeds.
- CI workflows pass.
- Meaningful automated tests pass.
- Actual performance benchmarks are recorded.
- Clean-clone reproduction succeeds.
- No paid infrastructure is required.

Do not represent an incomplete requirement as completed.

## 28. Final independent engineering review

Before declaring completion, review the application as a principal engineer evaluating a job candidate.

Identify:

- Data leakage.
- Fragile deployment logic.
- Missing validation.
- Unsafe deserialization.
- Incorrect model version reporting.
- Uncontrolled retraining.
- Duplicate operational jobs.
- Poor failure recovery.
- Inconsistent API contracts.
- Unnecessary architectural complexity.
- Unsupported performance claims.
- Missing reproducibility evidence.

Fix identified problems and rerun affected tests.

Deliver the final repository, actual CI links, benchmark reports, operational demonstration and a concise mapping between implemented features and the Pernod Ricard job description.

Generate résumé bullet points only from verified results.

The objective is to demonstrate professional MLOps engineering competence, not to maximize code volume or the number of technologies used.