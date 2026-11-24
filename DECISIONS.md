# Engineering decisions

## Single-host storage and orchestration

DuckDB performs analytical aggregation. An immutable Parquet snapshot supplies feature generation, training and inference. SQLite provides local transactions for the deployment manifest, audit history and job leases. A worker runs training in a separate process so a slow fit or timeout leaves the API free to serve. Operate each database on a local filesystem shared by the containers.

## Model trust and recovery

Store fitted trees in LightGBM's native text format and metadata in JSON. Only worker-produced, locally registered directories are eligible for release. A manifest transaction records the active version and deployment event together. A process verifies the digest, schema and smoke predictions on loading a version. MLflow aliases can be reconstructed from this state. The filesystem and local operator are part of the trust boundary.

## Forecast and evaluation boundaries

The target is weekly wholesale purchase volume in litres. Monday identifies the start of the predicted week. A trained model sees features available at that time. Products are selected before validation begins. The controller persists its alert week and reserves subsequent weeks for champion–challenger comparison. This prevents the weeks selected for poor incumbent performance from doubling as evidence of challenger improvement.

## Bootstrap

The seasonal naive model gives an initial service a verifiable fallback. It may bootstrap following integrity, numerical and operational checks. Later releases face the quality gates even when training and registration succeed.

## Historical replay

The replay owns a fresh home, independent registry and explicit as-of clock. Its forecast API advances one week at a time. Error calculations join issued forecasts to matured actuals. The demonstration records a sensitivity change if needed to exercise the controller. Fault injection moves the active artifact temporarily inside that isolated home.

