# Data provenance

The Iowa Alcohol Operations Bureau publishes wholesale purchases by Class E licensees. The bundled demonstration projects the official 2023 and 2024 archives onto two Iowa City stores, 2622 and 4161. The projection contains 13,358 invoice lines. Source URLs, retrieval timestamps, archive hashes, metadata hashes and the extract checksum are recorded in the [sample manifest](../src/spiritops/ingestion/resources/demo.json). Redistribution and adaptation are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`spiritops data --mode demo` processes the compressed sample. Full mode resolves annual catalog metadata, streams the official archives and records provenance. The smaller CI extract contains 2023 transactions. All modes use DuckDB validation and weekly aggregation. Inspect the published data/current.json under SPIRITOPS_HOME for the reporting window, product list, quarantine path, checksums and quality counts. Week labels identify Mondays. The product selection ends before model validation. Inference uses only earlier weeks.

[Official 2023 catalog](https://catalog.data.gov/dataset/iowa-liquor-sales-2023) · [2024 catalog](https://catalog.data.gov/dataset/iowa-liquor-sales-2024) · [2025 catalog](https://catalog.data.gov/dataset/iowa-liquor-sales-2025)

