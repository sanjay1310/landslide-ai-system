# API Guide

## Base Server

```bash
PYTHONPATH=src python3 run_api.py
```

Default base URL:

```text
http://127.0.0.1:8000
```

If API auth is enabled, pass:

```text
X-API-Key: <your secret>
```

## Endpoints

### `GET /health`

Returns service status.

Example response:

```json
{
  "status": "ok",
  "service": "landslide-ai-system"
}
```

### `GET /config/status`

Returns non-secret runtime configuration summary.

### `POST /assess`

Runs landslide assessment on a region CSV.

Request:

```json
{
  "csv_path": "data/india/india_regions.csv"
}
```

### `POST /scenario`

Runs scenario simulation and returns recalculated risk scores.

Request:

```json
{
  "csv_path": "data/india/india_regions.csv",
  "rainfall_multiplier": 1.25,
  "antecedent_rainfall_multiplier": 1.10,
  "soil_wetness_delta": 0.05,
  "ndvi_delta": -0.03,
  "neighbor_risk_delta": 0.04,
  "target_state": "Kerala"
}
```

### `POST /forecast`

Runs saved forecast inference on rainfall windows.

Request:

```json
{
  "windows_csv_path": "data/india/rainfall_windows.csv"
}
```

### `POST /analytics/summary`

Returns aggregated system analytics for a region dataset.

Request:

```json
{
  "csv_path": "data/india/india_regions.csv"
}
```

### `POST /graph/infer`

Runs saved graph-model inference on a CSV that contains the graph feature columns.

Request:

```json
{
  "csv_path": "data/india/india_graph_training.csv",
  "artifact_path": "artifacts/graph_gnn.pt"
}
```

Example response shape:

```json
{
  "row_count": 24,
  "artifact_path": "artifacts/graph_gnn.pt",
  "rows": [
    {
      "region_id": "IN-AIZ-AIZAWL__2026-04-01",
      "state": "Mizoram",
      "district": "Aizawl",
      "label": 0,
      "graph_gnn_probability": 0.93
    }
  ]
}
```

## Graph Model Workflow

Build the richer India graph-training dataset:

```bash
PYTHONPATH=src python3 scripts/run_ingestion_jobs.py
```

This writes:

```text
data/india/india_graph_training.csv
```

Train the graph model:

```bash
PYTHONPATH=src python3 train_gnn.py
```

Current behavior:

- training uses a validation-aware split instead of reporting only train-on-train scores
- graph features include rainfall ratios, interaction terms, seasonal encoding, and spatial load features
- the saved artifact is written to `artifacts/graph_gnn.pt`

Enable graph-model blending in the main app/API pipeline by setting:

```text
LANDSLIDE_ENABLE_GRAPH_MODEL=true
LANDSLIDE_GRAPH_MODEL_ARTIFACT=artifacts/graph_gnn.pt
```

## Example Requests

Graph inference with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/graph/infer \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/india/india_graph_training.csv",
    "artifact_path": "artifacts/graph_gnn.pt"
  }'
```

Analytics summary with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/analytics/summary \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/india/india_regions.csv"
  }'
```

## Local Usage Examples

See [api_examples.py](/Users/sanjaykumar/Documents/New project 2/scripts/api_examples.py) for ready-made request payloads.
