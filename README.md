# Landslide AI System

An end-to-end landslide risk intelligence project for regional assessment, advisory generation, rainfall forecasting, graph-based risk propagation, API serving, and dashboard analytics.

## Highlights

- Multi-agent assessment pipeline with explainable risk drivers
- Streamlit dashboard with map views, analytics, alerts, and exports
- FastAPI service for assessment, scenario simulation, forecasting, and graph inference
- India-focused sample and regional datasets so the project runs locally without external downloads
- Optional PyTorch-powered deep-risk, LSTM, and graph models with safe fallback behavior
- Training, ingestion, evaluation, and reporting scripts suitable for academic demos and portfolio publication

## Repository Layout

- `src/landslide_ai/agents/`: prediction, graph, vision, and advisory agents
- `src/landslide_ai/api/`: FastAPI application and request/response schemas
- `src/landslide_ai/dashboard/`: Streamlit user interface
- `src/landslide_ai/models/`: baseline, trained, and deep risk models
- `src/landslide_ai/forecasting/`: baseline and LSTM forecasting components
- `src/landslide_ai/ingestion/`: ETL and data-source integration helpers
- `src/landslide_ai/services/`: reusable runtime services for API and UI layers
- `docs/`: architecture, API, operations, and project writeups
- `tests/`: automated regression coverage

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

To enable PyTorch-backed models as well:

```bash
pip install -e ".[ml]"
```

For local development and test tooling:

```bash
pip install -e ".[dev]"
```

## Quick Start

Run the CLI demo:

```bash
landslide-ai-demo
```

Start the API:

```bash
landslide-ai-api
```

Launch the dashboard:

```bash
streamlit run src/landslide_ai/dashboard/app.py
```

Run the tests:

```bash
pytest
```

## Core Workflows

Train the baseline rainfall forecaster:

```bash
python train_forecaster.py
```

Train the optional deep-risk model:

```bash
python train_deep_risk.py
```

Train the optional graph model:

```bash
python train_gnn.py
```

Run saved rainfall forecast inference:

```bash
python scripts/run_saved_forecast_inference.py
```

Run ingestion jobs:

```bash
python scripts/run_ingestion_jobs.py
```

## Configuration

Copy `.env.example` into `.env` and adjust values for your environment.

Important flags:

- `LANDSLIDE_API_AUTH_ENABLED=true` enables API key protection
- `LANDSLIDE_API_KEY` sets the expected `X-API-Key` value
- `LANDSLIDE_ENABLE_DEEP_RISK_MODEL=true` enables the PyTorch deep-risk model when the dependency and artifact are both available
- `LANDSLIDE_ENABLE_GRAPH_MODEL=true` enables graph-model inference when the dependency and artifact are both available

The application safely falls back to non-PyTorch baseline paths when optional ML dependencies are unavailable or unhealthy.

## API Endpoints

- `GET /health`
- `GET /config/status`
- `POST /assess`
- `POST /scenario`
- `POST /forecast`
- `POST /analytics/summary`
- `POST /graph/infer`

## Data Notes

The dashboard prefers:

- `data/india/india_regions.csv`
- `data/gis/india_districts_official.geojson`

If those are not present, the system falls back to `data/sample_regions.csv` where applicable.

## Deployment

### Docker (recommended)

Deploy the full stack (API + dashboard) on any VPS or local machine with Docker:

```bash
cp .env.example .env
# For production, set LANDSLIDE_API_AUTH_ENABLED=true and a strong LANDSLIDE_API_KEY

chmod +x scripts/deploy-docker.sh
./scripts/deploy-docker.sh
```

Manual equivalent:

```bash
docker compose up -d --build
```

Services:

- API: `http://localhost:8000` (docs at `/docs`)
- Dashboard: `http://localhost:8501`

Check logs: `docker compose logs -f`

### Render (cloud)

1. Push this repo to GitHub.
2. In [Render](https://render.com), create a **Blueprint** and connect the repo.
3. Render reads `render.yaml` and creates two services (API + dashboard).
4. Set `LANDSLIDE_API_KEY` in the Render dashboard for the API service.
5. After deploy, use the generated URLs for API and dashboard.

### Production checklist

- Set `LANDSLIDE_API_AUTH_ENABLED=true` and a strong `LANDSLIDE_API_KEY`
- Keep PyTorch models disabled unless you train and ship `.pt` artifacts in the image
- Restrict CORS with `LANDSLIDE_API_CORS_ORIGINS` if the API is public
- Re-run ingestion or copy `data/` and `artifacts/` into the deployment environment

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Guide](docs/API.md)
- [Operations Guide](docs/OPERATIONS.md)
- [NASA POWER Setup](docs/NASA_POWER_SETUP.md)
- [Final Year Writeup](docs/FINAL_YEAR_WRITEUP.md)
