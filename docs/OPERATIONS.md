# Operations Guide

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional:

```bash
cp .env.example .env
```

## 2. Data Ingestion

```bash
PYTHONPATH=src python3 scripts/run_ingestion_jobs.py
```

Outputs:

- `data/raw/imd/`
- `data/raw/nasa_power/`
- `data/raw/sentinel/`
- `data/india/india_regions.csv`
- `data/india/rainfall_windows.csv`

## 3. Model Training

Baseline risk model:

```bash
PYTHONPATH=src python3 train.py
```

Baseline rainfall forecaster:

```bash
PYTHONPATH=src python3 train_forecaster.py
```

LSTM rainfall forecaster:

```bash
PYTHONPATH=src python3 train_lstm_forecaster.py
```

## 4. Dashboard

```bash
streamlit run src/landslide_ai/dashboard/app.py
```

## 5. API

```bash
PYTHONPATH=src python3 run_api.py
```

## 6. Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## 7. Docker

```bash
docker build -t landslide-ai-system .
docker run -p 8000:8000 landslide-ai-system
```

## Real-Data Notes

- `LANDSLIDE_NASA_POWER_URL` should be set to the NASA POWER daily point endpoint base.
- `LANDSLIDE_IMD_SOURCE_URL` should point to a downloadable rainfall data source or ETL feed.
- `LANDSLIDE_OFFICIAL_DISTRICT_GEOJSON` and `LANDSLIDE_OFFICIAL_STATE_GEOJSON` should point to official boundary files.
