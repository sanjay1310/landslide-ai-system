# NASA POWER Setup

## Purpose

This guide explains how to switch the project from placeholder rainfall ingestion to real NASA POWER data.

## Base Endpoint

Use the NASA POWER daily point API base URL:

```text
https://power.larc.nasa.gov/api/temporal/daily/point
```

Set it in `.env`:

```text
LANDSLIDE_NASA_POWER_URL=https://power.larc.nasa.gov/api/temporal/daily/point
LANDSLIDE_NASA_POWER_START_DATE=20260101
LANDSLIDE_NASA_POWER_END_DATE=20260131
```

The project now defaults to this NASA POWER base URL if you do not override it.

The ingestion code automatically adds query parameters for:

- `PRECTOTCORR`
- `T2M`
- latitude
- longitude
- start date
- end date
- JSON format

## Run

```bash
PYTHONPATH=src python3 scripts/run_ingestion_jobs.py
```

## Outputs

- `data/raw/nasa_power/nasa_power_daily.csv`
- `data/raw/nasa_power/nasa_power_latest.csv`
- `data/india/india_regions.csv`
- `data/india/rainfall_windows.csv`

## Validation

After ingestion:

```bash
PYTHONPATH=src python3 scripts/run_saved_forecast_inference.py
PYTHONPATH=src python3 main.py
```
