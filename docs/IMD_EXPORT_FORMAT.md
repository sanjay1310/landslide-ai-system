# IMD Export Format

## Recommended Format

The easiest format to use with this project is a CSV with these columns:

```csv
date,state,district,rainfall_mm
2026-04-06,Kerala,Idukki,182.0
2026-04-06,West Bengal,Darjeeling,66.0
```

Use [imd_export_template.csv](/Users/sanjaykumar/Documents/New project 2/data/imd/imd_export_template.csv) as the template.

## What To Save From IMD

If the IMD website gives you a district rainfall table:

1. Save the webpage as HTML or export the table to CSV.
2. Prefer district-wise rainfall tables over map images.
3. Make sure the saved content includes:
   - state name
   - district name
   - rainfall value
   - observation date if available

## Parser Rules

The parser accepts:

- HTML tables
- CSV files

It looks for these logical fields:

- `state`
- `district`
- `rainfall_mm`
- optional `date`

It also auto-maps common variants:

- `State Name` -> `state`
- `District Name` -> `district`
- `Rainfall (mm)` -> `rainfall_mm`
- `Actual Rainfall` -> `rainfall_mm`
- `Observation Date` -> `date`

## How To Use

If you export CSV directly:

1. place it somewhere accessible
2. set:

```text
LANDSLIDE_IMD_SOURCE_URL=/absolute/path/to/imd_export.csv
```

If you save the page as HTML:

1. place the saved HTML file locally
2. set:

```text
LANDSLIDE_IMD_SOURCE_URL=/absolute/path/to/imd_export.html
```

Then run:

```bash
PYTHONPATH=src python3 scripts/run_ingestion_jobs.py
```
