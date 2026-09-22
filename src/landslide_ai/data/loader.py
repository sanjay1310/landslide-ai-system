from __future__ import annotations

from pathlib import Path

import pandas as pd

from landslide_ai.data.schemas import RegionRecord


REQUIRED_COLUMNS = {
    "region_id",
    "state",
    "district",
    "latitude",
    "longitude",
    "rainfall_24h_mm",
    "rainfall_7d_mm",
    "slope_deg",
    "elevation_m",
    "soil_wetness_index",
    "ndvi",
    "temperature_c",
    "neighbor_risk_mean",
    "neighbor_count",
}


def load_region_records(csv_path: str | Path) -> list[RegionRecord]:
    dataframe = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    records: list[RegionRecord] = []
    for row in dataframe.to_dict(orient="records"):
        records.append(
            RegionRecord(
                region_id=str(row["region_id"]),
                state=str(row["state"]),
                district=str(row["district"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                rainfall_24h_mm=float(row["rainfall_24h_mm"]),
                rainfall_7d_mm=float(row["rainfall_7d_mm"]),
                slope_deg=float(row["slope_deg"]),
                elevation_m=float(row["elevation_m"]),
                soil_wetness_index=float(row["soil_wetness_index"]),
                ndvi=float(row["ndvi"]),
                temperature_c=float(row["temperature_c"]),
                neighbor_risk_mean=float(row["neighbor_risk_mean"]),
                neighbor_count=int(row["neighbor_count"]),
            )
        )
    return records


def load_training_frame(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.union({"label"}).difference(dataframe.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")
    return dataframe
