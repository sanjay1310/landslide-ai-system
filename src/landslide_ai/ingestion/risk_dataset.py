from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RISK_DATASET_COLUMNS = [
    "sample_id",
    "region_id",
    "state",
    "district",
    "date",
    "latitude",
    "longitude",
    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_14d_mm",
    "rainfall_ratio_24h_to_7d",
    "rainfall_acceleration",
    "soil_wetness_index",
    "temperature_c",
    "neighbor_risk_mean",
    "neighbor_count",
    "elevation_m",
    "slope_deg",
    "ndvi",
    "vegetation_vulnerability",
    "terrain_rainfall_interaction",
    "soil_rainfall_interaction",
    "hazard_score",
    "label",
    "label_source",
]


def build_large_scale_risk_dataset(
    rainfall_timeseries_csv_path: str | Path,
    region_metadata_csv_path: str | Path,
    terrain_csv_path: str | Path,
    output_csv_path: str | Path,
    landslide_inventory_csv_path: str | Path | None = None,
    inventory_event_window_days: int = 2,
) -> Path:
    rainfall = pd.read_csv(rainfall_timeseries_csv_path)
    metadata = pd.read_csv(region_metadata_csv_path)
    terrain = pd.read_csv(terrain_csv_path)

    metadata_subset = metadata[
        [
            "region_id",
            "state",
            "district",
            "latitude",
            "longitude",
            "temperature_c",
            "neighbor_risk_mean",
            "neighbor_count",
        ]
    ].copy()

    merged = rainfall.merge(
        metadata_subset,
        on="region_id",
        how="inner",
        suffixes=("_rain", "_meta"),
    )

    for column in ["state", "district", "latitude", "longitude", "temperature_c"]:
        rain_column = f"{column}_rain"
        meta_column = f"{column}_meta"
        if rain_column in merged.columns and meta_column in merged.columns:
            merged[column] = merged[rain_column].fillna(merged[meta_column])
            merged = merged.drop(columns=[rain_column, meta_column])
        elif rain_column in merged.columns:
            merged = merged.rename(columns={rain_column: column})
        elif meta_column in merged.columns:
            merged = merged.rename(columns={meta_column: column})

    merged = (
        merged.merge(terrain, on=["region_id", "state", "district"], how="inner")
        .sort_values(["region_id", "date"])
        .reset_index(drop=True)
    )

    merged["date"] = _parse_mixed_dates(merged["date"])
    merged["rainfall_24h_mm"] = merged["rainfall_mm"].astype(float)
    merged["rainfall_3d_mm"] = _rolling_sum(merged, 3)
    merged["rainfall_7d_mm"] = _rolling_sum(merged, 7)
    merged["rainfall_14d_mm"] = _rolling_sum(merged, 14)
    merged["rainfall_ratio_24h_to_7d"] = (
        merged["rainfall_24h_mm"] / merged["rainfall_7d_mm"].clip(lower=1.0)
    ).clip(0.0, 1.0)
    merged["rainfall_acceleration"] = (
        merged["rainfall_24h_mm"] - merged.groupby("region_id", dropna=False)["rainfall_24h_mm"].shift(1).fillna(0.0)
    )
    max_rainfall_14d = max(float(merged["rainfall_14d_mm"].max()), 1.0)
    merged["soil_wetness_index"] = (merged["rainfall_14d_mm"] / max_rainfall_14d).clip(0.0, 1.0)
    merged["vegetation_vulnerability"] = (
        (1.0 - merged["ndvi"].astype(float).clip(0.0, 1.0)) * merged["soil_wetness_index"]
    ).clip(0.0, 1.0)
    merged["terrain_rainfall_interaction"] = merged["slope_deg"].astype(float) * merged["rainfall_24h_mm"].astype(float)
    merged["soil_rainfall_interaction"] = merged["soil_wetness_index"] * merged["rainfall_7d_mm"].astype(float)

    merged["hazard_score"] = _build_hazard_score(merged)
    inventory_labels = _build_inventory_labels(
        merged,
        landslide_inventory_csv_path=landslide_inventory_csv_path,
        event_window_days=inventory_event_window_days,
    )
    if inventory_labels is not None:
        merged["label"] = inventory_labels
        merged["label_source"] = "inventory"
    else:
        low_threshold = float(merged["hazard_score"].quantile(0.60))
        extreme_threshold = float(merged["hazard_score"].quantile(0.78))
        merged["label"] = (
            (
                (merged["hazard_score"] >= extreme_threshold)
                | (
                    (merged["hazard_score"] >= low_threshold)
                    & (merged["rainfall_24h_mm"] >= float(merged["rainfall_24h_mm"].quantile(0.70)))
                )
            )
        ).astype(int)
        merged["label_source"] = "proxy"

    merged["sample_id"] = merged["region_id"].astype(str) + "__" + merged["date"].dt.strftime("%Y-%m-%d")
    output = merged[RISK_DATASET_COLUMNS].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def _rolling_sum(frame: pd.DataFrame, window: int) -> pd.Series:
    return (
        frame.groupby("region_id", dropna=False)["rainfall_mm"]
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype(float)
    )


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    as_text = series.astype(str).str.strip()
    parsed = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(as_text, errors="coerce")
    return parsed.fillna(fallback)


def _normalize(series: pd.Series) -> pd.Series:
    max_value = max(float(series.max()), 1.0)
    return (series.astype(float) / max_value).clip(0.0, 1.0)


def _build_hazard_score(frame: pd.DataFrame) -> pd.Series:
    rainfall_24h = _normalize(frame["rainfall_24h_mm"])
    rainfall_3d = _normalize(frame["rainfall_3d_mm"])
    rainfall_7d = _normalize(frame["rainfall_7d_mm"])
    slope = _normalize(frame["slope_deg"])
    elevation = _normalize(frame["elevation_m"])
    soil = frame["soil_wetness_index"].astype(float).clip(0.0, 1.0)
    vegetation = frame["vegetation_vulnerability"].astype(float).clip(0.0, 1.0)
    neighbor = frame["neighbor_risk_mean"].astype(float).clip(0.0, 1.0)
    acceleration = _normalize(frame["rainfall_acceleration"].clip(lower=0.0))
    return (
        rainfall_24h * 0.18
        + rainfall_3d * 0.18
        + rainfall_7d * 0.16
        + slope * 0.14
        + elevation * 0.05
        + soil * 0.11
        + vegetation * 0.08
        + neighbor * 0.05
        + acceleration * 0.05
    ).clip(0.0, 1.0)


def _build_inventory_labels(
    frame: pd.DataFrame,
    landslide_inventory_csv_path: str | Path | None,
    event_window_days: int,
) -> pd.Series | None:
    if landslide_inventory_csv_path is None:
        return None
    inventory_path = Path(landslide_inventory_csv_path)
    if not inventory_path.exists():
        return None

    inventory = pd.read_csv(inventory_path)
    if not {"state", "district"}.issubset(inventory.columns):
        return None

    date_column = next(
        (
            column
            for column in ["event_date", "date", "incident_date", "landslide_date"]
            if column in inventory.columns
        ),
        None,
    )
    if date_column is None:
        return None

    inventory = inventory.copy()
    inventory["event_date"] = pd.to_datetime(inventory[date_column], errors="coerce")
    inventory = inventory.dropna(subset=["event_date"]).copy()
    if inventory.empty:
        return None

    inventory["state_key"] = inventory["state"].astype(str).str.strip().str.lower()
    inventory["district_key"] = inventory["district"].astype(str).str.strip().str.lower()

    working = frame.copy()
    working["sample_date"] = pd.to_datetime(working["date"], errors="coerce")
    working["state_key"] = working["state"].astype(str).str.strip().str.lower()
    working["district_key"] = working["district"].astype(str).str.strip().str.lower()

    labels: list[int] = []
    for row in working[["state_key", "district_key", "sample_date"]].to_dict(orient="records"):
        matches = inventory[
            (inventory["state_key"] == row["state_key"])
            & (inventory["district_key"] == row["district_key"])
        ]
        if matches.empty or pd.isna(row["sample_date"]):
            labels.append(0)
            continue
        day_delta = (matches["event_date"] - pd.Timestamp(row["sample_date"])).abs().dt.days
        labels.append(int((day_delta <= event_window_days).any()))

    return pd.Series(labels, index=frame.index, dtype="int64")
