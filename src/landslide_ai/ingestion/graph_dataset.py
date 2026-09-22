from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_balanced_india_graph_training_dataset(
    rainfall_timeseries_csv_path: str | Path,
    region_metadata_csv_path: str | Path,
    terrain_csv_path: str | Path,
    output_csv_path: str | Path,
) -> Path:
    rainfall = pd.read_csv(rainfall_timeseries_csv_path)
    metadata = pd.read_csv(region_metadata_csv_path)
    terrain = pd.read_csv(terrain_csv_path)

    metadata_columns = [
        "region_id",
        "state",
        "district",
        "latitude",
        "longitude",
        "temperature_c",
        "neighbor_risk_mean",
        "neighbor_count",
    ]
    merged = (
        rainfall.merge(metadata[metadata_columns], on="region_id", how="inner")
        .merge(terrain, on=["region_id", "state", "district"], how="inner")
        .sort_values(["region_id", "date"])
        .reset_index(drop=True)
    )

    merged["rainfall_24h_mm"] = merged["rainfall_mm"].astype(float)
    merged["rainfall_7d_mm"] = (
        merged.groupby("region_id", dropna=False)["rainfall_mm"]
        .rolling(window=7, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    max_rainfall_7d = max(float(merged["rainfall_7d_mm"].max()), 1.0)
    merged["soil_wetness_index"] = (merged["rainfall_7d_mm"] / max_rainfall_7d).clip(0.0, 1.0)

    rainfall_24h_norm = _normalize_series(merged["rainfall_24h_mm"])
    rainfall_7d_norm = _normalize_series(merged["rainfall_7d_mm"])
    slope_norm = _normalize_series(merged["slope_deg"])
    elevation_norm = _normalize_series(merged["elevation_m"])
    vegetation_stress = (1.0 - merged["ndvi"].clip(0.0, 1.0)).clip(0.0, 1.0)
    neighbor_norm = merged["neighbor_risk_mean"].clip(0.0, 1.0)
    intensity_ratio = (merged["rainfall_24h_mm"] / merged["rainfall_7d_mm"].clip(lower=1.0)).clip(0.0, 1.0)

    merged["hazard_score"] = (
        rainfall_24h_norm * 0.24
        + rainfall_7d_norm * 0.20
        + slope_norm * 0.18
        + elevation_norm * 0.06
        + merged["soil_wetness_index"] * 0.14
        + vegetation_stress * 0.07
        + neighbor_norm * 0.07
        + intensity_ratio * 0.04
    )
    low_threshold = float(merged["hazard_score"].quantile(0.45))
    high_threshold = float(merged["hazard_score"].quantile(0.60))
    merged["label"] = (
        (
            (merged["hazard_score"] >= high_threshold)
            | ((merged["hazard_score"] >= low_threshold) & (merged["rainfall_24h_mm"] >= merged["rainfall_24h_mm"].median()))
        )
    ).astype(int)

    merged["region_id"] = merged["region_id"].astype(str) + "__" + merged["date"].astype(str)
    output_columns = [
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
        "label",
        "date",
        "hazard_score",
    ]
    output = merged[output_columns].copy()

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def _normalize_series(series: pd.Series) -> pd.Series:
    max_value = float(series.max())
    if max_value <= 0.0:
        return pd.Series([0.0] * len(series), index=series.index, dtype="float64")
    return (series.astype(float) / max_value).clip(0.0, 1.0)
