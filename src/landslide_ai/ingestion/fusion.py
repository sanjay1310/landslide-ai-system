from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_region_dataset_from_weather_and_terrain(
    weather_csv_path: str | Path,
    terrain_csv_path: str | Path,
    output_csv_path: str | Path,
    imd_csv_path: str | Path | None = None,
) -> Path:
    weather = pd.read_csv(weather_csv_path)
    terrain = pd.read_csv(terrain_csv_path)
    if imd_csv_path:
        weather = apply_imd_overlay(weather, imd_csv_path)
    merged = weather.merge(terrain, on=["region_id", "state", "district"], how="inner")

    if "soil_wetness_index" not in merged.columns:
        merged["soil_wetness_index"] = (merged["rainfall_7d_mm"] / merged["rainfall_7d_mm"].max()).clip(0.0, 1.0)
    if "neighbor_risk_mean" not in merged.columns:
        merged["neighbor_risk_mean"] = 0.5
    if "neighbor_count" not in merged.columns:
        merged["neighbor_count"] = 4

    merged["hazard_score"] = _build_compound_hazard_score(merged)
    if "label" not in merged.columns:
        high_threshold = float(merged["hazard_score"].quantile(0.60))
        merged["label"] = (merged["hazard_score"] >= high_threshold).astype(int)
    else:
        proxy_label = (merged["hazard_score"] >= float(merged["hazard_score"].quantile(0.60))).astype(int)
        merged["label"] = ((merged["label"].astype(int) + proxy_label) >= 1).astype(int)

    columns = [
        "region_id",
        "state",
        "district",
        "latitude",
        "longitude",
        "rainfall_24h_mm",
        "rainfall_7d_mm",
        "soil_wetness_index",
        "temperature_c",
        "neighbor_risk_mean",
        "neighbor_count",
        "label",
        "hazard_score",
        "elevation_m",
        "slope_deg",
        "ndvi",
    ]
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[columns].to_csv(output_path, index=False)
    return output_path


def _build_compound_hazard_score(frame: pd.DataFrame) -> pd.Series:
    rainfall_24h = _normalize(frame["rainfall_24h_mm"])
    rainfall_7d = _normalize(frame["rainfall_7d_mm"])
    slope = _normalize(frame["slope_deg"])
    elevation = _normalize(frame["elevation_m"])
    soil = frame["soil_wetness_index"].astype(float).clip(0.0, 1.0)
    vegetation_stress = (1.0 - frame["ndvi"].astype(float).clip(0.0, 1.0)).clip(0.0, 1.0)
    neighbor = frame["neighbor_risk_mean"].astype(float).clip(0.0, 1.0)
    return (
        rainfall_24h * 0.27
        + rainfall_7d * 0.20
        + slope * 0.18
        + elevation * 0.05
        + soil * 0.14
        + vegetation_stress * 0.07
        + neighbor * 0.09
    ).clip(0.0, 1.0)


def _normalize(series: pd.Series) -> pd.Series:
    max_value = max(float(series.max()), 1.0)
    return (series.astype(float) / max_value).clip(0.0, 1.0)


def apply_imd_overlay(weather: pd.DataFrame, imd_csv_path: str | Path) -> pd.DataFrame:
    imd_path = Path(imd_csv_path)
    if not imd_path.exists():
        return weather

    imd_frame = pd.read_csv(imd_path)
    if imd_frame.empty or not {"state", "district", "rainfall_mm"}.issubset(imd_frame.columns):
        return weather

    latest_imd = (
        imd_frame.copy()
        .assign(
            state=lambda frame: frame["state"].astype(str).str.strip().str.lower(),
            district=lambda frame: frame["district"].astype(str).str.strip().str.lower(),
            rainfall_mm=lambda frame: pd.to_numeric(frame["rainfall_mm"], errors="coerce").fillna(0.0),
        )
        .groupby(["state", "district"], dropna=False)["rainfall_mm"]
        .max()
        .reset_index()
        .rename(columns={"rainfall_mm": "imd_rainfall_mm"})
    )

    overlaid = weather.copy()
    overlaid["state_key"] = overlaid["state"].astype(str).str.strip().str.lower()
    overlaid["district_key"] = overlaid["district"].astype(str).str.strip().str.lower()
    overlaid = overlaid.merge(
        latest_imd,
        left_on=["state_key", "district_key"],
        right_on=["state", "district"],
        how="left",
        suffixes=("", "_imd"),
    )
    overlaid["imd_rainfall_mm"] = overlaid["imd_rainfall_mm"].fillna(0.0)
    overlaid["rainfall_24h_mm"] = overlaid[["rainfall_24h_mm", "imd_rainfall_mm"]].max(axis=1)
    overlaid["source_mode"] = overlaid.apply(_combine_source_mode, axis=1)
    return overlaid.drop(columns=["state_key", "district_key", "state_imd", "district_imd"], errors="ignore")


def _combine_source_mode(row: pd.Series) -> str:
    base_mode = str(row.get("source_mode", "seeded"))
    imd_rainfall = float(row.get("imd_rainfall_mm", 0.0) or 0.0)
    if imd_rainfall > 0.0:
        return f"{base_mode}+imd_overlay"
    return base_mode
