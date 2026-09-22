from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import time

import pandas as pd


DEFAULT_DEM_API = "https://api.opentopodata.org/v1/srtm90m"


@dataclass(slots=True)
class DemFetchResult:
    output_path: Path
    row_count: int
    dataset: str


def fetch_dem_elevation_for_regions(
    region_csv_path: str | Path,
    output_csv_path: str | Path,
    dataset: str = "srtm90m",
    batch_size: int = 20,
) -> DemFetchResult:
    frame = pd.read_csv(region_csv_path)
    rows: list[dict[str, object]] = []
    records = frame.to_dict(orient="records")

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        elevations = fetch_elevation_batch(
            [
                (float(item["latitude"]), float(item["longitude"]))
                for item in batch
            ],
            dataset=dataset,
        )
        for row, elevation in zip(batch, elevations, strict=True):
            rows.append(
                {
                    "region_id": row["region_id"],
                    "state": row["state"],
                    "district": row["district"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "dem_elevation_m": elevation,
                    "dem_dataset": dataset,
                }
            )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return DemFetchResult(
        output_path=output_path,
        row_count=len(rows),
        dataset=dataset,
    )


def fetch_dem_terrain_for_regions(
    region_csv_path: str | Path,
    output_csv_path: str | Path,
    dataset: str = "srtm90m",
    sample_spacing_m: float = 90.0,
    batch_size: int = 25,
) -> DemFetchResult:
    frame = pd.read_csv(region_csv_path)
    rows: list[dict[str, object]] = []

    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size].copy()
        query_points: list[tuple[float, float]] = []
        point_map: list[tuple[str, str]] = []
        for row in batch.to_dict(orient="records"):
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            lat_step = sample_spacing_m / 111_320.0
            lon_step = sample_spacing_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))
            samples = {
                "center": (lat, lon),
                "north": (lat + lat_step, lon),
                "south": (lat - lat_step, lon),
                "east": (lat, lon + lon_step),
                "west": (lat, lon - lon_step),
            }
            for key, coords in samples.items():
                point_map.append((str(row["region_id"]), key))
                query_points.append(coords)

        elevations = fetch_elevation_batch(query_points, dataset=dataset)
        grouped: dict[str, dict[str, float | None]] = {}
        for (region_id, key), elevation in zip(point_map, elevations, strict=True):
            grouped.setdefault(region_id, {})[key] = elevation

        for row in batch.to_dict(orient="records"):
            region_samples = grouped[str(row["region_id"])]
            slope = _slope_from_dem_samples(region_samples, sample_spacing_m)
            rows.append(
                {
                    "region_id": row["region_id"],
                    "state": row["state"],
                    "district": row["district"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "dem_elevation_m": region_samples.get("center"),
                    "dem_slope_proxy_deg": slope,
                    "dem_dataset": dataset,
                    "dem_sample_spacing_m": sample_spacing_m,
                }
            )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return DemFetchResult(
        output_path=output_path,
        row_count=len(rows),
        dataset=dataset,
    )


def fetch_elevation(latitude: float, longitude: float, dataset: str = "srtm90m") -> float | None:
    return fetch_elevation_batch([(latitude, longitude)], dataset=dataset)[0]


def fetch_elevation_batch(
    locations: list[tuple[float, float]],
    dataset: str = "srtm90m",
    max_attempts: int = 4,
) -> list[float | None]:
    if not locations:
        return []

    query = urlencode(
        {
            "locations": "|".join(f"{latitude},{longitude}" for latitude, longitude in locations)
        }
    )
    url = f"https://api.opentopodata.org/v1/{dataset}?{query}"
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            results = payload.get("results") or []
            elevations: list[float | None] = []
            for item in results:
                elevation = item.get("elevation")
                elevations.append(None if elevation is None else float(elevation))
            if len(elevations) != len(locations):
                raise ValueError("DEM API returned an unexpected number of results.")
            return elevations
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(1.5 * attempt)

    return [None for _ in locations]


def _slope_from_dem_samples(samples: dict[str, float | None], spacing_m: float) -> float | None:
    north = samples.get("north")
    south = samples.get("south")
    east = samples.get("east")
    west = samples.get("west")
    if None in {north, south, east, west}:
        return None
    dz_dy = (float(north) - float(south)) / (2.0 * spacing_m)
    dz_dx = (float(east) - float(west)) / (2.0 * spacing_m)
    slope_rise_run = math.sqrt((dz_dx ** 2) + (dz_dy ** 2))
    return math.degrees(math.atan(slope_rise_run))
