from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from landslide_ai.config import load_config


@dataclass(frozen=True, slots=True)
class IndiaDatasetPaths:
    region_csv: Path
    district_geojson: Path
    state_geojson: Path


def default_india_dataset_paths(project_root: str | Path = ".") -> IndiaDatasetPaths:
    config = load_config(project_root)
    root = Path(project_root)
    return IndiaDatasetPaths(
        region_csv=config.default_region_csv,
        district_geojson=(
            config.official_district_geojson
            if config.official_district_geojson.exists()
            else root / "data" / "india" / "district_boundaries.geojson"
        ),
        state_geojson=config.official_state_geojson,
    )


def dataset_status(project_root: str | Path = ".") -> dict[str, bool]:
    paths = default_india_dataset_paths(project_root)
    return {
        "region_csv_exists": paths.region_csv.exists(),
        "district_geojson_exists": paths.district_geojson.exists(),
        "state_geojson_exists": paths.state_geojson.exists(),
    }
