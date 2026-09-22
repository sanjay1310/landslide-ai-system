from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from landslide_ai.config import AppConfig


@dataclass(slots=True)
class GISStatus:
    district_boundary_path: Path
    state_boundary_path: Path
    district_boundary_exists: bool
    state_boundary_exists: bool


def get_gis_status(config: AppConfig) -> GISStatus:
    return GISStatus(
        district_boundary_path=config.official_district_geojson,
        state_boundary_path=config.official_state_geojson,
        district_boundary_exists=config.official_district_geojson.exists(),
        state_boundary_exists=config.official_state_geojson.exists(),
    )
