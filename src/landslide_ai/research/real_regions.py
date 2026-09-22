from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from landslide_ai.config import load_config
from landslide_ai.ingestion.nasa_power import ingest_nasa_power_by_regions
from landslide_ai.research.large_scale import evaluate_large_scale_risk_dataset, prepare_large_scale_dataset


TARGET_STATES = ("Kerala", "Uttarakhand")


@dataclass(slots=True)
class RegionalRealDataPaths:
    region_csv: Path
    terrain_csv: Path
    rainfall_metadata_csv: Path
    nasa_daily_csv: Path
    nasa_latest_csv: Path
    risk_dataset_csv: Path


def default_regional_paths(project_root: str | Path = ".") -> RegionalRealDataPaths:
    root = Path(project_root)
    return RegionalRealDataPaths(
        region_csv=root / "data" / "regional" / "kerala_uttarakhand_regions.csv",
        terrain_csv=root / "data" / "regional" / "kerala_uttarakhand_terrain.csv",
        rainfall_metadata_csv=root / "data" / "regional" / "kerala_uttarakhand_rainfall_metadata.csv",
        nasa_daily_csv=root / "data" / "regional" / "nasa_power_daily_kerala_uttarakhand.csv",
        nasa_latest_csv=root / "data" / "regional" / "nasa_power_latest_kerala_uttarakhand.csv",
        risk_dataset_csv=root / "data" / "regional" / "kerala_uttarakhand_risk_timeseries.csv",
    )


def prepare_target_state_inputs(
    project_root: str | Path = ".",
    states: tuple[str, ...] = TARGET_STATES,
) -> RegionalRealDataPaths:
    root = Path(project_root)
    paths = default_regional_paths(root)
    paths.region_csv.parent.mkdir(parents=True, exist_ok=True)

    regions = pd.read_csv(root / "data" / "india" / "india_regions.csv")
    rainfall_metadata = pd.read_csv(root / "data" / "raw" / "india_rainfall.csv")
    terrain = pd.read_csv(root / "data" / "raw" / "india_terrain.csv")

    region_subset = regions[regions["state"].isin(states)].copy().reset_index(drop=True)
    rainfall_subset = rainfall_metadata[rainfall_metadata["state"].isin(states)].copy().reset_index(drop=True)
    terrain_subset = terrain[terrain["state"].isin(states)].copy().reset_index(drop=True)

    region_subset.to_csv(paths.region_csv, index=False)
    rainfall_subset.to_csv(paths.rainfall_metadata_csv, index=False)
    terrain_subset.to_csv(paths.terrain_csv, index=False)
    return paths


def fetch_nasa_power_for_target_states(
    project_root: str | Path = ".",
    states: tuple[str, ...] = TARGET_STATES,
) -> RegionalRealDataPaths:
    root = Path(project_root)
    paths = prepare_target_state_inputs(root, states=states)
    config = load_config(root)
    result = ingest_nasa_power_by_regions(config, paths.region_csv)
    Path(result.daily_output_path).replace(paths.nasa_daily_csv)
    Path(result.latest_output_path).replace(paths.nasa_latest_csv)
    return paths


def build_target_state_real_dataset(
    project_root: str | Path = ".",
    landslide_inventory_csv_path: str | None = None,
    states: tuple[str, ...] = TARGET_STATES,
) -> RegionalRealDataPaths:
    root = Path(project_root)
    paths = fetch_nasa_power_for_target_states(root, states=states)
    prepare_large_scale_dataset(
        rainfall_timeseries_csv_path=str(paths.nasa_daily_csv),
        region_metadata_csv_path=str(paths.rainfall_metadata_csv),
        terrain_csv_path=str(paths.terrain_csv),
        output_csv_path=str(paths.risk_dataset_csv),
        landslide_inventory_csv_path=landslide_inventory_csv_path,
    )
    return paths


def run_target_state_real_evaluation(
    project_root: str | Path = ".",
    landslide_inventory_csv_path: str | None = None,
    states: tuple[str, ...] = TARGET_STATES,
):
    paths = build_target_state_real_dataset(
        project_root=project_root,
        landslide_inventory_csv_path=landslide_inventory_csv_path,
        states=states,
    )
    result = evaluate_large_scale_risk_dataset(dataset_csv_path=str(paths.risk_dataset_csv))
    return paths, result
