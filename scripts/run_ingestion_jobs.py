from landslide_ai.config import load_config
from landslide_ai.ingestion.fusion import build_region_dataset_from_weather_and_terrain
from landslide_ai.ingestion.imd import ingest_imd_compatible_dataset
from landslide_ai.ingestion.forecast_windows import build_forecast_windows_from_daily_weather
from landslide_ai.ingestion.graph_dataset import build_balanced_india_graph_training_dataset
from landslide_ai.ingestion.nasa_power import ingest_nasa_power_by_regions
from landslide_ai.ingestion.real_sources import (
    ingest_sentinel_catalog_snapshot,
)
import pandas as pd


def main() -> None:
    config = load_config(".")
    imd_result = ingest_imd_compatible_dataset(config)
    artifacts = [ingest_sentinel_catalog_snapshot(config)]
    nasa_result = ingest_nasa_power_by_regions(config, config.default_region_csv)
    fused_output = build_region_dataset_from_weather_and_terrain(
        weather_csv_path=nasa_result.latest_output_path,
        terrain_csv_path="data/raw/india_terrain.csv",
        output_csv_path=config.default_region_csv,
        imd_csv_path=imd_result.normalized_output_path,
    )
    forecast_source_path = _choose_forecast_source(nasa_result.daily_output_path)
    forecast_windows_output = build_forecast_windows_from_daily_weather(
        daily_weather_csv_path=forecast_source_path,
        output_csv_path=config.default_rainfall_windows_csv,
    )
    graph_training_output = build_balanced_india_graph_training_dataset(
        rainfall_timeseries_csv_path="data/raw/india_rainfall_timeseries.csv",
        region_metadata_csv_path="data/raw/india_rainfall.csv",
        terrain_csv_path="data/raw/india_terrain.csv",
        output_csv_path="data/india/india_graph_training.csv",
    )
    print(f"imd: {imd_result.mode}")
    print(f"  raw output: {imd_result.raw_output_path}")
    print(f"  normalized output: {imd_result.normalized_output_path}")
    print(f"  rows: {imd_result.row_count}")
    for artifact in artifacts:
        print(f"{artifact.source_name}: {artifact.mode}")
        print(f"  output: {artifact.output_path}")
        print(f"  manifest: {artifact.manifest_path}")
    print(f"nasa_power: {nasa_result.mode}")
    print(f"  daily output: {nasa_result.daily_output_path}")
    print(f"  latest output: {nasa_result.latest_output_path}")
    print(f"  rows: {nasa_result.row_count}")
    print(f"forecast source: {forecast_source_path}")
    print(f"fused region dataset: {fused_output}")
    print(f"forecast windows dataset: {forecast_windows_output}")
    print(f"graph training dataset: {graph_training_output}")


def _choose_forecast_source(nasa_daily_path: str) -> str:
    fallback_path = "data/raw/india_rainfall_timeseries.csv"
    try:
        daily_frame = pd.read_csv(nasa_daily_path)
        if daily_frame.empty:
            return fallback_path
        date_counts = daily_frame.groupby("region_id", dropna=False)["date"].nunique()
        if date_counts.empty or int(date_counts.min()) < 4:
            return fallback_path
        return nasa_daily_path
    except Exception:
        return fallback_path


if __name__ == "__main__":
    main()
