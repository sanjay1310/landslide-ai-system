from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from landslide_ai.config import load_config
from landslide_ai.ingestion.forecast_windows import build_forecast_windows_from_daily_weather
from landslide_ai.ingestion.graph_dataset import build_balanced_india_graph_training_dataset
from landslide_ai.ingestion.real_sources import (
    ingest_imd_dataset,
    ingest_nasa_power_dataset,
    ingest_sentinel_catalog_snapshot,
)


class ConfigAndIngestionTest(unittest.TestCase):
    def test_load_config_returns_expected_paths(self) -> None:
        config = load_config(".")
        self.assertTrue(str(config.default_region_csv).endswith("data/india/india_regions.csv"))
        self.assertTrue(str(config.official_district_geojson).endswith("data/gis/india_districts_official.geojson"))

    def test_placeholder_ingestion_writes_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = load_config(root)
            artifact_imd = ingest_imd_dataset(config)
            artifact_nasa = ingest_nasa_power_dataset(config)
            artifact_sentinel = ingest_sentinel_catalog_snapshot(config)

            self.assertTrue(artifact_imd.output_path.exists())
            self.assertTrue(artifact_imd.manifest_path.exists())
            self.assertIn(artifact_imd.mode, {"placeholder", "fallback_placeholder"})
            self.assertTrue(artifact_nasa.manifest_path.exists())
            self.assertTrue(artifact_sentinel.manifest_path.exists())

    def test_forecast_windows_builder_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = build_forecast_windows_from_daily_weather(
                daily_weather_csv_path="data/raw/india_rainfall_timeseries.csv",
                output_csv_path=Path(tmp_dir) / "rainfall_windows.csv",
                window_size=3,
                forecast_horizon=1,
            )
            self.assertTrue(output_path.exists())

    def test_graph_training_builder_writes_balanced_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = build_balanced_india_graph_training_dataset(
                rainfall_timeseries_csv_path="data/raw/india_rainfall_timeseries.csv",
                region_metadata_csv_path="data/raw/india_rainfall.csv",
                terrain_csv_path="data/raw/india_terrain.csv",
                output_csv_path=Path(tmp_dir) / "india_graph_training.csv",
            )
            self.assertTrue(output_path.exists())
            frame = __import__("pandas").read_csv(output_path)
            self.assertGreater(len(frame), 6)
            self.assertIn("label", frame.columns)
            self.assertEqual(set(frame["label"].unique().tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
