from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from landslide_ai.ingestion.inventory import (
    create_inventory_template,
    merge_mapped_inventory_frames,
    normalize_inventory_name,
    prepare_nasa_global_landslide_catalog_frame,
    standardize_inventory_frame,
    write_standardized_inventory,
)


class InventoryToolsTest(unittest.TestCase):
    def test_create_inventory_template_writes_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = create_inventory_template(Path(tmp_dir) / "template.csv")
            frame = pd.read_csv(path)
            self.assertIn("state", frame.columns)
            self.assertIn("district", frame.columns)
            self.assertIn("event_date", frame.columns)

    def test_standardize_inventory_frame_normalizes_names(self) -> None:
        raw = pd.DataFrame(
            {
                "State": ["Uttaranchal"],
                "District": ["Uttar Kashi District"],
                "Date": ["2026-04-10"],
                "Cause": ["Rainfall"],
            }
        )
        standardized = standardize_inventory_frame(raw, source_name="test_source")
        self.assertEqual(standardized.loc[0, "state"], "uttarakhand")
        self.assertEqual(standardized.loc[0, "district"], "uttarkashi")
        self.assertEqual(standardized.loc[0, "event_date"], "2026-04-10")

    def test_write_standardized_inventory_outputs_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "raw.csv"
            output_path = Path(tmp_dir) / "clean.csv"
            pd.DataFrame(
                {
                    "state": ["Kerala"],
                    "district": ["Idukki"],
                    "event_date": ["2018-08-16"],
                }
            ).to_csv(input_path, index=False)
            write_standardized_inventory(input_path, output_path, source_name="nrsc")
            self.assertTrue(output_path.exists())

    def test_prepare_nasa_global_landslide_catalog_frame_filters_target_states(self) -> None:
        raw = pd.DataFrame(
            {
                "event_id": [1, 2],
                "event_date": ["2013-06-16", "2013-06-17"],
                "admin_division_name": ["Uttarakhand", "Nepal"],
                "gazeteer_closest_point": ["Uttarkashi", "Kathmandu"],
                "location_description": ["Uttarkashi road", "Kathmandu valley"],
                "latitude": [30.7, 27.7],
                "longitude": [78.4, 85.3],
                "landslide_trigger": ["rain", "rain"],
                "event_title": ["A", "B"],
                "location_accuracy": ["5km", "5km"],
                "source_name": ["NASA", "NASA"],
                "source_link": ["https://example.com/a", "https://example.com/b"],
            }
        )

        prepared = prepare_nasa_global_landslide_catalog_frame(raw, allowed_states=("Kerala", "Uttarakhand"))
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.loc[0, "state"], "Uttarakhand")
        self.assertEqual(prepared.loc[0, "district"], "Uttarkashi")
        self.assertIn("NASA Global Landslide Catalog import", prepared.loc[0, "notes"])

    def test_merge_mapped_inventory_frames_collapses_duplicate_district_dates(self) -> None:
        existing = pd.DataFrame(
            {
                "state": ["Uttarakhand"],
                "district": ["Chamoli"],
                "event_date": ["2013-06-16"],
                "source": ["provided_inventory"],
                "event_id": ["base-1"],
                "trigger": ["rain"],
                "notes": ["provided note"],
            }
        )
        nasa = pd.DataFrame(
            {
                "state": ["Uttarakhand", "Kerala"],
                "district": ["Chamoli", "Idukki"],
                "event_date": ["2013-06-16", "2013-08-05"],
                "source": ["nasa_glc", "nasa_glc"],
                "event_id": ["glc-1", "glc-2"],
                "trigger": ["downpour", "rain"],
                "notes": ["nasa note", "second nasa note"],
            }
        )

        merged = merge_mapped_inventory_frames([existing, nasa])
        self.assertEqual(len(merged), 2)
        chamoli = merged[merged["district"] == "Chamoli"].iloc[0]
        self.assertEqual(chamoli["source"], "multi_source")
        self.assertIn("base-1", chamoli["event_id"])
        self.assertIn("glc-1", chamoli["event_id"])


if __name__ == "__main__":
    unittest.main()
