from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from landslide_ai.research.large_scale import (
    augment_large_scale_features,
    evaluate_large_scale_risk_dataset,
    prepare_large_scale_dataset,
    temporal_split_by_region,
    write_large_scale_evaluation,
)


class LargeScalePipelineTest(unittest.TestCase):
    def test_prepare_large_scale_dataset_writes_expected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = prepare_large_scale_dataset(output_csv_path=Path(tmp_dir) / "risk_timeseries.csv")
            self.assertTrue(output.exists())
            frame = pd.read_csv(output)
            self.assertGreater(len(frame), 50)
            self.assertIn("sample_id", frame.columns)
            self.assertIn("rainfall_14d_mm", frame.columns)
            self.assertIn("label", frame.columns)
            self.assertIn("label_source", frame.columns)

    def test_temporal_evaluation_generates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "risk_timeseries.csv"
            prepare_large_scale_dataset(output_csv_path=dataset_path)
            result = evaluate_large_scale_risk_dataset(dataset_csv_path=str(dataset_path))
            self.assertGreater(result.dataset_rows, 50)
            self.assertGreater(result.validation_rows, 0)
            self.assertGreaterEqual(result.metrics.accuracy, 0.0)
            self.assertGreaterEqual(result.metrics.brier_score, 0.0)
            json_path = Path(tmp_dir) / "eval.json"
            markdown_path = Path(tmp_dir) / "eval.md"
            write_large_scale_evaluation(result, json_path=json_path, markdown_path=markdown_path)
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())

    def test_inventory_labels_override_proxy_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            inventory_path = Path(tmp_dir) / "inventory.csv"
            inventory_path.write_text(
                "state,district,event_date\nKerala,Alappuzha,2026-04-10\n",
                encoding="utf-8",
            )
            dataset_path = Path(tmp_dir) / "risk_timeseries.csv"
            prepare_large_scale_dataset(
                output_csv_path=dataset_path,
                landslide_inventory_csv_path=str(inventory_path),
            )
            frame = pd.read_csv(dataset_path)
            self.assertEqual(set(frame["label_source"].unique().tolist()), {"inventory"})
            matched = frame[
                (frame["state"] == "Kerala")
                & (frame["district"] == "Alappuzha")
                & (frame["date"] == "2026-04-10")
            ]
            self.assertEqual(int(matched["label"].iloc[0]), 1)

    def test_temporal_split_keeps_validation_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "region_id": ["A", "A", "A", "B", "B", "B"],
                "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02", "2026-01-03"],
            }
        )
        train_frame, validation_frame = temporal_split_by_region(frame, validation_fraction=0.33)
        self.assertGreater(len(train_frame), 0)
        self.assertGreater(len(validation_frame), 0)
        self.assertEqual(len(train_frame) + len(validation_frame), len(frame))

    def test_augment_large_scale_features_adds_expected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "rainfall_24h_mm": [10.0],
                "rainfall_3d_mm": [24.0],
                "rainfall_7d_mm": [40.0],
                "rainfall_14d_mm": [80.0],
                "rainfall_acceleration": [4.0],
                "soil_wetness_index": [0.5],
                "temperature_c": [22.0],
                "neighbor_risk_mean": [0.4],
                "neighbor_count": [4],
                "elevation_m": [1000.0],
                "slope_deg": [30.0],
                "ndvi": [0.6],
            }
        )
        augmented = augment_large_scale_features(frame)
        self.assertIn("rainfall_3d_to_14d_ratio", augmented.columns)
        self.assertIn("neighbor_pressure", augmented.columns)
        self.assertIn("ndvi_loss", augmented.columns)


if __name__ == "__main__":
    unittest.main()
