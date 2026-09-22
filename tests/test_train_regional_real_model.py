from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from landslide_ai.training.train_regional_real_model import train_regional_real_runtime_model


class TrainRegionalRealModelTest(unittest.TestCase):
    def test_regional_runtime_training_writes_calibrated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "regional_real.csv"
            model_path = tmp_path / "risk_model.pkl"

            rows: list[dict[str, object]] = []
            for region_id, state, district, base_slope in [
                ("IN-AAA-ALPHA", "Kerala", "Alpha", 25.0),
                ("IN-BBB-BETA", "Uttarakhand", "Beta", 40.0),
            ]:
                for day in range(1, 13):
                    rows.append(
                        {
                            "sample_id": f"{region_id}__2026-01-{day:02d}",
                            "region_id": region_id,
                            "state": state,
                            "district": district,
                            "date": f"2026-01-{day:02d}",
                            "rainfall_24h_mm": float(day * 5),
                            "rainfall_3d_mm": float(day * 9),
                            "rainfall_7d_mm": float(day * 14),
                            "rainfall_14d_mm": float(day * 22),
                            "rainfall_ratio_24h_to_7d": min((day * 5) / max(day * 14, 1), 1.0),
                            "rainfall_acceleration": float(day),
                            "soil_wetness_index": min(0.1 + day * 0.05, 1.0),
                            "temperature_c": 22.0 if state == "Kerala" else 16.0,
                            "neighbor_risk_mean": 0.25 + day * 0.02,
                            "neighbor_count": 4,
                            "elevation_m": 1200.0 if state == "Uttarakhand" else 200.0,
                            "slope_deg": base_slope,
                            "ndvi": 0.65 - day * 0.01,
                            "vegetation_vulnerability": 0.15 + day * 0.01,
                            "terrain_rainfall_interaction": base_slope * float(day * 5),
                            "soil_rainfall_interaction": min(0.1 + day * 0.05, 1.0) * float(day * 14),
                            "hazard_score": min(0.1 + day * 0.04, 1.0),
                            "label": 1 if day in {8, 9, 10, 11} else 0,
                            "label_source": "inventory",
                        }
                    )

            pd.DataFrame(rows).to_csv(dataset_path, index=False)
            result = train_regional_real_runtime_model(dataset_csv_path=dataset_path, model_output_path=model_path)

            self.assertTrue(model_path.exists())
            self.assertTrue(result.metadata_path.exists())
            self.assertGreater(result.sample_count, 0)
            self.assertGreaterEqual(result.roc_auc, 0.0)
            self.assertGreaterEqual(result.pr_auc, 0.0)
            self.assertGreaterEqual(result.brier_score, 0.0)

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertIn("calibration_method", metadata)
            self.assertIn("decision_threshold", metadata)


if __name__ == "__main__":
    unittest.main()
