from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from landslide_ai.research.real_regions import TARGET_STATES, prepare_target_state_inputs


class RealRegionsPipelineTest(unittest.TestCase):
    def test_prepare_target_state_inputs_filters_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "data" / "india").mkdir(parents=True, exist_ok=True)
            (root / "data" / "raw").mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"region_id": "1", "state": "Kerala", "district": "Idukki", "latitude": 10.1, "longitude": 77.0, "rainfall_24h_mm": 10, "rainfall_7d_mm": 20, "soil_wetness_index": 0.2, "temperature_c": 24, "neighbor_risk_mean": 0.5, "neighbor_count": 4, "label": 1, "hazard_score": 0.7, "elevation_m": 1000, "slope_deg": 35, "ndvi": 0.4},
                    {"region_id": "2", "state": "Uttarakhand", "district": "Chamoli", "latitude": 30.4, "longitude": 79.3, "rainfall_24h_mm": 12, "rainfall_7d_mm": 30, "soil_wetness_index": 0.3, "temperature_c": 18, "neighbor_risk_mean": 0.6, "neighbor_count": 5, "label": 1, "hazard_score": 0.8, "elevation_m": 1800, "slope_deg": 40, "ndvi": 0.35},
                    {"region_id": "3", "state": "Goa", "district": "North Goa", "latitude": 15.5, "longitude": 73.8, "rainfall_24h_mm": 8, "rainfall_7d_mm": 14, "soil_wetness_index": 0.1, "temperature_c": 29, "neighbor_risk_mean": 0.2, "neighbor_count": 2, "label": 0, "hazard_score": 0.2, "elevation_m": 200, "slope_deg": 10, "ndvi": 0.6},
                ]
            ).to_csv(root / "data" / "india" / "india_regions.csv", index=False)
            pd.DataFrame(
                [
                    {"region_id": "1", "state": "Kerala", "district": "Idukki", "latitude": 10.1, "longitude": 77.0, "rainfall_24h_mm": 10, "rainfall_7d_mm": 20, "soil_wetness_index": 0.2, "temperature_c": 24, "neighbor_risk_mean": 0.5, "neighbor_count": 4, "label": 1},
                    {"region_id": "2", "state": "Uttarakhand", "district": "Chamoli", "latitude": 30.4, "longitude": 79.3, "rainfall_24h_mm": 12, "rainfall_7d_mm": 30, "soil_wetness_index": 0.3, "temperature_c": 18, "neighbor_risk_mean": 0.6, "neighbor_count": 5, "label": 1},
                    {"region_id": "3", "state": "Goa", "district": "North Goa", "latitude": 15.5, "longitude": 73.8, "rainfall_24h_mm": 8, "rainfall_7d_mm": 14, "soil_wetness_index": 0.1, "temperature_c": 29, "neighbor_risk_mean": 0.2, "neighbor_count": 2, "label": 0},
                ]
            ).to_csv(root / "data" / "raw" / "india_rainfall.csv", index=False)
            pd.DataFrame(
                [
                    {"region_id": "1", "state": "Kerala", "district": "Idukki", "elevation_m": 1000, "slope_deg": 35, "ndvi": 0.4},
                    {"region_id": "2", "state": "Uttarakhand", "district": "Chamoli", "elevation_m": 1800, "slope_deg": 40, "ndvi": 0.35},
                    {"region_id": "3", "state": "Goa", "district": "North Goa", "elevation_m": 200, "slope_deg": 10, "ndvi": 0.6},
                ]
            ).to_csv(root / "data" / "raw" / "india_terrain.csv", index=False)

            paths = prepare_target_state_inputs(root)
            region_frame = pd.read_csv(paths.region_csv)
            self.assertEqual(set(region_frame["state"].unique().tolist()), set(TARGET_STATES))


if __name__ == "__main__":
    unittest.main()
