from __future__ import annotations

import unittest

from landslide_ai.services.system_service import LandslideSystemService


class LandslideSystemServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LandslideSystemService(".")
        self.csv_path = "data/india/india_regions.csv"
        self.windows_path = "data/india/rainfall_windows.csv"

    def test_assess_csv_returns_ranked_results(self) -> None:
        results = self.service.assess_csv(self.csv_path)
        self.assertGreater(len(results), 0)
        self.assertGreaterEqual(results[0].risk_score, results[-1].risk_score)
        self.assertTrue(results[0].primary_driver)

    def test_scenario_increases_target_state_risk(self) -> None:
        base_results = self.service.assess_csv(self.csv_path)
        base_kerala = next(result for result in base_results if result.state == "Kerala")

        scenario_results = self.service.assess_scenario(
            self.csv_path,
            rainfall_multiplier=1.25,
            target_state="Kerala",
        )
        scenario_kerala = next(result for result in scenario_results if result.state == "Kerala")
        self.assertGreaterEqual(scenario_kerala.risk_score, base_kerala.risk_score)

    def test_forecast_from_windows_returns_predictions(self) -> None:
        forecast_frame = self.service.forecast_from_windows(self.windows_path)
        if forecast_frame.empty:
            self.assertEqual(len(forecast_frame.columns), 0)
        else:
            self.assertIn("predicted_rainfall", forecast_frame.columns)
            self.assertIn("absolute_error", forecast_frame.columns)


if __name__ == "__main__":
    unittest.main()
