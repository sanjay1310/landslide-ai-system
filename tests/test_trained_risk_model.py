from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from landslide_ai.agents.graph_agent import GraphAgent
from landslide_ai.agents.vision_agent import VisionAgent
from landslide_ai.data.loader import load_region_records
from landslide_ai.models.baseline import BaselineRiskModel
from landslide_ai.models.trained_risk import TrainedRiskModel
from landslide_ai.pipeline.feature_engineering import build_features
from landslide_ai.training.train_baseline import train_random_forest_from_csv


class TrainedRiskModelTest(unittest.TestCase):
    def test_random_forest_training_writes_artifact_and_predicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "risk_model.pkl"
            result = train_random_forest_from_csv(
                "data/sample_regions.csv",
                model_output_path=str(artifact_path),
            )
            self.assertTrue(artifact_path.exists())
            self.assertGreater(result.sample_count, 0)
            self.assertGreaterEqual(result.validation_accuracy, 0.0)

            records = load_region_records("data/sample_regions.csv")
            record = records[0]
            graph_agent = GraphAgent()
            vision_agent = VisionAgent()
            graph_map = graph_agent.evaluate_network(records)
            terrain_change_signal, vegetation_signal = vision_agent.evaluate(record)
            graph_insights = graph_map[record.region_id]
            features = build_features(
                record=record,
                terrain_change_signal=terrain_change_signal,
                vegetation_signal=vegetation_signal,
                graph_signal=graph_insights.local_influence,
            )

            model = TrainedRiskModel()
            model.load(artifact_path)
            risk_score, forecast_risk, components = model.predict(
                features,
                propagated_graph_signal=graph_insights.propagated_influence,
                fallback_model=BaselineRiskModel(),
            )
            self.assertGreaterEqual(risk_score, 0.0)
            self.assertLessEqual(risk_score, 1.0)
            self.assertGreaterEqual(forecast_risk, 0.0)
            self.assertLessEqual(forecast_risk, 1.0)
            self.assertGreaterEqual(components.terrain_component, 0.0)


if __name__ == "__main__":
    unittest.main()
