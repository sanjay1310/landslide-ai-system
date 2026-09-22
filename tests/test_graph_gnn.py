from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from landslide_ai.graph.gnn import predict_graph_probabilities_for_csv
from landslide_ai.training.train_gnn import train_graph_risk_model
from landslide_ai.utils.optional_dependencies import torch_available


@unittest.skipUnless(torch_available(), "torch is not importable in this environment")
@unittest.skipUnless(os.getenv("LANDSLIDE_RUN_TORCH_TRAINING_TESTS") == "1", "torch training tests disabled")
class GraphGnnTrainingTest(unittest.TestCase):
    def test_train_graph_model_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "graph_gnn.pt"
            result = train_graph_risk_model(
                csv_path="data/sample_regions.csv",
                artifact_path=str(artifact_path),
                epochs=25,
                learning_rate=0.02,
            )
            self.assertTrue(artifact_path.exists())
            self.assertGreater(result.sample_count, 0)
            self.assertTrue(result.feature_columns)
            self.assertGreaterEqual(result.accuracy, 0.0)
            prediction_frame = predict_graph_probabilities_for_csv("data/sample_regions.csv", artifact_path)
            self.assertIn("graph_gnn_probability", prediction_frame.columns)
            self.assertEqual(len(prediction_frame), result.sample_count)


if __name__ == "__main__":
    unittest.main()
