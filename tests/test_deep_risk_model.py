from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from landslide_ai.utils.optional_dependencies import torch_available

@unittest.skipUnless(torch_available(), "torch is not importable in this environment")
class DeepRiskModelTest(unittest.TestCase):
    def test_deep_risk_training_writes_artifact(self) -> None:
        from landslide_ai.training.train_deep_risk import train_deep_risk_model_from_csv

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact = Path(tmp_dir) / "deep_risk_model.pt"
            result = train_deep_risk_model_from_csv(
                "data/sample_regions.csv",
                model_output_path=str(artifact),
                epochs=5,
            )
            self.assertTrue(artifact.exists())
            self.assertGreater(result.sample_count, 0)
            self.assertGreaterEqual(result.validation_accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
