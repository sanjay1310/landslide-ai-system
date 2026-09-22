from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from landslide_ai.research.evaluation import (
    build_research_frame,
    run_full_research_evaluation,
    write_research_outputs,
)


class ResearchEvaluationTest(unittest.TestCase):
    def test_build_research_frame_contains_expected_columns(self) -> None:
        frame = build_research_frame("data/sample_regions.csv")
        self.assertIn("rainfall_intensity", frame.columns)
        self.assertIn("propagated_graph_signal", frame.columns)
        self.assertIn("label", frame.columns)
        self.assertGreater(len(frame), 0)

    def test_research_evaluation_outputs_are_written(self) -> None:
        result = run_full_research_evaluation("data/sample_regions.csv")
        self.assertGreaterEqual(len(result.model_comparisons), 3)
        self.assertGreaterEqual(len(result.scenario_examples), 2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "research.json"
            md_path = Path(tmp_dir) / "research.md"
            write_research_outputs(result, json_path=json_path, markdown_path=md_path)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
