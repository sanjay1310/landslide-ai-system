from __future__ import annotations

import unittest

import numpy as np

from landslide_ai.data.loader import load_region_records
from landslide_ai.graph.spatial import build_spatial_graph, propagate_signal


class GraphPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_region_records("data/india/india_regions.csv")

    def test_spatial_graph_shapes_match_record_count(self) -> None:
        graph = build_spatial_graph(self.records)
        count = len(self.records)
        self.assertEqual(graph.adjacency.shape, (count, count))
        self.assertEqual(graph.normalized_adjacency.shape, (count, count))

    def test_propagate_signal_preserves_vector_length(self) -> None:
        graph = build_spatial_graph(self.records)
        base_signal = np.linspace(0.2, 0.8, len(self.records))
        propagated = propagate_signal(graph, base_signal, steps=2, alpha=0.65)
        self.assertEqual(len(propagated), len(base_signal))
        self.assertTrue((propagated >= 0.0).all())
        self.assertTrue((propagated <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
