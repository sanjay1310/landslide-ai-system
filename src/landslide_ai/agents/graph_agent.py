import numpy as np

from landslide_ai.config import load_config
from landslide_ai.data.schemas import RegionRecord
from landslide_ai.graph.gnn import GraphNeuralRiskModel, torch_available
from landslide_ai.graph.spatial import build_spatial_graph, propagate_signal
from landslide_ai.models.schemas import GraphInsights


class GraphAgent:
    """Builds a simple spatial interaction graph between nearby regions."""

    def __init__(self) -> None:
        config = load_config(".")
        self.graph_model_artifact = config.graph_model_artifact
        self.learned_model = None
        if config.enable_graph_model and torch_available() and self.graph_model_artifact.exists():
            try:
                model = GraphNeuralRiskModel()
                model.load(self.graph_model_artifact)
                self.learned_model = model
            except Exception:
                self.learned_model = None

    def evaluate(self, record: RegionRecord) -> float:
        neighbor_factor = record.neighbor_risk_mean * 0.7
        connectivity_factor = min(record.neighbor_count / 8.0, 1.0) * 0.3
        return min(neighbor_factor + connectivity_factor, 1.0)

    def evaluate_network(self, records: list[RegionRecord], neighbor_limit: int = 3) -> dict[str, GraphInsights]:
        if not records:
            return {}

        heuristic_scores = {record.region_id: self.evaluate(record) for record in records}
        graph = build_spatial_graph(records)
        base_scores = heuristic_scores.copy()
        if self.learned_model is not None:
            try:
                learned_scores = self._predict_learned_scores(records, graph.normalized_adjacency)
                for region_id, score in learned_scores.items():
                    base_scores[region_id] = min(
                        (heuristic_scores[region_id] * 0.55) + (score * 0.45),
                        1.0,
                    )
            except Exception:
                pass

        base_vector = np.array([base_scores[record.region_id] for record in records], dtype=float)
        propagated_vector = propagate_signal(graph, base_vector, steps=2, alpha=0.65)
        graph_map: dict[str, GraphInsights] = {}
        for record in records:
            neighbors = self._nearest_neighbors(record, records, neighbor_limit)
            record_index = graph.region_ids.index(record.region_id)
            if not neighbors:
                graph_map[record.region_id] = GraphInsights(
                    local_influence=base_scores[record.region_id],
                    propagated_influence=float(propagated_vector[record_index]),
                    proximity_score=0.0,
                    neighbor_ids=[],
                )
                continue

            weighted_sum = 0.0
            weight_total = 0.0
            proximity_total = 0.0
            neighbor_ids: list[str] = []
            for neighbor, distance_km in neighbors:
                weight = 1.0 / max(distance_km, 1.0)
                weighted_sum += base_scores[neighbor.region_id] * weight
                weight_total += weight
                proximity_total += min(150.0 / max(distance_km, 1.0), 1.0)
                neighbor_ids.append(neighbor.region_id)

            graph_map[record.region_id] = GraphInsights(
                local_influence=base_scores[record.region_id],
                propagated_influence=float(min(propagated_vector[record_index], 1.0)),
                proximity_score=min(proximity_total / len(neighbors), 1.0),
                neighbor_ids=neighbor_ids,
            )

        return graph_map

    def _nearest_neighbors(
        self,
        source: RegionRecord,
        records: list[RegionRecord],
        limit: int,
    ) -> list[tuple[RegionRecord, float]]:
        distances: list[tuple[RegionRecord, float]] = []
        for candidate in records:
            if candidate.region_id == source.region_id:
                continue
            distances.append((candidate, self._distance_km(source, candidate)))
        return sorted(distances, key=lambda item: item[1])[:limit]

    @staticmethod
    def _distance_km(source: RegionRecord, target: RegionRecord) -> float:
        from landslide_ai.graph.spatial import _distance_km

        return _distance_km(source, target)

    def _predict_learned_scores(
        self,
        records: list[RegionRecord],
        adjacency: np.ndarray,
    ) -> dict[str, float]:
        assert self.learned_model is not None
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "region_id": record.region_id,
                    "rainfall_24h_mm": record.rainfall_24h_mm,
                    "rainfall_7d_mm": record.rainfall_7d_mm,
                    "slope_deg": record.slope_deg,
                    "elevation_m": record.elevation_m,
                    "soil_wetness_index": record.soil_wetness_index,
                    "ndvi": record.ndvi,
                    "temperature_c": record.temperature_c,
                    "neighbor_risk_mean": record.neighbor_risk_mean,
                    "neighbor_count": record.neighbor_count,
                }
                for record in records
            ]
        )
        probabilities = self.learned_model.predict_proba_from_frame(frame, adjacency)
        return {
            record.region_id: float(probability)
            for record, probability in zip(records, probabilities, strict=True)
        }
