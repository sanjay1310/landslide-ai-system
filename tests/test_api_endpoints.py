from __future__ import annotations

import importlib.util
import unittest

import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("fastapi") is not None, "fastapi not installed")
class ApiEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from landslide_ai.api.app import create_app, service_dependency
        from landslide_ai.models.schemas import RiskAssessment

        class StubService:
            def assess_csv(self, csv_path: str):
                return [
                    RiskAssessment(
                        region_id="R1",
                        state="Kerala",
                        district="Idukki",
                        latitude=10.0,
                        longitude=76.9,
                        risk_score=0.91,
                        forecast_risk=0.88,
                        risk_level="High",
                        primary_driver="Terrain",
                        uncertainty_score=0.08,
                        graph_exposure=0.42,
                        propagated_graph_exposure=0.48,
                    rainfall_exposure=0.75,
                    terrain_exposure=0.82,
                    vegetation_exposure=0.33,
                    calibrated_risk_score=0.86,
                    confidence_level="High",
                    alert_band="Critical",
                    recommended_action="Escalate district monitoring, verify slope conditions, and prepare field advisories.",
                    advisory="Monitor slopes closely.",
                ),
                    RiskAssessment(
                        region_id="R2",
                        state="Kerala",
                        district="Wayanad",
                        latitude=11.6,
                        longitude=76.1,
                        risk_score=0.51,
                        forecast_risk=0.57,
                        risk_level="Moderate",
                        primary_driver="Rainfall",
                        uncertainty_score=0.12,
                        graph_exposure=0.35,
                        propagated_graph_exposure=0.39,
                    rainfall_exposure=0.68,
                    terrain_exposure=0.44,
                    vegetation_exposure=0.29,
                    calibrated_risk_score=0.49,
                    confidence_level="Medium",
                    alert_band="Watch",
                    recommended_action="Increase rainfall watch frequency and inspect districts with recent terrain or soil stress.",
                    advisory="Maintain monitoring.",
                ),
                ]

            def assess_scenario(self, **_: object):
                return self.assess_csv("unused")

            def forecast_from_windows(self, windows_csv_path: str) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {"region_id": "R1", "reference_date": "2026-01-01", "predicted_rainfall": 12.4, "absolute_error": 1.2},
                        {"region_id": "R2", "reference_date": "2026-01-01", "predicted_rainfall": 7.1, "absolute_error": 0.8},
                    ]
                )

            def analytics_summary(self, csv_path: str) -> dict[str, object]:
                return {
                    "region_count": 2,
                    "high_risk_region_count": 1,
                    "alert_region_count": 1,
                    "mean_risk_score": 0.71,
                    "top_region": {
                        "region_id": "R1",
                        "state": "Kerala",
                        "district": "Idukki",
                        "risk_score": 0.91,
                        "forecast_risk": 0.88,
                        "primary_driver": "Terrain",
                    },
                    "state_summary": [
                        {"state": "Kerala", "risk_score": 0.71, "forecast_risk": 0.72, "uncertainty_score": 0.10}
                    ],
                    "driver_summary": [
                        {"primary_driver": "Terrain", "region_count": 1},
                        {"primary_driver": "Rainfall", "region_count": 1},
                    ],
                }

            def graph_inference_from_csv(self, csv_path: str, artifact_path: str | None = None) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "region_id": "R1__2026-01-01",
                            "state": "Kerala",
                            "district": "Idukki",
                            "label": 1,
                            "graph_gnn_probability": 0.72,
                        },
                        {
                            "region_id": "R2__2026-01-01",
                            "state": "Kerala",
                            "district": "Wayanad",
                            "label": 0,
                            "graph_gnn_probability": 0.31,
                        },
                    ]
                )

            def knowledge_query(self, query: str, top_k: int = 3) -> dict[str, object]:
                return {
                    "query": query,
                    "row_count": 1,
                    "rows": [
                        {"source_id": "project_knowledge", "title": "Graph", "content": "Graph model blends spatial context.", "score": 0.8}
                    ],
                    "answer": "Graph model blends spatial context.",
                    "llm_enabled": False,
                }

            def graph_runtime_status(self) -> dict[str, object]:
                return {
                    "configured_enabled": True,
                    "torch_available": True,
                    "artifact_exists": True,
                    "artifact_path": "artifacts/graph_gnn.pt",
                    "runtime_ready": True,
                }

            def spatiotemporal_runtime_status(self) -> dict[str, object]:
                return {
                    "configured_enabled": True,
                    "torch_available": True,
                    "artifact_exists": True,
                    "artifact_path": "artifacts/spatiotemporal_gnn.pt",
                    "runtime_ready": True,
                }

            def spatiotemporal_graph_inference_from_csv(self, csv_path: str, artifact_path: str | None = None) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "region_id": "R1",
                            "state": "Kerala",
                            "district": "Idukki",
                            "date": "2026-01-01",
                            "spatiotemporal_gnn_probability": 0.81,
                            "uncertainty_score": 0.03,
                            "confidence_level": "High",
                            "recommended_interpretation": "Escalate monitoring and field verification.",
                            "top_driver": "rainfall_24h_mm",
                            "top_driver_score": 1.27,
                            "top_feature_drivers": ["rainfall_24h_mm", "soil_wetness_index", "neighbor_pressure"],
                            "top_feature_driver_scores": [1.27, 0.91, 0.66],
                            "top_neighbor_influences": [{"region_id": "R2", "weight": 0.42}],
                            "sequence_length": 7,
                        }
                    ]
                )

            def model_registry_status(self) -> dict[str, object]:
                registry_row = {
                    "artifact_path": "artifacts/example.pkl",
                    "artifact_exists": True,
                    "metadata_path": "artifacts/example_metadata.json",
                    "metadata_exists": True,
                    "artifact_modified_at": "2026-01-01T00:00:00+00:00",
                    "runtime_versions": {"python": "3.12.7", "scikit_learn": "1.8.0", "torch": "2.8.0"},
                    "metadata": {"model_type": "example"},
                    "version_match": True,
                }
                return {
                    "risk_model": registry_row,
                    "graph_model": registry_row,
                    "spatiotemporal_graph_model": registry_row,
                    "forecast_model": registry_row,
                }

            def refresh_real_data_sources(self) -> dict[str, object]:
                return {
                    "nasa_snapshot": {"mode": "downloaded", "output_path": "data/raw/nasa.json", "manifest_path": "data/raw/nasa.json.manifest"},
                    "sentinel_snapshot": {"mode": "downloaded", "output_path": "data/raw/sentinel.json", "manifest_path": "data/raw/sentinel.json.manifest"},
                    "regional_weather_refresh": {
                        "mode": "downloaded",
                        "daily_output_path": "data/raw/daily.csv",
                        "latest_output_path": "data/raw/latest.csv",
                        "row_count": 10,
                    },
                }

        self.app = create_app(".")
        self.app.dependency_overrides[service_dependency] = lambda: StubService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    def test_assess_endpoint_returns_paginated_envelope(self) -> None:
        response = self.client.post("/assess", json={"csv_path": "data/india/india_regions.csv", "limit": 1, "offset": 0})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_count"], 2)
        self.assertEqual(payload["returned_count"], 1)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["summary"]["top_region_id"], "R1")
        self.assertEqual(payload["rows"][0]["confidence_level"], "High")
        self.assertEqual(payload["rows"][0]["alert_band"], "Critical")

    def test_scenario_endpoint_returns_paginated_envelope(self) -> None:
        response = self.client.post(
            "/scenario",
            json={"csv_path": "data/india/india_regions.csv", "target_state": "Kerala", "limit": 1, "offset": 1},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["offset"], 1)
        self.assertEqual(payload["returned_count"], 1)

    def test_forecast_endpoint_returns_summary_and_rows(self) -> None:
        response = self.client.post("/forecast", json={"windows_csv_path": "data/india/rainfall_windows.csv", "limit": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_count"], 2)
        self.assertEqual(payload["returned_count"], 1)
        self.assertIn("predicted_mean", payload["summary"])

    def test_analytics_summary_endpoint_returns_compact_summary(self) -> None:
        response = self.client.post("/analytics/summary", json={"csv_path": "data/india/india_regions.csv"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["region_count"], 2)
        self.assertEqual(payload["top_region"]["region_id"], "R1")

    def test_graph_infer_endpoint_returns_paginated_rows(self) -> None:
        response = self.client.post("/graph/infer", json={"csv_path": "data/india/india_graph_training.csv", "limit": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_count"], 2)
        self.assertEqual(payload["returned_count"], 1)
        self.assertIn("graph_gnn_probability", payload["rows"][0])

    def test_knowledge_query_endpoint_returns_retrieval_answer(self) -> None:
        response = self.client.post("/knowledge/query", json={"query": "What is the graph model?", "top_k": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["row_count"], 1)
        self.assertTrue(payload["answer"])

    def test_spatiotemporal_graph_endpoint_returns_explainable_rows(self) -> None:
        response = self.client.post("/graph/spatiotemporal", json={"csv_path": "data/regional/mock.csv", "limit": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_count"], 1)
        self.assertEqual(payload["rows"][0]["confidence_level"], "High")
        self.assertIn("top_neighbor_influences", payload["rows"][0])

    def test_model_status_endpoint_returns_artifact_registry(self) -> None:
        response = self.client.get("/models/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["risk_model"]["artifact_exists"])
        self.assertTrue(payload["spatiotemporal_graph_model"]["version_match"])

    def test_ingestion_refresh_endpoint_returns_refresh_summary(self) -> None:
        response = self.client.post("/ingestion/refresh", json={"refresh_weather": True, "refresh_satellite": True})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["regional_weather_refresh"]["row_count"], 10)


if __name__ == "__main__":
    unittest.main()
