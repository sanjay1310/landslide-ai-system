from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import ast

import pandas as pd

from landslide_ai.config import load_config
from landslide_ai.data.loader import load_region_records
from landslide_ai.graph.gnn import predict_graph_probabilities_for_csv
from landslide_ai.graph.spatiotemporal import run_spatiotemporal_inference_from_frame
from landslide_ai.ingestion.nasa_power import ingest_nasa_power_by_regions
from landslide_ai.ingestion.real_sources import (
    ingest_nasa_power_dataset,
    ingest_sentinel_catalog_snapshot,
)
from landslide_ai.mlops import build_artifact_lineage
from landslide_ai.pipeline.orchestrator import LandslideSystemOrchestrator
from landslide_ai.services.forecast_service import (
    ForecastInferenceService,
    default_forecast_artifacts,
)
from landslide_ai.services.rag_service import get_rag_service
from landslide_ai.utils.optional_dependencies import torch_available


class LandslideSystemService:
    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root)
        self.config = load_config(self.project_root)
        self.orchestrator = LandslideSystemOrchestrator()
        self.forecast_service = ForecastInferenceService(default_forecast_artifacts(self.project_root))
        self.rag_service = get_rag_service(self.project_root)

    def graph_runtime_status(self) -> dict[str, object]:
        artifact_path = self.config.graph_model_artifact
        torch_ready = torch_available()
        configured_enabled = self.config.enable_graph_model
        artifact_exists = artifact_path.exists()
        runtime_ready = bool(configured_enabled and torch_ready and artifact_exists)
        return {
            "configured_enabled": configured_enabled,
            "torch_available": torch_ready,
            "artifact_exists": artifact_exists,
            "artifact_path": str(artifact_path),
            "runtime_ready": runtime_ready,
        }

    def spatiotemporal_runtime_status(self) -> dict[str, object]:
        artifact_path = self.config.spatiotemporal_graph_model_artifact
        torch_ready = torch_available()
        configured_enabled = self.config.enable_spatiotemporal_graph_model
        artifact_exists = artifact_path.exists()
        runtime_ready = bool(configured_enabled and torch_ready and artifact_exists)
        return {
            "configured_enabled": configured_enabled,
            "torch_available": torch_ready,
            "artifact_exists": artifact_exists,
            "artifact_path": str(artifact_path),
            "runtime_ready": runtime_ready,
        }

    @lru_cache(maxsize=8)
    def _load_records(self, csv_path: str) -> tuple:
        return tuple(load_region_records(csv_path))

    def assess_csv(self, csv_path: str | Path):
        records = list(self._load_records(str(Path(csv_path))))
        return self.orchestrator.run_records(records)

    def assess_scenario(
        self,
        csv_path: str | Path,
        rainfall_multiplier: float = 1.0,
        antecedent_rainfall_multiplier: float = 1.0,
        soil_wetness_delta: float = 0.0,
        ndvi_delta: float = 0.0,
        neighbor_risk_delta: float = 0.0,
        target_state: str | None = None,
    ):
        records = list(self._load_records(str(Path(csv_path))))
        return self.orchestrator.run_scenario(
            records=records,
            rainfall_multiplier=rainfall_multiplier,
            antecedent_rainfall_multiplier=antecedent_rainfall_multiplier,
            soil_wetness_delta=soil_wetness_delta,
            ndvi_delta=ndvi_delta,
            neighbor_risk_delta=neighbor_risk_delta,
            target_state=target_state,
        )

    def forecast_from_windows(self, windows_csv_path: str | Path) -> pd.DataFrame:
        try:
            frame = pd.read_csv(windows_csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        if frame.empty:
            return frame
        return self.forecast_service.predict_frame(frame)

    def analytics_summary(self, csv_path: str | Path) -> dict[str, object]:
        results = self.assess_csv(csv_path)
        risk_scores = [result.risk_score for result in results]
        high_risk_count = sum(result.risk_score >= 0.75 for result in results)
        alert_count = sum(result.forecast_risk >= 0.60 for result in results)
        top = results[0] if results else None
        state_frame = (
            pd.DataFrame(
                [
                    {
                        "state": result.state,
                        "risk_score": result.risk_score,
                        "forecast_risk": result.forecast_risk,
                        "uncertainty_score": result.uncertainty_score,
                    }
                    for result in results
                ]
            )
            .groupby("state", dropna=False)
            .mean()
            .round(3)
            .reset_index()
            .sort_values("risk_score", ascending=False)
        )
        driver_frame = (
            pd.DataFrame([{"primary_driver": result.primary_driver} for result in results])
            .groupby("primary_driver", dropna=False)
            .size()
            .reset_index(name="region_count")
            .sort_values("region_count", ascending=False)
        )
        return {
            "region_count": len(results),
            "high_risk_region_count": high_risk_count,
            "alert_region_count": alert_count,
            "mean_risk_score": round(sum(risk_scores) / len(risk_scores), 3) if risk_scores else 0.0,
            "top_region": None
            if top is None
            else {
                "region_id": top.region_id,
                "state": top.state,
                "district": top.district,
                "risk_score": round(top.risk_score, 3),
                "forecast_risk": round(top.forecast_risk, 3),
                "primary_driver": top.primary_driver,
            },
            "state_summary": state_frame.to_dict(orient="records"),
            "driver_summary": driver_frame.to_dict(orient="records"),
        }

    def graph_inference_from_csv(
        self,
        csv_path: str | Path,
        artifact_path: str | Path | None = None,
    ) -> pd.DataFrame:
        graph_status = self.graph_runtime_status()
        if not graph_status["configured_enabled"]:
            raise ImportError("Graph model runtime is disabled in configuration. Set LANDSLIDE_ENABLE_GRAPH_MODEL=true.")
        if not graph_status["torch_available"]:
            raise ImportError("PyTorch is not installed. Install `torch` to run graph inference.")
        graph_artifact = Path(artifact_path) if artifact_path else self.config.graph_model_artifact
        if not graph_artifact.exists():
            raise FileNotFoundError(f"Graph model artifact not found: {graph_artifact}")
        return predict_graph_probabilities_for_csv(csv_path, graph_artifact)

    def spatiotemporal_graph_inference_from_csv(
        self,
        csv_path: str | Path,
        artifact_path: str | Path | None = None,
    ) -> pd.DataFrame:
        runtime = self.spatiotemporal_runtime_status()
        if not runtime["configured_enabled"]:
            raise ImportError(
                "Spatio-temporal graph model runtime is disabled in configuration. "
                "Set LANDSLIDE_ENABLE_SPATIOTEMPORAL_GRAPH_MODEL=true."
            )
        if not runtime["torch_available"]:
            raise ImportError("PyTorch is not installed. Install `torch` to run spatio-temporal graph inference.")
        model_artifact = Path(artifact_path) if artifact_path else self.config.spatiotemporal_graph_model_artifact
        if not model_artifact.exists():
            raise FileNotFoundError(f"Spatio-temporal graph model artifact not found: {model_artifact}")
        csv_file = Path(csv_path)
        cache_path = self._spatiotemporal_cache_path(csv_file, model_artifact)
        if cache_path.exists() and cache_path.stat().st_mtime >= max(csv_file.stat().st_mtime, model_artifact.stat().st_mtime):
            cached = pd.read_csv(cache_path)
            for column in ["top_feature_drivers", "top_feature_driver_scores", "top_neighbor_influences"]:
                if column in cached.columns:
                    cached[column] = cached[column].apply(
                        lambda value: value if isinstance(value, list) else ast.literal_eval(value)
                    )
            return cached
        frame = pd.read_csv(csv_file)
        result = run_spatiotemporal_inference_from_frame(frame, model_artifact)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        return result

    def model_registry_status(self) -> dict[str, object]:
        return {
            "risk_model": build_artifact_lineage(self.config.risk_model_artifact),
            "graph_model": build_artifact_lineage(self.config.graph_model_artifact),
            "spatiotemporal_graph_model": build_artifact_lineage(self.config.spatiotemporal_graph_model_artifact),
            "forecast_model": build_artifact_lineage(self.project_root / "artifacts/rainfall_baseline.pkl"),
        }

    def refresh_real_data_sources(self) -> dict[str, object]:
        nasa_snapshot = ingest_nasa_power_dataset(self.config)
        sentinel_snapshot = ingest_sentinel_catalog_snapshot(self.config)
        regional_fetch = ingest_nasa_power_by_regions(self.config, self.config.default_region_csv)
        return {
            "nasa_snapshot": {
                "mode": nasa_snapshot.mode,
                "output_path": str(nasa_snapshot.output_path),
                "manifest_path": str(nasa_snapshot.manifest_path),
            },
            "sentinel_snapshot": {
                "mode": sentinel_snapshot.mode,
                "output_path": str(sentinel_snapshot.output_path),
                "manifest_path": str(sentinel_snapshot.manifest_path),
            },
            "regional_weather_refresh": {
                "mode": regional_fetch.mode,
                "daily_output_path": str(regional_fetch.daily_output_path),
                "latest_output_path": str(regional_fetch.latest_output_path),
                "row_count": regional_fetch.row_count,
            },
        }

    def _spatiotemporal_cache_path(self, csv_path: Path, model_artifact: Path) -> Path:
        key = hashlib.sha1(f"{csv_path.resolve()}::{model_artifact.resolve()}".encode("utf-8")).hexdigest()[:12]
        return self.project_root / "artifacts" / "cache" / f"spatiotemporal_{key}.csv"

    def knowledge_query(self, query: str, top_k: int = 3) -> dict[str, object]:
        return self.rag_service.answer_query(query, top_k=top_k)
