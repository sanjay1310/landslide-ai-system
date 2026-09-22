from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from landslide_ai.api.schemas import (
    AssessmentRequest,
    AssessmentSummaryResponse,
    RiskAssessmentListResponse,
    AnalyticsRequest,
    ArtifactRegistryRowResponse,
    GraphInferenceRequest,
    GraphInferenceResponse,
    AnalyticsSummaryResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    ModelRegistryResponse,
    RefreshIngestionRequest,
    RefreshIngestionResponse,
    RiskAssessmentResponse,
    ScenarioRequest,
    SpatioTemporalGraphInferenceRequest,
    SpatioTemporalGraphInferenceResponse,
    SpatioTemporalGraphInferenceRowResponse,
)
from landslide_ai.api.security import require_api_key
from landslide_ai.config import load_config
from landslide_ai.services.system_service import LandslideSystemService


@lru_cache(maxsize=4)
def get_service(project_root: str) -> LandslideSystemService:
    return LandslideSystemService(project_root)


def service_dependency(request: Request) -> LandslideSystemService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        project_root = request.app.state.project_root
        service = get_service(project_root)
        request.app.state.service = service
    return service


def create_app(project_root: str | Path = ".") -> FastAPI:
    root = str(Path(project_root).resolve())
    config = load_config(root)
    app = FastAPI(
        title="Landslide AI System API",
        version="0.2.0",
        description="API for landslide assessment, scenario simulation, analytics, and saved forecast inference.",
    )
    app.state.project_root = root
    auth_dependency = require_api_key(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.api_cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="landslide-ai-system")

    def _paginate_rows(rows: list[dict[str, object]], limit: int, offset: int) -> list[dict[str, object]]:
        return rows[offset : offset + limit]

    @app.post("/assess", response_model=RiskAssessmentListResponse)
    def assess(
        payload: AssessmentRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> RiskAssessmentListResponse:
        try:
            results = service.assess_csv(payload.csv_path)
            serialized = [asdict(result) for result in results]
            page = _paginate_rows(serialized, payload.limit, payload.offset)
            top = results[0] if results else None
            return RiskAssessmentListResponse(
                total_count=len(results),
                returned_count=len(page),
                offset=payload.offset,
                limit=payload.limit,
                rows=[RiskAssessmentResponse(**row) for row in page],
                summary=AssessmentSummaryResponse(
                    high_risk_region_count=sum(result.risk_level == "High" for result in results),
                    mean_risk_score=round(
                        sum(result.risk_score for result in results) / len(results),
                        4,
                    )
                    if results
                    else 0.0,
                    top_region_id=None if top is None else top.region_id,
                    top_state=None if top is None else top.state,
                    top_district=None if top is None else top.district,
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/scenario", response_model=RiskAssessmentListResponse)
    def scenario(
        payload: ScenarioRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> RiskAssessmentListResponse:
        try:
            results = service.assess_scenario(
                csv_path=payload.csv_path,
                rainfall_multiplier=payload.rainfall_multiplier,
                antecedent_rainfall_multiplier=payload.antecedent_rainfall_multiplier,
                soil_wetness_delta=payload.soil_wetness_delta,
                ndvi_delta=payload.ndvi_delta,
                neighbor_risk_delta=payload.neighbor_risk_delta,
                target_state=payload.target_state,
            )
            limit = getattr(payload, "limit", 50)
            offset = getattr(payload, "offset", 0)
            serialized = [asdict(result) for result in results]
            page = _paginate_rows(serialized, limit, offset)
            top = results[0] if results else None
            return RiskAssessmentListResponse(
                total_count=len(results),
                returned_count=len(page),
                offset=offset,
                limit=limit,
                rows=[RiskAssessmentResponse(**row) for row in page],
                summary=AssessmentSummaryResponse(
                    high_risk_region_count=sum(result.risk_level == "High" for result in results),
                    mean_risk_score=round(
                        sum(result.risk_score for result in results) / len(results),
                        4,
                    )
                    if results
                    else 0.0,
                    top_region_id=None if top is None else top.region_id,
                    top_state=None if top is None else top.state,
                    top_district=None if top is None else top.district,
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(
        payload: ForecastRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> ForecastResponse:
        try:
            frame = service.forecast_from_windows(payload.windows_csv_path)
            rows = frame.to_dict(orient="records")
            page = _paginate_rows(rows, payload.limit, payload.offset)
            summary: dict[str, object] = {
                "predicted_mean": round(float(frame["predicted_rainfall"].mean()), 4)
                if "predicted_rainfall" in frame.columns and not frame.empty
                else None,
                "absolute_error_mean": round(float(frame["absolute_error"].mean()), 4)
                if "absolute_error" in frame.columns and not frame.empty
                else None,
            }
            return ForecastResponse(
                total_count=len(frame),
                returned_count=len(page),
                offset=payload.offset,
                limit=payload.limit,
                columns=frame.columns.tolist(),
                rows=page,
                summary=summary,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/analytics/summary", response_model=AnalyticsSummaryResponse)
    def analytics_summary(
        payload: AnalyticsRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> AnalyticsSummaryResponse:
        try:
            summary = service.analytics_summary(payload.csv_path)
            return AnalyticsSummaryResponse(**summary)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/graph/infer", response_model=GraphInferenceResponse)
    def graph_infer(
        payload: GraphInferenceRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> GraphInferenceResponse:
        try:
            frame = service.graph_inference_from_csv(
                csv_path=payload.csv_path,
                artifact_path=payload.artifact_path,
            )
            rows = frame[["region_id", "state", "district", "label", "graph_gnn_probability"]].copy()
            rows["label"] = rows["label"].where(rows["label"].notna(), None)
            serialized_rows = rows.to_dict(orient="records")
            page = _paginate_rows(serialized_rows, payload.limit, payload.offset)
            return GraphInferenceResponse(
                total_count=len(rows),
                returned_count=len(page),
                offset=payload.offset,
                limit=payload.limit,
                artifact_path=str(payload.artifact_path or config.graph_model_artifact),
                rows=page,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/config/status")
    def config_status(
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> dict[str, object]:
        graph_status = service.graph_runtime_status()
        st_graph_status = service.spatiotemporal_runtime_status()
        return {
            "api_auth_enabled": config.api_auth_enabled,
            "api_cors_origins": list(config.api_cors_origins),
            "enable_llm": config.enable_llm,
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "enable_trained_risk_model": config.enable_trained_risk_model,
            "risk_model_artifact": str(config.risk_model_artifact),
            "enable_graph_model": config.enable_graph_model,
            "graph_model_artifact": str(config.graph_model_artifact),
            "default_region_csv": str(config.default_region_csv),
            "default_rainfall_windows_csv": str(config.default_rainfall_windows_csv),
            "official_district_geojson": str(config.official_district_geojson),
            "official_state_geojson": str(config.official_state_geojson),
            "graph_distance_scale_km": config.graph_distance_scale_km,
            "graph_propagation_steps": config.graph_propagation_steps,
            "graph_propagation_alpha": config.graph_propagation_alpha,
            "graph_runtime": graph_status,
            "spatiotemporal_graph_runtime": st_graph_status,
        }

    @app.get("/models/status", response_model=ModelRegistryResponse)
    def model_status(
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> ModelRegistryResponse:
        registry = service.model_registry_status()
        return ModelRegistryResponse(
            risk_model=ArtifactRegistryRowResponse(**registry["risk_model"]),
            graph_model=ArtifactRegistryRowResponse(**registry["graph_model"]),
            spatiotemporal_graph_model=ArtifactRegistryRowResponse(**registry["spatiotemporal_graph_model"]),
            forecast_model=ArtifactRegistryRowResponse(**registry["forecast_model"]),
        )

    @app.post("/ingestion/refresh", response_model=RefreshIngestionResponse)
    def refresh_ingestion(
        payload: RefreshIngestionRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> RefreshIngestionResponse:
        del payload
        try:
            response = service.refresh_real_data_sources()
            return RefreshIngestionResponse(**response)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/knowledge/query", response_model=KnowledgeQueryResponse)
    def knowledge_query(
        payload: KnowledgeQueryRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> KnowledgeQueryResponse:
        try:
            response = service.knowledge_query(payload.query, top_k=payload.top_k)
            return KnowledgeQueryResponse(**response)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/graph/spatiotemporal", response_model=SpatioTemporalGraphInferenceResponse)
    def spatiotemporal_graph_infer(
        payload: SpatioTemporalGraphInferenceRequest,
        _: None = Depends(auth_dependency),
        service: LandslideSystemService = Depends(service_dependency),
    ) -> SpatioTemporalGraphInferenceResponse:
        try:
            frame = service.spatiotemporal_graph_inference_from_csv(
                csv_path=payload.csv_path,
                artifact_path=payload.artifact_path,
            )
            rows = frame.to_dict(orient="records")
            page = _paginate_rows(rows, payload.limit, payload.offset)
            return SpatioTemporalGraphInferenceResponse(
                total_count=len(rows),
                returned_count=len(page),
                offset=payload.offset,
                limit=payload.limit,
                artifact_path=str(payload.artifact_path or config.spatiotemporal_graph_model_artifact),
                rows=[SpatioTemporalGraphInferenceRowResponse(**row) for row in page],
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app(".")
