from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ScenarioRequest(BaseModel):
    csv_path: str = Field(default="data/india/india_regions.csv")
    rainfall_multiplier: float = Field(default=1.0, ge=0.0)
    antecedent_rainfall_multiplier: float = Field(default=1.0, ge=0.0)
    soil_wetness_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    ndvi_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    neighbor_risk_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    target_state: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class AssessmentRequest(BaseModel):
    csv_path: str = Field(default="data/india/india_regions.csv")
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class ForecastRequest(BaseModel):
    windows_csv_path: str = Field(default="data/india/rainfall_windows.csv")
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class AnalyticsRequest(BaseModel):
    csv_path: str = Field(default="data/india/india_regions.csv")


class GraphInferenceRequest(BaseModel):
    csv_path: str = Field(default="data/india/india_graph_training.csv")
    artifact_path: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class SpatioTemporalGraphInferenceRequest(BaseModel):
    csv_path: str = Field(default="data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv")
    artifact_path: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class RefreshIngestionRequest(BaseModel):
    refresh_weather: bool = True
    refresh_satellite: bool = True


class KnowledgeQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class RiskAssessmentResponse(BaseModel):
    region_id: str
    state: str
    district: str
    latitude: float
    longitude: float
    risk_score: float
    forecast_risk: float
    risk_level: str
    primary_driver: str
    uncertainty_score: float
    graph_exposure: float
    propagated_graph_exposure: float
    rainfall_exposure: float
    terrain_exposure: float
    vegetation_exposure: float
    calibrated_risk_score: float
    confidence_level: str
    alert_band: str
    recommended_action: str
    advisory: str


class AssessmentSummaryResponse(BaseModel):
    high_risk_region_count: int
    mean_risk_score: float
    top_region_id: str | None = None
    top_state: str | None = None
    top_district: str | None = None


class RiskAssessmentListResponse(BaseModel):
    total_count: int
    returned_count: int
    offset: int
    limit: int
    rows: list[RiskAssessmentResponse]
    summary: AssessmentSummaryResponse


class ForecastRowResponse(BaseModel):
    row: dict[str, object]


class AnalyticsTopRegionResponse(BaseModel):
    region_id: str
    state: str
    district: str
    risk_score: float
    forecast_risk: float
    primary_driver: str


class AnalyticsStateRowResponse(BaseModel):
    state: str
    risk_score: float
    forecast_risk: float
    uncertainty_score: float


class AnalyticsDriverRowResponse(BaseModel):
    primary_driver: str
    region_count: int


class AnalyticsSummaryResponse(BaseModel):
    region_count: int
    high_risk_region_count: int
    alert_region_count: int
    mean_risk_score: float
    top_region: AnalyticsTopRegionResponse | None
    state_summary: list[AnalyticsStateRowResponse]
    driver_summary: list[AnalyticsDriverRowResponse]


class GraphInferenceRowResponse(BaseModel):
    region_id: str
    state: str
    district: str
    label: int | None = None
    graph_gnn_probability: float


class GraphInferenceResponse(BaseModel):
    total_count: int
    returned_count: int
    offset: int
    limit: int
    artifact_path: str
    rows: list[GraphInferenceRowResponse]


class SpatioTemporalGraphInferenceRowResponse(BaseModel):
    region_id: str
    state: str
    district: str
    date: str
    spatiotemporal_gnn_probability: float
    uncertainty_score: float
    confidence_level: str
    recommended_interpretation: str
    top_driver: str
    top_driver_score: float
    top_feature_drivers: list[str]
    top_feature_driver_scores: list[float]
    top_neighbor_influences: list[dict[str, object]]
    sequence_length: int


class SpatioTemporalGraphInferenceResponse(BaseModel):
    total_count: int
    returned_count: int
    offset: int
    limit: int
    artifact_path: str
    rows: list[SpatioTemporalGraphInferenceRowResponse]


class ArtifactRegistryRowResponse(BaseModel):
    artifact_path: str
    artifact_exists: bool
    metadata_path: str
    metadata_exists: bool
    artifact_modified_at: str | None = None
    runtime_versions: dict[str, str | None]
    metadata: dict[str, object]
    version_match: bool | None = None


class ModelRegistryResponse(BaseModel):
    risk_model: ArtifactRegistryRowResponse
    graph_model: ArtifactRegistryRowResponse
    spatiotemporal_graph_model: ArtifactRegistryRowResponse
    forecast_model: ArtifactRegistryRowResponse


class RefreshIngestionResponse(BaseModel):
    nasa_snapshot: dict[str, object]
    sentinel_snapshot: dict[str, object]
    regional_weather_refresh: dict[str, object]


class ForecastResponse(BaseModel):
    total_count: int
    returned_count: int
    offset: int
    limit: int
    columns: list[str]
    rows: list[dict[str, object]]
    summary: dict[str, object]


class KnowledgeQueryRowResponse(BaseModel):
    source_id: str
    title: str
    content: str
    score: float


class KnowledgeQueryResponse(BaseModel):
    query: str
    row_count: int
    rows: list[KnowledgeQueryRowResponse]
    answer: str = ""
    llm_enabled: bool = False
