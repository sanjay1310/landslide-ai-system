from dataclasses import dataclass


@dataclass(slots=True)
class AgentSignals:
    rainfall_signal: float
    terrain_signal: float
    vegetation_signal: float
    graph_signal: float
    propagated_graph_signal: float


@dataclass(slots=True)
class GraphInsights:
    local_influence: float
    propagated_influence: float
    proximity_score: float
    neighbor_ids: list[str]


@dataclass(slots=True)
class RiskComponents:
    rainfall_component: float
    antecedent_component: float
    terrain_component: float
    soil_component: float
    vegetation_component: float
    graph_component: float
    propagated_graph_component: float
    uncertainty_score: float


@dataclass(slots=True)
class RiskAssessment:
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
    calibrated_risk_score: float = 0.0
    confidence_level: str = "Medium"
    alert_band: str = "Routine"
    recommended_action: str = ""
    advisory: str = ""
