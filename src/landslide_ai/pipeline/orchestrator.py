from landslide_ai.agents.advisory_agent import AdvisoryGenerationAgent
from landslide_ai.agents.graph_agent import GraphAgent
from landslide_ai.agents.prediction_agent import PredictionAgent
from landslide_ai.agents.vision_agent import VisionAgent
from landslide_ai.data.loader import load_region_records
from landslide_ai.data.sample_data import build_sample_records
from landslide_ai.data.schemas import RegionRecord
from landslide_ai.models.schemas import AgentSignals, RiskAssessment
from landslide_ai.pipeline.feature_engineering import build_features, normalize
from landslide_ai.services.scenario_service import apply_scenario


class LandslideSystemOrchestrator:
    def __init__(self) -> None:
        self.vision_agent = VisionAgent()
        self.prediction_agent = PredictionAgent()
        self.graph_agent = GraphAgent()
        self.advisory_agent = AdvisoryGenerationAgent()

    def assess_region(self, record: RegionRecord, graph_map: dict[str, object] | None = None) -> RiskAssessment:
        terrain_change_signal, vegetation_signal = self.vision_agent.evaluate(record)
        graph_insights = graph_map.get(record.region_id) if graph_map else None
        graph_signal = graph_insights.local_influence if graph_insights else self.graph_agent.evaluate(record)
        propagated_graph_signal = graph_insights.propagated_influence if graph_insights else graph_signal

        features = build_features(
            record=record,
            terrain_change_signal=terrain_change_signal,
            vegetation_signal=vegetation_signal,
            graph_signal=graph_signal,
        )

        risk_score, forecast_risk, components = self.prediction_agent.predict(
            features,
            propagated_graph_signal=propagated_graph_signal,
        )
        assessment = RiskAssessment(
            region_id=record.region_id,
            state=record.state,
            district=record.district,
            latitude=record.latitude,
            longitude=record.longitude,
            risk_score=risk_score,
            forecast_risk=forecast_risk,
            risk_level=self._risk_level(risk_score),
            primary_driver=self._primary_driver(components),
            uncertainty_score=components.uncertainty_score,
            graph_exposure=graph_signal,
            propagated_graph_exposure=propagated_graph_signal,
            rainfall_exposure=features.rainfall_intensity,
            terrain_exposure=features.slope_factor,
            vegetation_exposure=features.vegetation_stress,
            calibrated_risk_score=self._calibrated_risk_score(risk_score, forecast_risk, components.uncertainty_score),
            confidence_level=self._confidence_level(components.uncertainty_score),
            alert_band=self._alert_band(risk_score, forecast_risk, components.uncertainty_score),
            recommended_action=self._recommended_action(risk_score, forecast_risk, components.uncertainty_score),
        )

        signals = AgentSignals(
            rainfall_signal=normalize(record.rainfall_24h_mm, 250.0),
            terrain_signal=terrain_change_signal,
            vegetation_signal=vegetation_signal,
            graph_signal=graph_signal,
            propagated_graph_signal=propagated_graph_signal,
        )
        assessment.advisory = self.advisory_agent.generate(signals, assessment)
        return assessment

    def run_demo(self) -> RiskAssessment:
        records = build_sample_records()
        graph_map = self.graph_agent.evaluate_network(records)
        return self.assess_region(records[0], graph_map)

    def run_batch_demo(self) -> list[RiskAssessment]:
        records = build_sample_records()
        graph_map = self.graph_agent.evaluate_network(records)
        assessments = [self.assess_region(record, graph_map) for record in records]
        return sorted(assessments, key=lambda item: item.risk_score, reverse=True)

    def run_from_csv(self, csv_path: str) -> list[RiskAssessment]:
        records = load_region_records(csv_path)
        return self.run_records(records)

    def run_records(self, records: list[RegionRecord]) -> list[RiskAssessment]:
        graph_map = self.graph_agent.evaluate_network(records)
        assessments = [self.assess_region(record, graph_map) for record in records]
        return sorted(assessments, key=lambda item: item.risk_score, reverse=True)

    def run_scenario(
        self,
        records: list[RegionRecord],
        rainfall_multiplier: float = 1.0,
        antecedent_rainfall_multiplier: float = 1.0,
        soil_wetness_delta: float = 0.0,
        ndvi_delta: float = 0.0,
        neighbor_risk_delta: float = 0.0,
        target_state: str | None = None,
    ) -> list[RiskAssessment]:
        scenario_records = apply_scenario(
            records=records,
            rainfall_multiplier=rainfall_multiplier,
            antecedent_rainfall_multiplier=antecedent_rainfall_multiplier,
            soil_wetness_delta=soil_wetness_delta,
            ndvi_delta=ndvi_delta,
            neighbor_risk_delta=neighbor_risk_delta,
            target_state=target_state,
        )
        return self.run_records(scenario_records)

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.75:
            return "High"
        if score >= 0.45:
            return "Moderate"
        return "Low"

    @staticmethod
    def _calibrated_risk_score(risk_score: float, forecast_risk: float, uncertainty_score: float) -> float:
        blended = (0.6 * risk_score) + (0.4 * forecast_risk)
        confidence_penalty = max(0.0, 1.0 - (0.45 * uncertainty_score))
        return max(0.0, min(1.0, blended * confidence_penalty))

    @staticmethod
    def _confidence_level(uncertainty_score: float) -> str:
        if uncertainty_score <= 0.08:
            return "High"
        if uncertainty_score <= 0.16:
            return "Medium"
        return "Low"

    @classmethod
    def _alert_band(cls, risk_score: float, forecast_risk: float, uncertainty_score: float) -> str:
        calibrated = cls._calibrated_risk_score(risk_score, forecast_risk, uncertainty_score)
        confidence = cls._confidence_level(uncertainty_score)
        if calibrated >= 0.7 and confidence in {"High", "Medium"}:
            return "Critical"
        if calibrated >= 0.5 or forecast_risk >= 0.65:
            return "Watch"
        if confidence == "Low" and forecast_risk >= 0.4:
            return "Review"
        return "Routine"

    @classmethod
    def _recommended_action(cls, risk_score: float, forecast_risk: float, uncertainty_score: float) -> str:
        band = cls._alert_band(risk_score, forecast_risk, uncertainty_score)
        if band == "Critical":
            return "Escalate district monitoring, verify slope conditions, and prepare field advisories."
        if band == "Watch":
            return "Increase rainfall watch frequency and inspect districts with recent terrain or soil stress."
        if band == "Review":
            return "Treat this as an early-warning signal and recheck after the next data refresh."
        return "Continue routine monitoring and maintain baseline situational awareness."

    @staticmethod
    def _primary_driver(components) -> str:
        component_map = {
            "Rainfall": components.rainfall_component,
            "Antecedent Rainfall": components.antecedent_component,
            "Terrain": components.terrain_component,
            "Soil Wetness": components.soil_component,
            "Vegetation": components.vegetation_component,
            "Graph": components.graph_component,
            "Propagated Graph": components.propagated_graph_component,
        }
        return max(component_map.items(), key=lambda item: item[1])[0]
