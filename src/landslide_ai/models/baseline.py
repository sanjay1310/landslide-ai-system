from landslide_ai.data.schemas import RegionFeatures
from landslide_ai.models.schemas import RiskComponents


class BaselineRiskModel:
    """Weighted ensemble-style baseline with interpretable component scores."""

    def predict(
        self,
        features: RegionFeatures,
        propagated_graph_signal: float,
    ) -> tuple[float, float, RiskComponents]:
        rainfall_stress = self._stress_curve(features.rainfall_intensity, midpoint=0.50, sharpness=6.0)
        antecedent_stress = self._stress_curve(features.antecedent_rainfall, midpoint=0.45, sharpness=5.2)
        terrain_stress = self._stress_curve(features.slope_factor, midpoint=0.50, sharpness=5.6)
        soil_stress = self._stress_curve(features.soil_wetness, midpoint=0.42, sharpness=4.2)
        vegetation_stress = self._stress_curve(features.vegetation_stress, midpoint=0.35, sharpness=4.0)
        graph_stress = self._stress_curve(features.graph_influence, midpoint=0.30, sharpness=3.6)
        propagated_graph_stress = self._stress_curve(propagated_graph_signal, midpoint=0.30, sharpness=3.8)

        compound_trigger = min(
            rainfall_stress * 0.45
            + antecedent_stress * 0.25
            + terrain_stress * 0.30,
            1.0,
        )
        saturation_penalty = min(
            0.60 * min(features.soil_wetness, 1.0) + 0.40 * antecedent_stress,
            1.0,
        )

        rainfall_component = rainfall_stress * 0.20
        antecedent_component = antecedent_stress * 0.16
        terrain_component = terrain_stress * 0.18
        soil_component = soil_stress * 0.13
        vegetation_component = vegetation_stress * 0.08
        graph_component = graph_stress * 0.09
        propagated_graph_component = propagated_graph_stress * 0.09
        trigger_component = compound_trigger * 0.07

        current_risk = min(
            rainfall_component
            + antecedent_component
            + terrain_component
            + soil_component
            + vegetation_component
            + graph_component
            + propagated_graph_component,
            0.82,
        )
        current_risk = min(
            current_risk
            + trigger_component
            + saturation_penalty * 0.06,
            1.0,
        )

        forecast_risk = min(
            current_risk * 0.54
            + antecedent_stress * 0.16
            + propagated_graph_stress * 0.12
            + rainfall_stress * 0.08
            + compound_trigger * 0.10,
            1.0,
        )
        uncertainty_score = min(
            abs(rainfall_stress - antecedent_stress) * 0.35
            + abs(graph_stress - propagated_graph_stress) * 0.30
            + abs(compound_trigger - current_risk) * 0.20
            + (1.0 - min(features.soil_wetness, 1.0)) * 0.15,
            1.0,
        )
        return current_risk, forecast_risk, RiskComponents(
            rainfall_component=rainfall_component,
            antecedent_component=antecedent_component,
            terrain_component=terrain_component,
            soil_component=soil_component,
            vegetation_component=vegetation_component,
            graph_component=graph_component,
            propagated_graph_component=propagated_graph_component,
            uncertainty_score=uncertainty_score,
        )

    @staticmethod
    def _stress_curve(value: float, midpoint: float, sharpness: float) -> float:
        centered = (value - midpoint) * sharpness
        if centered >= 0:
            return min(centered / (1.0 + centered) + 0.5, 1.0)
        return max(0.5 + centered / (1.0 - centered), 0.0)
