from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from landslide_ai.data.schemas import RegionFeatures
from landslide_ai.models.baseline import BaselineRiskModel
from landslide_ai.models.schemas import RiskComponents


TRAINED_RISK_FEATURE_COLUMNS = [
    "rainfall_intensity",
    "antecedent_rainfall",
    "slope_factor",
    "soil_wetness",
    "vegetation_stress",
    "graph_influence",
    "propagated_graph_signal",
    "rainfall_terrain_interaction",
    "rainfall_soil_interaction",
    "vegetation_soil_interaction",
    "graph_pressure",
    "terrain_wetness_interaction",
]


@dataclass(slots=True)
class TrainedRiskArtifactBundle:
    model_path: Path
    metadata_path: Path


def default_trained_risk_metadata_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_name(f"{path.stem}_metadata.json")


class TrainedRiskModel:
    """Supervised classifier for current risk with rule-based explainability fallback."""

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator
        self.metadata: dict[str, object] | None = None

    @staticmethod
    def build_feature_row(features: RegionFeatures, propagated_graph_signal: float) -> dict[str, float]:
        rainfall_terrain_interaction = float(features.rainfall_intensity) * float(features.slope_factor)
        rainfall_soil_interaction = float(features.antecedent_rainfall) * float(features.soil_wetness)
        vegetation_soil_interaction = float(features.vegetation_stress) * float(features.soil_wetness)
        graph_pressure = float(features.graph_influence) * float(propagated_graph_signal)
        terrain_wetness_interaction = float(features.slope_factor) * float(features.soil_wetness)
        return {
            "rainfall_intensity": float(features.rainfall_intensity),
            "antecedent_rainfall": float(features.antecedent_rainfall),
            "slope_factor": float(features.slope_factor),
            "soil_wetness": float(features.soil_wetness),
            "vegetation_stress": float(features.vegetation_stress),
            "graph_influence": float(features.graph_influence),
            "propagated_graph_signal": float(propagated_graph_signal),
            "rainfall_terrain_interaction": rainfall_terrain_interaction,
            "rainfall_soil_interaction": rainfall_soil_interaction,
            "vegetation_soil_interaction": vegetation_soil_interaction,
            "graph_pressure": graph_pressure,
            "terrain_wetness_interaction": terrain_wetness_interaction,
        }

    @staticmethod
    def create_estimator(model_type: str):
        if model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=160,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        return RandomForestClassifier(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )

    def available(self) -> bool:
        return self.estimator is not None

    def save(self, model_path: str | Path, metadata: dict[str, object]) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.estimator, handle)
        metadata_path = default_trained_risk_metadata_path(path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load(self, model_path: str | Path) -> None:
        path = Path(model_path)
        with open(path, "rb") as handle:
            self.estimator = pickle.load(handle)
        metadata_path = default_trained_risk_metadata_path(path)
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def predict_proba(self, features: RegionFeatures, propagated_graph_signal: float) -> float:
        if self.estimator is None:
            raise RuntimeError("Trained risk estimator is not loaded.")
        frame = pd.DataFrame([self.build_feature_row(features, propagated_graph_signal)])
        selected_features = list(self.metadata.get("feature_columns", TRAINED_RISK_FEATURE_COLUMNS)) if self.metadata else list(TRAINED_RISK_FEATURE_COLUMNS)
        probability = float(self.estimator.predict_proba(frame[selected_features])[0][1])
        return min(max(probability, 0.0), 1.0)

    def predict(
        self,
        features: RegionFeatures,
        propagated_graph_signal: float,
        fallback_model: BaselineRiskModel | None = None,
    ) -> tuple[float, float, RiskComponents]:
        baseline_model = fallback_model or BaselineRiskModel()
        baseline_risk, baseline_forecast, components = baseline_model.predict(features, propagated_graph_signal)
        if self.estimator is None:
            return baseline_risk, baseline_forecast, components

        trained_risk = self.predict_proba(features, propagated_graph_signal)
        forecast_risk = min((trained_risk * 0.60) + (baseline_forecast * 0.40), 1.0)
        return trained_risk, forecast_risk, components
