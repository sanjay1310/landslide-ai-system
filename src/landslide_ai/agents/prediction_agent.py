from landslide_ai.data.schemas import RegionFeatures
from landslide_ai.config import load_config
from landslide_ai.models.baseline import BaselineRiskModel
from landslide_ai.models.deep_risk import DeepRiskModel
from landslide_ai.models.trained_risk import TrainedRiskModel
from landslide_ai.models.schemas import RiskComponents
from landslide_ai.utils.optional_dependencies import torch_available


class PredictionAgent:
    """Uses tabular features to estimate landslide risk."""

    def __init__(self) -> None:
        self.fallback_model = BaselineRiskModel()
        self.model = self.fallback_model
        config = load_config(".")
        if config.enable_trained_risk_model and config.risk_model_artifact.exists():
            try:
                trained_model = TrainedRiskModel()
                trained_model.load(config.risk_model_artifact)
                self.model = trained_model
                return
            except Exception:
                self.model = self.fallback_model
        if (
            config.enable_deep_risk_model
            and torch_available()
            and config.deep_risk_model_artifact.exists()
        ):
            try:
                deep_model = DeepRiskModel()
                deep_model.load(config.deep_risk_model_artifact)
                self.model = deep_model
                return
            except Exception:
                self.model = self.fallback_model

    def predict(self, features: RegionFeatures, propagated_graph_signal: float) -> tuple[float, float, RiskComponents]:
        if isinstance(self.model, DeepRiskModel):
            return self.model.predict(
                features,
                propagated_graph_signal=propagated_graph_signal,
                fallback_model=self.fallback_model,
            )
        if isinstance(self.model, TrainedRiskModel):
            return self.model.predict(
                features,
                propagated_graph_signal=propagated_graph_signal,
                fallback_model=self.fallback_model,
            )
        return self.fallback_model.predict(features, propagated_graph_signal)
