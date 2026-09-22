from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from landslide_ai.data.schemas import RegionFeatures
from landslide_ai.models.baseline import BaselineRiskModel
from landslide_ai.models.schemas import RiskComponents
from landslide_ai.models.trained_risk import TRAINED_RISK_FEATURE_COLUMNS
from landslide_ai.utils.optional_dependencies import torch_available


@dataclass(slots=True)
class DeepRiskArtifacts:
    model_path: Path
    metadata_path: Path


def deep_risk_metadata_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_name(f"{path.stem}_metadata.json")


class DeepRiskModel:
    """Feed-forward neural risk model for tabular landslide features."""

    def __init__(self, hidden_dims: tuple[int, int] = (32, 16), dropout: float = 0.15) -> None:
        if not torch_available():
            raise ImportError("PyTorch is not installed. Install `torch` to use the deep risk model.")
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.feature_columns = list(TRAINED_RISK_FEATURE_COLUMNS)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.model = self._build_model(len(self.feature_columns))

    def _build_model(self, input_dim: int):
        nn = self.nn
        layers: list[object] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        return nn.Sequential(*layers)

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

    def fit(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        label_column: str = "label",
        epochs: int = 180,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        torch = self.torch
        selected = feature_columns or self.feature_columns
        self.feature_columns = list(selected)
        x = frame[self.feature_columns].astype("float32").to_numpy()
        y = frame[label_column].astype("float32").to_numpy()
        self.feature_mean = x.mean(axis=0)
        self.feature_std = np.where(x.std(axis=0) == 0.0, 1.0, x.std(axis=0))
        scaled = (x - self.feature_mean) / self.feature_std

        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        self.model = self._build_model(len(self.feature_columns))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = self.nn.BCEWithLogitsLoss()

        losses: list[float] = []
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        return losses

    def predict_proba_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("DeepRiskModel must be fit or loaded before inference.")
        x = frame[self.feature_columns].astype("float32").to_numpy()
        scaled = (x - self.feature_mean) / self.feature_std
        x_tensor = self.torch.tensor(scaled, dtype=self.torch.float32)
        self.model.eval()
        with self.torch.no_grad():
            probabilities = self.torch.sigmoid(self.model(x_tensor)).squeeze(-1).cpu().numpy()
        return probabilities.astype(float)

    def predict_proba(self, features: RegionFeatures, propagated_graph_signal: float) -> float:
        frame = pd.DataFrame([self.build_feature_row(features, propagated_graph_signal)])
        return float(self.predict_proba_from_frame(frame)[0])

    def predict(
        self,
        features: RegionFeatures,
        propagated_graph_signal: float,
        fallback_model: BaselineRiskModel | None = None,
    ) -> tuple[float, float, RiskComponents]:
        baseline = fallback_model or BaselineRiskModel()
        baseline_risk, baseline_forecast, components = baseline.predict(features, propagated_graph_signal)
        raw_probability = min(max(self.predict_proba(features, propagated_graph_signal), 0.0), 1.0)

        # Compress extreme neural outputs and blend with the interpretable baseline
        # so the runtime score behaves more like a practical operational index.
        softened_probability = 0.10 + (raw_probability * 0.80)
        risk_score = min(
            max((softened_probability * 0.62) + (baseline_risk * 0.38), 0.0),
            0.93,
        )
        forecast_risk = min(
            max((risk_score * 0.62) + (baseline_forecast * 0.38), 0.0),
            0.95,
        )
        return risk_score, forecast_risk, components

    def save(self, model_path: str | Path) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
            path,
        )
        metadata = {
            "feature_columns": self.feature_columns,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }
        deep_risk_metadata_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load(self, model_path: str | Path) -> None:
        payload = self.torch.load(Path(model_path), map_location="cpu", weights_only=False)
        self.feature_columns = list(payload["feature_columns"])
        self.feature_mean = np.asarray(payload["feature_mean"], dtype="float32")
        self.feature_std = np.asarray(payload["feature_std"], dtype="float32")
        self.hidden_dims = tuple(int(item) for item in payload.get("hidden_dims", self.hidden_dims))
        self.dropout = float(payload.get("dropout", self.dropout))
        self.model = self._build_model(len(self.feature_columns))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
