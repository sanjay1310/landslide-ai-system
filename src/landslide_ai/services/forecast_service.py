from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from landslide_ai.forecasting.baseline import BaselineRainfallForecaster


@dataclass(slots=True)
class ForecastArtifactBundle:
    model_path: Path
    metadata_path: Path


class ForecastInferenceService:
    def __init__(self, artifacts: ForecastArtifactBundle) -> None:
        self.artifacts = artifacts
        self.model = BaselineRainfallForecaster()
        self.metadata: dict[str, object] | None = None

    def load(self) -> None:
        self.model.load(self.artifacts.model_path)
        with open(self.artifacts.metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

    def available(self) -> bool:
        return self.artifacts.model_path.exists() and self.artifacts.metadata_path.exists()

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.metadata is None:
            self.load()
        assert self.metadata is not None
        feature_columns = list(self.metadata["feature_columns"])
        target_column = str(self.metadata["target_column"])
        predictions = self.model.predict(frame, feature_columns)
        output = frame.copy()
        output["predicted_rainfall"] = predictions
        if target_column in output.columns:
            output["absolute_error"] = (output[target_column] - output["predicted_rainfall"]).abs()
        return output


def default_forecast_artifacts(project_root: str | Path = ".") -> ForecastArtifactBundle:
    root = Path(project_root)
    return ForecastArtifactBundle(
        model_path=root / "artifacts" / "rainfall_baseline.pkl",
        metadata_path=root / "artifacts" / "rainfall_baseline_metadata.json",
    )
