from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
from pathlib import Path

# Keep the persisted ensemble usable in constrained local and CI environments
# where OpenMP shared-memory setup can fail.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(slots=True)
class ForecastPrediction:
    region_id: str
    reference_date: str
    actual_rainfall: float
    predicted_rainfall: float
    absolute_error: float


class BaselineRainfallForecaster:
    """Ensemble forecaster over lagged and engineered rainfall window features."""

    def __init__(self) -> None:
        self.model = VotingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=140,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=140,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=160,
                        min_samples_leaf=10,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> None:
        engineered = self._build_feature_frame(frame, feature_columns)
        self.model.fit(engineered, frame[target_column])

    def predict(self, frame: pd.DataFrame, feature_columns: list[str]) -> list[float]:
        engineered = self._build_feature_frame(frame, feature_columns)
        predicted = self.model.predict(engineered)
        clipped = [max(float(value), 0.0) for value in predicted.tolist()]
        return clipped

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self.model, handle)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as handle:
            self.model = pickle.load(handle)

    @staticmethod
    def _build_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        feature_frame = frame[feature_columns].copy()
        lag_columns = [column for column in feature_columns if column.startswith("rainfall_t_minus_")]
        if lag_columns:
            lag_frame = feature_frame[lag_columns].astype(float)
            feature_frame["lag_sum"] = lag_frame.sum(axis=1)
            feature_frame["lag_mean"] = lag_frame.mean(axis=1)
            feature_frame["lag_max"] = lag_frame.max(axis=1)
            feature_frame["lag_min"] = lag_frame.min(axis=1)
            feature_frame["lag_std"] = lag_frame.std(axis=1, ddof=0)
            feature_frame["lag_range"] = feature_frame["lag_max"] - feature_frame["lag_min"]
            feature_frame["lag_wet_days"] = (lag_frame >= 1.0).sum(axis=1)
            feature_frame["lag_heavy_days"] = (lag_frame >= 35.0).sum(axis=1)
            feature_frame["lag_recent_value"] = lag_frame.iloc[:, -1]
            feature_frame["lag_oldest_value"] = lag_frame.iloc[:, 0]
            feature_frame["lag_trend"] = feature_frame["lag_recent_value"] - feature_frame["lag_oldest_value"]
            feature_frame["lag_recent_share"] = feature_frame["lag_recent_value"] / feature_frame["lag_sum"].clip(lower=1.0)
        if "reference_date" in frame.columns:
            reference_ts = pd.to_datetime(frame["reference_date"], errors="coerce")
            feature_frame["reference_dayofyear"] = reference_ts.dt.dayofyear.fillna(1).astype(float)
            feature_frame["reference_month"] = reference_ts.dt.month.fillna(1).astype(float)
        return feature_frame.fillna(0.0)
