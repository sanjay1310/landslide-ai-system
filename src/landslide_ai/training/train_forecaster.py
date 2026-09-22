from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
import platform

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn

from landslide_ai.forecasting.baseline import BaselineRainfallForecaster, ForecastPrediction
from landslide_ai.mlops import write_experiment_log


@dataclass(slots=True)
class ForecastTrainingResult:
    sample_count: int
    active_sample_count: int
    feature_columns: list[str]
    target_column: str
    train_mae: float
    train_rmse: float
    train_r2: float
    validation_mae: float
    validation_rmse: float
    validation_r2: float
    validation_active_mae: float
    validation_active_rmse: float
    validation_split_type: str
    model_path: str
    predictions: list[ForecastPrediction]


def _sorted_feature_columns(columns: list[str]) -> list[str]:
    lag_columns = sorted(
        (column for column in columns if column.startswith("rainfall_t_minus_")),
        key=lambda item: int(item.rsplit("_", 1)[-1]),
        reverse=True,
    )
    other_columns = sorted(column for column in columns if not column.startswith("rainfall_t_minus_"))
    return lag_columns + other_columns


def _temporal_holdout_split(
    frame: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "reference_date" not in frame.columns:
        split_index = max(1, int(len(frame) * (1.0 - validation_fraction)))
        return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()

    working = frame.copy()
    working["reference_date"] = pd.to_datetime(working["reference_date"], errors="coerce")
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for _, group in working.sort_values(["region_id", "reference_date"]).groupby("region_id", dropna=False):
        validation_count = max(1, ceil(len(group) * validation_fraction))
        if len(group) <= validation_count:
            validation_count = 1
        train_part = group.iloc[:-validation_count]
        validation_part = group.iloc[-validation_count:]
        if train_part.empty:
            train_part = group.iloc[:-1]
            validation_part = group.iloc[-1:]
        train_parts.append(train_part)
        validation_parts.append(validation_part)
    train_frame = pd.concat(train_parts, ignore_index=True)
    validation_frame = pd.concat(validation_parts, ignore_index=True)
    return train_frame, validation_frame


def _active_window_mask(frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.Series:
    lag_columns = [column for column in feature_columns if column.startswith("rainfall_t_minus_")]
    if not lag_columns:
        return frame[target_column] >= 10.0
    lag_peak = frame[lag_columns].max(axis=1)
    return (frame[target_column] >= 10.0) | (lag_peak >= 10.0)


def train_rainfall_forecaster_from_windows(
    csv_path: str,
    model_output_path: str = "artifacts/rainfall_baseline.pkl",
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> ForecastTrainingResult:
    frame = pd.read_csv(csv_path)
    feature_columns = _sorted_feature_columns(
        [
            column
            for column in frame.columns
            if column not in {"region_id", "reference_date"}
            and not column.startswith("rainfall_t_plus_")
        ]
    )
    target_column = next(column for column in frame.columns if column.startswith("rainfall_t_plus_"))
    if frame.empty:
        raise ValueError("Forecast training dataset is empty. Re-run ingestion to generate rainfall windows.")

    train_frame, validation_frame = _temporal_holdout_split(frame, validation_fraction=validation_fraction)

    forecaster = BaselineRainfallForecaster()
    forecaster.fit(train_frame, feature_columns, target_column)
    train_predicted = forecaster.predict(train_frame, feature_columns)
    validation_predicted = forecaster.predict(validation_frame, feature_columns)
    validation_active_mask = _active_window_mask(validation_frame, feature_columns, target_column)
    validation_active_actual = validation_frame.loc[validation_active_mask, target_column]
    validation_active_predicted = [
        prediction
        for prediction, is_active in zip(validation_predicted, validation_active_mask.tolist(), strict=True)
        if is_active
    ]

    final_forecaster = BaselineRainfallForecaster()
    final_forecaster.fit(frame, feature_columns, target_column)
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    final_forecaster.save(model_output_path)
    metadata_path = Path(model_output_path).with_name(f"{Path(model_output_path).stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "target_column": target_column,
                "source_csv": csv_path,
                "validation_fraction": validation_fraction,
                "validation_split_type": "temporal_holdout_by_region",
                "sample_count": len(frame),
                "active_sample_count": int(_active_window_mask(frame, feature_columns, target_column).sum()),
                "train_mae": float(mean_absolute_error(train_frame[target_column], train_predicted)),
                "train_rmse": float(mean_squared_error(train_frame[target_column], train_predicted) ** 0.5),
                "train_r2": float(r2_score(train_frame[target_column], train_predicted)),
                "validation_mae": float(mean_absolute_error(validation_frame[target_column], validation_predicted)),
                "validation_rmse": float(mean_squared_error(validation_frame[target_column], validation_predicted) ** 0.5),
                "validation_r2": float(r2_score(validation_frame[target_column], validation_predicted)),
                "validation_active_mae": float(mean_absolute_error(validation_active_actual, validation_active_predicted))
                if len(validation_active_actual) > 0
                else 0.0,
                "validation_active_rmse": float(mean_squared_error(validation_active_actual, validation_active_predicted) ** 0.5)
                if len(validation_active_actual) > 0
                else 0.0,
                "random_state": random_state,
                "scikit_learn_version": sklearn.__version__,
                "python_version": platform.python_version(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    experiment_log_path = write_experiment_log(
        experiment_name="rainfall_forecaster",
        parameters={
            "csv_path": csv_path,
            "model_output_path": model_output_path,
            "validation_fraction": validation_fraction,
            "random_state": random_state,
        },
        metrics=json.loads(metadata_path.read_text(encoding="utf-8")),
        artifacts={
            "model_path": model_output_path,
            "metadata_path": str(metadata_path),
        },
    )
    predicted = final_forecaster.predict(frame, feature_columns)

    predictions = [
        ForecastPrediction(
            region_id=str(row["region_id"]),
            reference_date=str(row["reference_date"]),
            actual_rainfall=float(actual),
            predicted_rainfall=float(prediction),
            absolute_error=abs(float(actual) - float(prediction)),
        )
        for (_, row), actual, prediction in zip(
            frame.iterrows(),
            frame[target_column].tolist(),
            predicted,
            strict=True,
        )
    ]

    return ForecastTrainingResult(
        sample_count=len(frame),
        active_sample_count=int(_active_window_mask(frame, feature_columns, target_column).sum()),
        feature_columns=feature_columns,
        target_column=target_column,
        train_mae=float(mean_absolute_error(train_frame[target_column], train_predicted)),
        train_rmse=float(mean_squared_error(train_frame[target_column], train_predicted) ** 0.5),
        train_r2=float(r2_score(train_frame[target_column], train_predicted)),
        validation_mae=float(mean_absolute_error(validation_frame[target_column], validation_predicted)),
        validation_rmse=float(mean_squared_error(validation_frame[target_column], validation_predicted) ** 0.5),
        validation_r2=float(r2_score(validation_frame[target_column], validation_predicted)),
        validation_active_mae=float(mean_absolute_error(validation_active_actual, validation_active_predicted))
        if len(validation_active_actual) > 0
        else 0.0,
        validation_active_rmse=float(mean_squared_error(validation_active_actual, validation_active_predicted) ** 0.5)
        if len(validation_active_actual) > 0
        else 0.0,
        validation_split_type="temporal_holdout_by_region",
        model_path=model_output_path,
        predictions=predictions,
    )


def forecast_predictions_frame(result: ForecastTrainingResult) -> pd.DataFrame:
    return pd.DataFrame([asdict(prediction) for prediction in result.predictions])
