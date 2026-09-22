from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from landslide_ai.alerts.generator import AlertRecord
from landslide_ai.models.schemas import RiskAssessment
from landslide_ai.training.train_forecaster import ForecastTrainingResult
from landslide_ai.training.train_lstm_forecaster import LSTMForecastTrainingResult


def assessments_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def alerts_to_csv_bytes(alerts: list[AlertRecord]) -> bytes:
    frame = pd.DataFrame([asdict(alert) for alert in alerts])
    return frame.to_csv(index=False).encode("utf-8")


def assessment_summary_to_json_bytes(
    assessments: list[RiskAssessment],
    alerts: list[AlertRecord],
) -> bytes:
    payload = {
        "assessment_count": len(assessments),
        "alert_count": len(alerts),
        "top_region": assessments[0].region_id if assessments else None,
        "assessments": [asdict(assessment) for assessment in assessments],
        "alerts": [asdict(alert) for alert in alerts],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def forecast_summary_to_json_bytes(result: ForecastTrainingResult) -> bytes:
    payload = {
        "sample_count": result.sample_count,
        "active_sample_count": result.active_sample_count,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "validation_split_type": result.validation_split_type,
        "train_mae": result.train_mae,
        "train_rmse": result.train_rmse,
        "train_r2": result.train_r2,
        "validation_mae": result.validation_mae,
        "validation_rmse": result.validation_rmse,
        "validation_r2": result.validation_r2,
        "validation_active_mae": result.validation_active_mae,
        "validation_active_rmse": result.validation_active_rmse,
        "model_path": result.model_path,
        "predictions": [asdict(prediction) for prediction in result.predictions],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def lstm_forecast_summary_to_json_bytes(result: LSTMForecastTrainingResult) -> bytes:
    payload = {
        "sample_count": result.sample_count,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "mae": result.mae,
        "rmse": result.rmse,
        "r2": result.r2,
        "final_loss": result.losses[-1] if result.losses else None,
        "predictions": [asdict(prediction) for prediction in result.predictions],
    }
    return json.dumps(payload, indent=2).encode("utf-8")
