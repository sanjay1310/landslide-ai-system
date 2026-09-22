from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - older scikit-learn
    FrozenEstimator = None

from landslide_ai.mlops import write_experiment_log
from landslide_ai.models.trained_risk import TrainedRiskModel
from landslide_ai.research.large_scale import (
    _optimal_f1_threshold,
    augment_large_scale_features,
    temporal_train_calibration_validation_split,
)


REGIONAL_RUNTIME_FEATURE_COLUMNS = [
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
class RegionalRealTrainingResult:
    model_path: Path
    metadata_path: Path
    sample_count: int
    train_rows: int
    validation_rows: int
    roc_auc: float
    pr_auc: float
    brier_score: float
    decision_threshold: float


def train_regional_real_runtime_model(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_real.csv",
    model_output_path: str | Path = "artifacts/risk_model.pkl",
) -> RegionalRealTrainingResult:
    frame = augment_large_scale_features(pd.read_csv(dataset_csv_path))
    runtime_frame = _build_runtime_feature_frame(frame)
    train_frame, calibration_frame, validation_frame = temporal_train_calibration_validation_split(
        runtime_frame,
        validation_fraction=0.20,
        calibration_fraction=0.20,
    )

    base_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
    )
    base_estimator.fit(train_frame[REGIONAL_RUNTIME_FEATURE_COLUMNS], train_frame["label"].astype(int))

    if len(calibration_frame["label"].astype(int).unique()) > 1:
        if FrozenEstimator is not None:
            calibration_estimator = FrozenEstimator(base_estimator)
            calibration_cv: str | int | None = None
        else:
            calibration_estimator = base_estimator
            calibration_cv = "prefit"

        trained_estimator = CalibratedClassifierCV(
            estimator=calibration_estimator,
            method="sigmoid",
            cv=calibration_cv,
        )
        trained_estimator.fit(
            calibration_frame[REGIONAL_RUNTIME_FEATURE_COLUMNS],
            calibration_frame["label"].astype(int),
        )
        calibration_probabilities = trained_estimator.predict_proba(
            calibration_frame[REGIONAL_RUNTIME_FEATURE_COLUMNS]
        )[:, 1].tolist()
        decision_threshold = _optimal_f1_threshold(calibration_frame["label"].astype(int), calibration_probabilities)
        calibration_method = "sigmoid"
    else:
        trained_estimator = base_estimator
        calibration_probabilities = trained_estimator.predict_proba(
            train_frame[REGIONAL_RUNTIME_FEATURE_COLUMNS]
        )[:, 1].tolist()
        decision_threshold = _optimal_f1_threshold(train_frame["label"].astype(int), calibration_probabilities)
        calibration_method = "none"

    validation_probabilities = trained_estimator.predict_proba(
        validation_frame[REGIONAL_RUNTIME_FEATURE_COLUMNS]
    )[:, 1].tolist()

    model = TrainedRiskModel(trained_estimator)
    metadata = {
        "model_type": "regional_logistic_regression_balanced_calibrated",
        "feature_columns": REGIONAL_RUNTIME_FEATURE_COLUMNS,
        "target_column": "label",
        "source_csv": str(dataset_csv_path),
        "sample_count": len(runtime_frame),
        "train_rows": len(train_frame) + len(calibration_frame),
        "validation_rows": len(validation_frame),
        "label_source": "inventory",
        "region_scope": ["Kerala", "Uttarakhand"],
        "roc_auc": float(roc_auc_score(validation_frame["label"].astype(int), validation_probabilities)),
        "pr_auc": float(average_precision_score(validation_frame["label"].astype(int), validation_probabilities)),
        "brier_score": float(brier_score_loss(validation_frame["label"].astype(int), validation_probabilities)),
        "decision_threshold": float(decision_threshold),
        "calibration_method": calibration_method,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "risk_bands": {
            "low_max": 0.20,
            "watch_max": 0.45,
            "elevated_max": 0.70,
            "severe_min": 0.70,
        },
    }
    model.save(model_output_path, metadata)
    metadata_path = Path(model_output_path).with_name(f"{Path(model_output_path).stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_experiment_log(
        experiment_name="regional_real_runtime_model",
        parameters={
            "dataset_csv_path": str(dataset_csv_path),
            "model_output_path": str(model_output_path),
            "feature_columns": REGIONAL_RUNTIME_FEATURE_COLUMNS,
        },
        metrics={
            "roc_auc": metadata["roc_auc"],
            "pr_auc": metadata["pr_auc"],
            "brier_score": metadata["brier_score"],
            "decision_threshold": metadata["decision_threshold"],
            "sample_count": metadata["sample_count"],
            "train_rows": metadata["train_rows"],
            "validation_rows": metadata["validation_rows"],
        },
        artifacts={
            "model_path": str(model_output_path),
            "metadata_path": str(metadata_path),
        },
    )
    return RegionalRealTrainingResult(
        model_path=Path(model_output_path),
        metadata_path=metadata_path,
        sample_count=len(runtime_frame),
        train_rows=len(train_frame) + len(calibration_frame),
        validation_rows=len(validation_frame),
        roc_auc=float(roc_auc_score(validation_frame["label"].astype(int), validation_probabilities)),
        pr_auc=float(average_precision_score(validation_frame["label"].astype(int), validation_probabilities)),
        brier_score=float(brier_score_loss(validation_frame["label"].astype(int), validation_probabilities)),
        decision_threshold=float(decision_threshold),
    )


def _build_runtime_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["rainfall_intensity"] = working["rainfall_24h_mm"].astype(float).clip(0.0, 250.0) / 250.0
    working["antecedent_rainfall"] = working["rainfall_7d_mm"].astype(float).clip(0.0, 600.0) / 600.0
    working["slope_factor"] = working["slope_deg"].astype(float).clip(0.0, 50.0) / 50.0
    working["soil_wetness"] = working["soil_wetness_index"].astype(float).clip(0.0, 1.0)
    working["vegetation_stress"] = (1.0 - working["ndvi"].astype(float).clip(0.0, 1.0)).clip(0.0, 1.0)
    working["graph_influence"] = working["neighbor_risk_mean"].astype(float).clip(0.0, 1.0)
    working["propagated_graph_signal"] = (
        working["neighbor_risk_mean"].astype(float).clip(0.0, 1.0)
        * (0.55 + (working["neighbor_count"].astype(float).clip(upper=8.0) / 8.0) * 0.45)
    ).clip(0.0, 1.0)
    working["rainfall_terrain_interaction"] = (working["rainfall_intensity"] * working["slope_factor"]).clip(0.0, 1.0)
    working["rainfall_soil_interaction"] = (working["antecedent_rainfall"] * working["soil_wetness"]).clip(0.0, 1.0)
    working["vegetation_soil_interaction"] = (working["vegetation_stress"] * working["soil_wetness"]).clip(0.0, 1.0)
    working["graph_pressure"] = (working["graph_influence"] * working["propagated_graph_signal"]).clip(0.0, 1.0)
    working["terrain_wetness_interaction"] = (working["slope_factor"] * working["soil_wetness"]).clip(0.0, 1.0)
    return working[REGIONAL_RUNTIME_FEATURE_COLUMNS + ["label", "region_id", "date"]].copy()
