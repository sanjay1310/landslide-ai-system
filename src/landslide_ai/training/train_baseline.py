from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import platform

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import sklearn

from landslide_ai.agents.graph_agent import GraphAgent
from landslide_ai.agents.vision_agent import VisionAgent
from landslide_ai.data.loader import load_region_records, load_training_frame
from landslide_ai.models.trained_risk import TRAINED_RISK_FEATURE_COLUMNS, TrainedRiskModel
from landslide_ai.pipeline.feature_engineering import build_features


FEATURE_COLUMNS = TRAINED_RISK_FEATURE_COLUMNS


@dataclass(slots=True)
class TrainingResult:
    model_type: str
    sample_count: int
    feature_columns: list[str]
    train_accuracy: float
    validation_accuracy: float
    train_f1: float
    validation_f1: float
    model_path: str
    report_text: str


def _build_training_frame(csv_path: str) -> pd.DataFrame:
    frame = load_training_frame(csv_path)
    records = load_region_records(csv_path)
    graph_agent = GraphAgent()
    vision_agent = VisionAgent()
    graph_map = graph_agent.evaluate_network(records)

    feature_rows: list[dict[str, float]] = []
    for record in records:
        terrain_change_signal, vegetation_signal = vision_agent.evaluate(record)
        graph_insights = graph_map.get(record.region_id)
        graph_signal = graph_insights.local_influence if graph_insights else graph_agent.evaluate(record)
        propagated_signal = graph_insights.propagated_influence if graph_insights else graph_signal
        features = build_features(
            record=record,
            terrain_change_signal=terrain_change_signal,
            vegetation_signal=vegetation_signal,
            graph_signal=graph_signal,
        )
        feature_rows.append(
            TrainedRiskModel.build_feature_row(features, propagated_signal)
        )
    training_frame = pd.DataFrame(feature_rows)
    training_frame["label"] = frame["label"].astype(int).tolist()
    return training_frame


def train_random_forest_from_csv(
    csv_path: str,
    model_output_path: str = "artifacts/risk_model.pkl",
    model_type: str = "random_forest",
    validation_fraction: float = 0.25,
    random_state: int = 42,
) -> TrainingResult:
    frame = _build_training_frame(csv_path)
    x = frame[FEATURE_COLUMNS]
    y = frame["label"]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y,
    )

    model = TrainedRiskModel(TrainedRiskModel.create_estimator(model_type))
    assert model.estimator is not None
    model.estimator.fit(x_train, y_train)
    train_predictions = model.estimator.predict(x_train)
    val_predictions = model.estimator.predict(x_val)
    report = classification_report(y_val, val_predictions, zero_division=0)

    final_model = TrainedRiskModel(TrainedRiskModel.create_estimator(model_type))
    assert final_model.estimator is not None
    final_model.estimator.fit(x, y)
    metadata = {
        "model_type": model_type,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "label",
        "source_csv": csv_path,
        "sample_count": len(frame),
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "validation_accuracy": float(accuracy_score(y_val, val_predictions)),
        "train_f1": float(f1_score(y_train, train_predictions, zero_division=0)),
        "validation_f1": float(f1_score(y_val, val_predictions, zero_division=0)),
        "validation_fraction": validation_fraction,
        "random_state": random_state,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    final_model.save(model_output_path, metadata)

    return TrainingResult(
        model_type=model_type,
        sample_count=len(frame),
        feature_columns=FEATURE_COLUMNS,
        train_accuracy=float(accuracy_score(y_train, train_predictions)),
        validation_accuracy=float(accuracy_score(y_val, val_predictions)),
        train_f1=float(f1_score(y_train, train_predictions, zero_division=0)),
        validation_f1=float(f1_score(y_val, val_predictions, zero_division=0)),
        model_path=model_output_path,
        report_text=report,
    )


def preview_training_data(csv_path: str) -> pd.DataFrame:
    return _build_training_frame(csv_path).head()
