from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from landslide_ai.agents.graph_agent import GraphAgent
from landslide_ai.agents.vision_agent import VisionAgent
from landslide_ai.data.loader import load_region_records, load_training_frame
from landslide_ai.models.deep_risk import DeepRiskModel, torch_available
from landslide_ai.models.trained_risk import TRAINED_RISK_FEATURE_COLUMNS
from landslide_ai.pipeline.feature_engineering import build_features


@dataclass(slots=True)
class DeepRiskTrainingResult:
    sample_count: int
    feature_columns: list[str]
    train_accuracy: float
    validation_accuracy: float
    train_f1: float
    validation_f1: float
    validation_precision: float
    validation_recall: float
    validation_roc_auc: float
    epochs: int
    model_path: str
    losses: list[float]


def _build_training_frame(csv_path: str) -> pd.DataFrame:
    training_frame = load_training_frame(csv_path)
    records = load_region_records(csv_path)
    graph_agent = GraphAgent()
    vision_agent = VisionAgent()
    graph_map = graph_agent.evaluate_network(records)
    rows: list[dict[str, float]] = []
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
        row = DeepRiskModel.build_feature_row(features, propagated_signal)
        row["label"] = int(training_frame.loc[training_frame["region_id"] == record.region_id, "label"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def train_deep_risk_model_from_csv(
    csv_path: str,
    model_output_path: str = "artifacts/deep_risk_model.pt",
    validation_fraction: float = 0.25,
    random_state: int = 42,
    epochs: int = 180,
    learning_rate: float = 0.003,
) -> DeepRiskTrainingResult:
    if not torch_available():
        raise ImportError("PyTorch is not installed. Install `torch` to train the deep risk model.")

    frame = _build_training_frame(csv_path)
    x = frame[TRAINED_RISK_FEATURE_COLUMNS]
    y = frame["label"].astype(int)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y,
    )

    train_frame = x_train.copy()
    train_frame["label"] = y_train.values
    model = DeepRiskModel()
    losses = model.fit(
        train_frame,
        feature_columns=TRAINED_RISK_FEATURE_COLUMNS,
        label_column="label",
        epochs=epochs,
        learning_rate=learning_rate,
    )

    train_prob = model.predict_proba_from_frame(x_train).tolist()
    val_prob = model.predict_proba_from_frame(x_val).tolist()
    train_pred = [1 if score >= 0.5 else 0 for score in train_prob]
    val_pred = [1 if score >= 0.5 else 0 for score in val_prob]
    model.save(model_output_path)

    return DeepRiskTrainingResult(
        sample_count=len(frame),
        feature_columns=list(TRAINED_RISK_FEATURE_COLUMNS),
        train_accuracy=float(accuracy_score(y_train, train_pred)),
        validation_accuracy=float(accuracy_score(y_val, val_pred)),
        train_f1=float(f1_score(y_train, train_pred, zero_division=0)),
        validation_f1=float(f1_score(y_val, val_pred, zero_division=0)),
        validation_precision=float(precision_score(y_val, val_pred, zero_division=0)),
        validation_recall=float(recall_score(y_val, val_pred, zero_division=0)),
        validation_roc_auc=float(roc_auc_score(y_val, val_prob)) if len(set(y_val.tolist())) > 1 else 0.0,
        epochs=epochs,
        model_path=model_output_path,
        losses=losses,
    )
