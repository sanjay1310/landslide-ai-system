from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

from landslide_ai.graph.gnn import (
    predict_graph_probabilities_for_csv,
    train_graph_model_from_csv,
)


# ==============================
# 📦 Result Dataclass
# ==============================
@dataclass(slots=True)
class GraphTrainingResult:
    sample_count: int
    feature_columns: List[str]
    artifact_path: str

    # Metrics
    accuracy: float
    validation_accuracy: float
    f1: float
    validation_f1: float
    roc_auc: float

    # Reports
    report_text: str

    # Training curves
    losses: List[float]
    val_losses: List[float]

    # Predictions
    predictions: pd.DataFrame

    # Threshold used
    threshold: float


# ==============================
# 🎯 Threshold Optimization
# ==============================
def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    return thresholds[max(best_idx - 1, 0)]


# ==============================
# 🚀 Main Training Function
# ==============================
def train_graph_risk_model(
    csv_path: str,
    artifact_path: str = "artifacts/graph_gnn.pt",
    epochs: int = 300,
    learning_rate: float = 0.02,
    validation_fraction: float = 0.25,
    random_state: int = 42,
    optimize_threshold: bool = True,
    manual_threshold: Optional[float] = None,
    save_predictions_path: Optional[str] = None,
) -> GraphTrainingResult:
    """
    Train Graph Neural Network for Landslide Risk Prediction
    """

    # ==============================
    # 🔥 Train Model
    # ==============================
    result = train_graph_model_from_csv(
        csv_path=csv_path,
        artifact_path=artifact_path,
        epochs=epochs,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    # ==============================
    # 🔮 Predict Probabilities
    # ==============================
    prediction_frame = predict_graph_probabilities_for_csv(
        csv_path, artifact_path
    )

    # ==============================
    # 📊 Assign Splits
    # ==============================
    prediction_frame["split"] = "train"
    prediction_frame.loc[result["validation_indices"], "split"] = "validation"

    val_df = prediction_frame[prediction_frame["split"] == "validation"]

    # ==============================
    # 🎯 Threshold Selection
    # ==============================
    if manual_threshold is not None:
        threshold = manual_threshold
    elif optimize_threshold:
        threshold = find_best_threshold(
            val_df["label"].values,
            val_df["graph_gnn_probability"].values,
        )
    else:
        threshold = 0.5

    # ==============================
    # 🏷️ Convert to Labels
    # ==============================
    prediction_frame["predicted_label"] = (
        prediction_frame["graph_gnn_probability"] >= threshold
    ).astype(int)

    val_df = prediction_frame[prediction_frame["split"] == "validation"]

    # ==============================
    # 📈 Metrics
    # ==============================
    validation_f1 = f1_score(
        val_df["label"],
        val_df["predicted_label"],
        zero_division=0,
    )

    overall_f1 = f1_score(
        prediction_frame["label"],
        prediction_frame["predicted_label"],
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        val_df["label"],
        val_df["graph_gnn_probability"],
    )

    report = classification_report(
        val_df["label"],
        val_df["predicted_label"],
        zero_division=0,
    )

    # ==============================
    # 💾 Save Predictions (Optional)
    # ==============================
    if save_predictions_path:
        prediction_frame.to_csv(save_predictions_path, index=False)

    # ==============================
    # 📦 Return Result
    # ==============================
    return GraphTrainingResult(
        sample_count=len(prediction_frame),
        feature_columns=list(result["feature_columns"]),
        artifact_path=str(result["artifact_path"]),
        accuracy=float(result["accuracy"]),
        validation_accuracy=float(result["validation_accuracy"]),
        f1=float(overall_f1),
        validation_f1=float(validation_f1),
        roc_auc=float(roc_auc),
        report_text=report,
        losses=list(result["losses"]),
        val_losses=list(result["val_losses"]),
        predictions=prediction_frame,
        threshold=float(threshold),
    )