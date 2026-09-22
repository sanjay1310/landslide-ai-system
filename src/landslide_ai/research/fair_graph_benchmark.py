from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from landslide_ai.graph.gnn import GraphNeuralRiskModel, build_graph_feature_frame
from landslide_ai.graph.spatiotemporal import (
    SpatioTemporalGraphRiskModel,
    build_spatiotemporal_dataset,
)
from landslide_ai.graph.spatial import build_spatial_graph
from landslide_ai.models.trained_risk import TRAINED_RISK_FEATURE_COLUMNS, TrainedRiskModel
from landslide_ai.research.temporal_models import fit_probability_calibrator
from landslide_ai.training.train_regional_real_model import _build_runtime_feature_frame


@dataclass(slots=True)
class FairBenchmarkResult:
    output_path: Path
    row_count: int
    calibration_rows: int
    evaluation_rows: int


def generate_fair_graph_benchmark(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    st_artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    graph_artifact_path: str | Path = "artifacts/graph_gnn.pt",
    logistic_artifact_path: str | Path = "artifacts/risk_model.pkl",
    output_path: str | Path = "artifacts/fair_graph_sequence_benchmark.json",
) -> FairBenchmarkResult:
    frame = pd.read_csv(dataset_csv_path)
    dataset = build_spatiotemporal_dataset(frame)
    flat_frame = _flatten_spatiotemporal_dataset(dataset)
    if flat_frame.empty:
        raise ValueError("Spatio-temporal dataset is empty. No fair benchmark can be generated.")

    validation_indices = _load_validation_indices(st_artifact_path)
    validation_dates = [dataset.sample_dates[index] for index in validation_indices]
    split_index = max(1, len(validation_dates) // 2)
    calibration_dates = set(validation_dates[:split_index])
    evaluation_dates = set(validation_dates[split_index:])
    if not evaluation_dates:
        evaluation_dates = calibration_dates

    runtime_frame = _build_runtime_feature_frame(build_graph_feature_frame(flat_frame))
    logistic_model = TrainedRiskModel()
    logistic_model.load(logistic_artifact_path)
    selected_runtime_features = (
        list(logistic_model.metadata.get("feature_columns", TRAINED_RISK_FEATURE_COLUMNS))
        if logistic_model.metadata
        else list(TRAINED_RISK_FEATURE_COLUMNS)
    )
    logistic_probs = logistic_model.estimator.predict_proba(runtime_frame[selected_runtime_features])[:, 1]

    graph_probs = _predict_static_graph_same_scope(dataset, graph_artifact_path)
    st_probs = _predict_spatiotemporal_same_scope(dataset, st_artifact_path)

    benchmark_payload = {
        "scope": "Common district-date graph sequence evaluation",
        "node_count": int(len(dataset.region_ids)),
        "sample_dates": int(len(dataset.sample_dates)),
        "calibration_rows": int(flat_frame["sample_date"].isin(calibration_dates).sum()),
        "evaluation_rows": int(flat_frame["sample_date"].isin(evaluation_dates).sum()),
        "traditional_same_scope": _evaluate_model_rows(
            flat_frame,
            logistic_probs,
            calibration_dates,
            evaluation_dates,
            calibration_preference="sigmoid",
        ),
        "graph_gnn_same_scope": _evaluate_model_rows(
            flat_frame,
            graph_probs,
            calibration_dates,
            evaluation_dates,
            calibration_preference="isotonic",
        ),
        "spatiotemporal_gnn_same_scope": _evaluate_model_rows(
            flat_frame,
            st_probs,
            calibration_dates,
            evaluation_dates,
            calibration_preference="none",
        ),
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
    return FairBenchmarkResult(
        output_path=target,
        row_count=int(len(flat_frame)),
        calibration_rows=int(benchmark_payload["calibration_rows"]),
        evaluation_rows=int(benchmark_payload["evaluation_rows"]),
    )


def _flatten_spatiotemporal_dataset(dataset) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_index, sample_date in enumerate(dataset.sample_dates):
        for node_index, region_id in enumerate(dataset.region_ids):
            metadata_row = dataset.region_metadata.iloc[node_index]
            feature_values = dataset.sequences[sample_index, node_index, -1, :]
            row = {
                "region_id": region_id,
                "state": str(metadata_row["state"]),
                "district": str(metadata_row["district"]),
                "date": sample_date,
                "sample_date": sample_date,
                "label": int(dataset.labels[sample_index, node_index]),
                "sample_index": sample_index,
                "node_index": node_index,
                "latitude": float(metadata_row["latitude"]),
                "longitude": float(metadata_row["longitude"]),
            }
            for feature_name, feature_value in zip(dataset.feature_columns, feature_values, strict=True):
                row[feature_name] = float(feature_value)
            rows.append(row)
    return pd.DataFrame(rows)


def _load_validation_indices(st_artifact_path: str | Path) -> list[int]:
    metadata_path = Path(st_artifact_path).with_name(f"{Path(st_artifact_path).stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata.get("validation_indices", []))


def _predict_static_graph_same_scope(dataset, graph_artifact_path: str | Path) -> np.ndarray:
    model = GraphNeuralRiskModel()
    model.load(graph_artifact_path)
    if model.model is None or model.feature_mean is None or model.feature_std is None:
        raise RuntimeError("Static graph model is not loaded correctly.")
    feature_index = [dataset.feature_columns.index(column) for column in model.feature_columns]
    latest_features = dataset.sequences[:, :, -1, :][:, :, feature_index].astype("float32")
    scaled = (latest_features - model.feature_mean.reshape(1, 1, -1)) / model.feature_std.reshape(1, 1, -1)
    torch = model.torch
    x_tensor = torch.tensor(scaled, dtype=torch.float32)
    adjacency = torch.tensor(dataset.adjacency, dtype=torch.float32)
    model.model.eval()
    with torch.no_grad():
        hidden = torch.einsum("ij,bjf->bif", adjacency, x_tensor)
        hidden = model.model.conv1(hidden)
        hidden = torch.relu(hidden)
        hidden = torch.einsum("ij,bjh->bih", adjacency, hidden)
        logits = model.model.conv2(hidden).squeeze(-1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities.reshape(-1).astype("float32")


def _predict_spatiotemporal_same_scope(dataset, st_artifact_path: str | Path) -> np.ndarray:
    model = SpatioTemporalGraphRiskModel()
    model.load(st_artifact_path)
    probabilities, _ = model.predict_dataset(dataset, mc_passes=1)
    return probabilities.reshape(-1).astype("float32")


def _evaluate_model_rows(
    flat_frame: pd.DataFrame,
    probabilities: np.ndarray,
    calibration_dates: set[str],
    evaluation_dates: set[str],
    calibration_preference: str = "sigmoid",
) -> dict[str, object]:
    labels = flat_frame["label"].astype(int).to_numpy()
    calibration_mask = flat_frame["sample_date"].isin(calibration_dates).to_numpy()
    evaluation_mask = flat_frame["sample_date"].isin(evaluation_dates).to_numpy()
    calibration_labels = labels[calibration_mask]
    calibration_probs = probabilities[calibration_mask]
    eval_probs = probabilities[evaluation_mask]
    eval_labels = labels[evaluation_mask]
    calibration_method, calibrated_eval_probs, threshold = _calibrate_and_select_threshold(
        calibration_labels,
        calibration_probs,
        eval_probs,
        preference=calibration_preference,
    )
    eval_preds = (calibrated_eval_probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(eval_labels, eval_preds)),
        "precision": float(precision_score(eval_labels, eval_preds, zero_division=0)),
        "recall": float(recall_score(eval_labels, eval_preds, zero_division=0)),
        "f1": float(f1_score(eval_labels, eval_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(eval_labels, calibrated_eval_probs)) if len(np.unique(eval_labels)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(eval_labels, calibrated_eval_probs)) if len(np.unique(eval_labels)) > 1 else 0.0,
        "brier_score": float(brier_score_loss(eval_labels, calibrated_eval_probs)),
        "calibration_method": calibration_method,
        "positive_prediction_rate": float(eval_preds.mean()) if len(eval_preds) else 0.0,
    }


def _optimal_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-8)
    best_index = int(np.nanargmax(f1_scores))
    return float(np.clip(thresholds[best_index], 0.05, 0.95))


def _calibrate_and_select_threshold(
    calibration_labels: np.ndarray,
    calibration_probabilities: np.ndarray,
    evaluation_probabilities: np.ndarray,
    preference: str = "sigmoid",
) -> tuple[str, np.ndarray, float]:
    calibration_labels = np.asarray(calibration_labels, dtype=int)
    calibration_probabilities = np.asarray(calibration_probabilities, dtype=float)
    evaluation_probabilities = np.asarray(evaluation_probabilities, dtype=float)

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [("none", calibration_probabilities, evaluation_probabilities)]
    methods = [preference] if preference != "none" else []
    for method in ["sigmoid", "isotonic"]:
        if method not in methods and preference != "none":
            methods.append(method)
    for method in methods:
        calibrator = fit_probability_calibrator(calibration_labels, calibration_probabilities, method=method)
        if calibrator is None:
            continue
        candidates.append(
            (
                method,
                calibrator.transform(calibration_probabilities),
                calibrator.transform(evaluation_probabilities),
            )
        )

    best_method = "none"
    best_eval_probs = evaluation_probabilities
    best_threshold = _select_operating_threshold(calibration_labels, calibration_probabilities)
    best_score = float("-inf")
    for method, cal_probs, eval_probs in candidates:
        threshold = _select_operating_threshold(calibration_labels, cal_probs)
        cal_preds = (cal_probs >= threshold).astype(int)
        precision_value = float(precision_score(calibration_labels, cal_preds, zero_division=0))
        recall_value = float(recall_score(calibration_labels, cal_preds, zero_division=0))
        f1_value = float(f1_score(calibration_labels, cal_preds, zero_division=0))
        positive_rate = float(cal_preds.mean()) if len(cal_preds) else 0.0
        prevalence = float(calibration_labels.mean()) if len(calibration_labels) else 0.0
        score = (
            2.8 * f1_value
            + 0.7 * precision_value
            + 0.3 * recall_value
            + 0.4 * float(average_precision_score(calibration_labels, cal_probs))
            - 0.08 * abs(positive_rate - max(prevalence * 2.0, 0.01))
        )
        if score > best_score:
            best_score = score
            best_method = method
            best_eval_probs = eval_probs
            best_threshold = threshold
    return best_method, best_eval_probs, best_threshold


def _select_operating_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    prevalence = float(labels.mean()) if len(labels) else 0.0
    min_positive_rate = max(prevalence * 0.75, 0.0025)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    candidates: list[float] = [0.05, 0.10, 0.15, 0.20]
    candidates.extend(float(t) for t in thresholds)
    if len(probs) > 0:
        for positive_rate in {min_positive_rate, max(prevalence, 0.005), max(prevalence * 2.0, 0.01)}:
            quantile = max(0.0, min(1.0, 1.0 - positive_rate))
            candidates.append(float(np.quantile(probs, quantile)))
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in sorted({float(np.clip(value, 0.01, 0.95)) for value in candidates}):
        preds = (probs >= threshold).astype(int)
        positive_rate = float(preds.mean()) if len(preds) else 0.0
        precision_value = float(precision_score(labels, preds, zero_division=0))
        recall_value = float(recall_score(labels, preds, zero_division=0))
        f1_value = float(f1_score(labels, preds, zero_division=0))
        if positive_rate == 0.0:
            score = -1.0
        else:
            target_rate = max(prevalence * 2.0, min_positive_rate, 0.01)
            rate_penalty = abs(positive_rate - target_rate)
            score = (3.0 * f1_value) + (0.7 * precision_value) + (0.3 * recall_value) - (0.12 * rate_penalty)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)
