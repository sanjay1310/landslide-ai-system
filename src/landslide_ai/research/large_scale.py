from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from landslide_ai.ingestion.risk_dataset import build_large_scale_risk_dataset
from landslide_ai.data.schemas import RegionRecord
from landslide_ai.graph.gnn import GraphNeuralRiskModel
from landslide_ai.graph.spatial import build_spatial_graph
from landslide_ai.research.temporal_models import (
    TemporalSequenceRiskClassifier,
    build_temporal_sequence_dataset,
    fit_probability_calibrator,
)
from landslide_ai.utils.optional_dependencies import (
    imblearn_available,
    lightgbm_available,
    torch_available,
    xgboost_available,
)


LARGE_SCALE_FEATURE_COLUMNS = [
    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_14d_mm",
    "rainfall_ratio_24h_to_7d",
    "rainfall_acceleration",
    "soil_wetness_index",
    "temperature_c",
    "neighbor_risk_mean",
    "neighbor_count",
    "elevation_m",
    "slope_deg",
    "ndvi",
    "vegetation_vulnerability",
    "terrain_rainfall_interaction",
    "soil_rainfall_interaction",
]

DERIVED_LARGE_SCALE_FEATURE_COLUMNS = [
    "rainfall_3d_to_14d_ratio",
    "rainfall_7d_to_14d_ratio",
    "rainfall_intensity_x_acceleration",
    "slope_soil_interaction",
    "neighbor_pressure",
    "elevation_slope_interaction",
    "temperature_rainfall_interaction",
    "ndvi_loss",
]

ALL_LARGE_SCALE_FEATURE_COLUMNS = LARGE_SCALE_FEATURE_COLUMNS + DERIVED_LARGE_SCALE_FEATURE_COLUMNS


@dataclass(slots=True)
class LargeScaleMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float


@dataclass(slots=True)
class ConfusionSummary:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass(slots=True)
class ModelComparisonRow:
    model_name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float


@dataclass(slots=True)
class LargeScaleEvaluationResult:
    dataset_rows: int
    train_rows: int
    validation_rows: int
    label_source: str
    model_name: str
    decision_threshold: float
    feature_columns: list[str]
    metrics: LargeScaleMetrics
    confusion: ConfusionSummary
    model_comparisons: list[ModelComparisonRow]
    state_summary: list[dict[str, object]]
    top_positive_windows: list[dict[str, object]]
    limitations: list[str]


def prepare_large_scale_dataset(
    rainfall_timeseries_csv_path: str = "data/raw/india_rainfall_timeseries.csv",
    region_metadata_csv_path: str = "data/raw/india_rainfall.csv",
    terrain_csv_path: str = "data/raw/india_terrain.csv",
    output_csv_path: str = "data/india/india_risk_timeseries.csv",
    landslide_inventory_csv_path: str | None = None,
) -> Path:
    return build_large_scale_risk_dataset(
        rainfall_timeseries_csv_path=rainfall_timeseries_csv_path,
        region_metadata_csv_path=region_metadata_csv_path,
        terrain_csv_path=terrain_csv_path,
        output_csv_path=output_csv_path,
        landslide_inventory_csv_path=landslide_inventory_csv_path,
    )


def _safe_roc_auc(y_true: pd.Series, probabilities: list[float]) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return 0.0


def augment_large_scale_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    rainfall_14d = working["rainfall_14d_mm"].astype(float).clip(lower=1.0)
    rainfall_24h = working["rainfall_24h_mm"].astype(float)
    rainfall_3d = working["rainfall_3d_mm"].astype(float)
    rainfall_7d = working["rainfall_7d_mm"].astype(float)
    soil = working["soil_wetness_index"].astype(float).clip(0.0, 1.0)
    slope = working["slope_deg"].astype(float).clip(lower=0.0)
    elevation = working["elevation_m"].astype(float).clip(lower=0.0)
    neighbor_mean = working["neighbor_risk_mean"].astype(float).clip(0.0, 1.0)
    neighbor_scale = (working["neighbor_count"].astype(float).clip(lower=0.0, upper=8.0) / 8.0).clip(0.0, 1.0)
    temperature = working["temperature_c"].astype(float)
    ndvi = working["ndvi"].astype(float).clip(0.0, 1.0)
    rainfall_acceleration = working["rainfall_acceleration"].astype(float)

    working["rainfall_3d_to_14d_ratio"] = (rainfall_3d / rainfall_14d).clip(0.0, 1.0)
    working["rainfall_7d_to_14d_ratio"] = (rainfall_7d / rainfall_14d).clip(0.0, 1.0)
    working["rainfall_intensity_x_acceleration"] = rainfall_24h * rainfall_acceleration.clip(lower=0.0)
    working["slope_soil_interaction"] = slope * soil
    working["neighbor_pressure"] = (neighbor_mean * (0.5 + neighbor_scale * 0.5)).clip(0.0, 1.0)
    working["elevation_slope_interaction"] = elevation * slope
    working["temperature_rainfall_interaction"] = temperature * rainfall_3d
    working["ndvi_loss"] = (1.0 - ndvi).clip(0.0, 1.0)
    return working


def temporal_split_by_region(
    frame: pd.DataFrame,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for _, group in working.sort_values(["region_id", "date"]).groupby("region_id", dropna=False):
        validation_count = max(1, int(round(len(group) * validation_fraction)))
        validation_part = group.iloc[-validation_count:]
        train_part = group.iloc[:-validation_count]
        if train_part.empty:
            train_part = group.iloc[:-1]
            validation_part = group.iloc[-1:]
        train_parts.append(train_part)
        validation_parts.append(validation_part)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
    )


def temporal_train_calibration_validation_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.20,
    calibration_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, validation_frame = temporal_split_by_region(frame, validation_fraction=validation_fraction)
    calibration_fraction = min(max(calibration_fraction, 0.05), 0.40)
    fit_frame, calibration_frame = temporal_split_by_region(train_frame, validation_fraction=calibration_fraction)
    if fit_frame.empty or calibration_frame.empty:
        return train_frame, train_frame.iloc[0:0].copy(), validation_frame
    return fit_frame, calibration_frame, validation_frame


def evaluate_large_scale_risk_dataset(
    dataset_csv_path: str = "data/india/india_risk_timeseries.csv",
    validation_fraction: float = 0.20,
    calibration_fraction: float = 0.20,
    include_sequence_models: bool = True,
    sequence_length: int = 7,
) -> LargeScaleEvaluationResult:
    frame = augment_large_scale_features(pd.read_csv(dataset_csv_path))
    train_frame, calibration_frame, validation_frame = temporal_train_calibration_validation_split(
        frame,
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
    )
    label_source = str(frame["label_source"].mode().iloc[0]) if "label_source" in frame.columns else "unknown"
    feature_columns = [column for column in ALL_LARGE_SCALE_FEATURE_COLUMNS if column in frame.columns]

    candidate_models = _build_candidate_models(
        label_source=label_source,
        include_sequence_models=include_sequence_models and label_source == "inventory",
    )
    comparison_rows: list[ModelComparisonRow] = []
    best_payload: tuple[str, float, list[float], list[int], LargeScaleMetrics, pd.DataFrame] | None = None

    for candidate in candidate_models:
        candidate_name = str(candidate["name"])
        if candidate["type"] == "sequence":
            sequence_payload = _evaluate_sequence_candidate(
                candidate_name=candidate_name,
                architecture=str(candidate["architecture"]),
                train_frame=train_frame,
                calibration_frame=calibration_frame,
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                sequence_length=sequence_length,
            )
            if sequence_payload is None:
                continue
            threshold, validation_probabilities, predictions, candidate_metrics, validation_rows = sequence_payload
        elif candidate["type"] == "gnn":
            gnn_payload = _evaluate_gnn_candidate(
                train_frame=train_frame,
                calibration_frame=calibration_frame,
                validation_frame=validation_frame,
            )
            if gnn_payload is None:
                continue
            threshold, validation_probabilities, predictions, candidate_metrics, validation_rows = gnn_payload
        else:
            threshold, validation_probabilities, predictions, candidate_metrics, validation_rows = _evaluate_tabular_candidate(
                candidate_name=candidate_name,
                estimator=candidate["builder"](),
                train_frame=train_frame,
                calibration_frame=calibration_frame,
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                calibration_method=candidate.get("calibration_method"),
                resampling_method=candidate.get("resampling_method"),
                sample_weight_strategy=candidate.get("sample_weight_strategy"),
            )
        comparison_rows.append(
            ModelComparisonRow(
                model_name=candidate_name,
                threshold=float(threshold),
                accuracy=candidate_metrics.accuracy,
                precision=candidate_metrics.precision,
                recall=candidate_metrics.recall,
                f1=candidate_metrics.f1,
                roc_auc=candidate_metrics.roc_auc,
                pr_auc=candidate_metrics.pr_auc,
                brier_score=candidate_metrics.brier_score,
            )
        )
        score = (candidate_metrics.f1, candidate_metrics.recall, candidate_metrics.pr_auc, candidate_metrics.roc_auc)
        if best_payload is None or score > (
            best_payload[4].f1,
            best_payload[4].recall,
            best_payload[4].pr_auc,
            best_payload[4].roc_auc,
        ):
            best_payload = (
                candidate_name,
                float(threshold),
                validation_probabilities,
                predictions,
                candidate_metrics,
                validation_rows,
            )

    assert best_payload is not None
    model_name, threshold, probabilities, predictions, metrics, best_validation_rows = best_payload

    tn, fp, fn, tp = confusion_matrix(best_validation_rows["label"].astype(int), predictions, labels=[0, 1]).ravel()
    confusion = ConfusionSummary(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )

    validation_output = best_validation_rows.copy()
    validation_output["predicted_probability"] = probabilities
    validation_output["predicted_label"] = predictions
    state_summary = (
        validation_output.groupby("state", dropna=False)
        .agg(
            {
                "label": "sum",
                "predicted_probability": "mean",
                "hazard_score": "mean",
                "sample_id": "count",
            }
        )
        .rename(
            columns={
                "label": "Positive Windows",
                "predicted_probability": "Mean Predicted Probability",
                "hazard_score": "Mean Hazard Score",
                "sample_id": "Window Count",
            }
        )
        .round(3)
        .reset_index()
        .sort_values("Mean Predicted Probability", ascending=False)
        .to_dict(orient="records")
    )
    top_windows = (
        validation_output.sort_values("predicted_probability", ascending=False)
        .head(12)[
            [
                "sample_id",
                "state",
                "district",
                "date",
                "predicted_probability",
                "label",
                "hazard_score",
                "rainfall_24h_mm",
                "rainfall_7d_mm",
            ]
        ]
        .assign(date=lambda frame: frame["date"].astype(str))
        .round(3)
        .to_dict(orient="records")
    )
    limitations = [
        "The temporal dataset is substantially larger than the original snapshot dataset and is better suited for time-aware evaluation.",
        "The real-data Kerala/Uttarakhand dataset is highly imbalanced, so accuracy alone can be misleading and recall-oriented metrics should be interpreted carefully.",
        "External validation can be improved further by refining event-to-district mapping, expanding terrain and vegetation features, and calibrating decision thresholds.",
    ]
    if label_source != "inventory":
        limitations.insert(
            0,
            "Labels are still proxy hazard labels derived from engineered hazard scores rather than a verified landslide event inventory.",
        )
    else:
        limitations.insert(
            0,
            "Labels are derived from a mapped external landslide inventory, but some raw locality-level event names may still introduce district-assignment noise.",
        )

    return LargeScaleEvaluationResult(
        dataset_rows=len(frame),
        train_rows=len(train_frame) + len(calibration_frame),
        validation_rows=len(validation_frame),
        label_source=label_source,
        model_name=model_name,
        decision_threshold=float(threshold),
        feature_columns=feature_columns,
        metrics=metrics,
        confusion=confusion,
        model_comparisons=comparison_rows,
        state_summary=state_summary,
        top_positive_windows=top_windows,
        limitations=limitations,
    )


def write_large_scale_evaluation(
    result: LargeScaleEvaluationResult,
    json_path: str | Path = "artifacts/large_scale_risk_evaluation.json",
    markdown_path: str | Path = "docs/LARGE_SCALE_RESEARCH_EVALUATION.md",
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_rows": result.dataset_rows,
        "train_rows": result.train_rows,
        "validation_rows": result.validation_rows,
        "label_source": result.label_source,
        "model_name": result.model_name,
        "decision_threshold": result.decision_threshold,
        "feature_columns": result.feature_columns,
        "metrics": asdict(result.metrics),
        "confusion": asdict(result.confusion),
        "model_comparisons": [asdict(item) for item in result.model_comparisons],
        "state_summary": result.state_summary,
        "top_positive_windows": result.top_positive_windows,
        "limitations": result.limitations,
    }
    json_target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_target.write_text(
        (
            "# Large-Scale Risk Evaluation\n\n"
            "## Dataset\n"
            f"- Total district-date rows: {result.dataset_rows}\n"
            f"- Train rows: {result.train_rows}\n"
            f"- Validation rows: {result.validation_rows}\n"
            f"- Label source: {result.label_source}\n"
            f"- Model: {result.model_name}\n"
            f"- Decision threshold: {result.decision_threshold:.3f}\n"
            f"- Feature count: {len(result.feature_columns)}\n\n"
            "## Temporal Evaluation Metrics\n"
            f"- Accuracy: {result.metrics.accuracy:.3f}\n"
            f"- Precision: {result.metrics.precision:.3f}\n"
            f"- Recall: {result.metrics.recall:.3f}\n"
            f"- F1: {result.metrics.f1:.3f}\n"
            f"- ROC-AUC: {result.metrics.roc_auc:.3f}\n\n"
            f"- PR-AUC: {result.metrics.pr_auc:.3f}\n"
            f"- Brier score: {result.metrics.brier_score:.3f}\n\n"
            "## Confusion Matrix\n"
            f"- True negative: {result.confusion.true_negative}\n"
            f"- False positive: {result.confusion.false_positive}\n"
            f"- False negative: {result.confusion.false_negative}\n"
            f"- True positive: {result.confusion.true_positive}\n\n"
            "## Model Comparison\n"
            f"{json.dumps([asdict(item) for item in result.model_comparisons], indent=2)}\n\n"
            "## State Summary\n"
            f"{json.dumps(result.state_summary[:10], indent=2)}\n\n"
            "## Top Positive Windows\n"
            f"{json.dumps(result.top_positive_windows[:12], indent=2)}\n\n"
            "## Limitations\n"
            + "\n".join([f"- {item}" for item in result.limitations])
            + "\n"
        ),
        encoding="utf-8",
    )


def _optimal_f1_threshold(y_true: pd.Series, probabilities: list[float]) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    best_threshold = 0.5
    best_f1 = -1.0
    for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds):
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return float(best_threshold)


def _compute_metrics(y_true: pd.Series, predictions: list[int], probabilities: list[float]) -> LargeScaleMetrics:
    return LargeScaleMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=_safe_roc_auc(y_true, probabilities),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
    )


def _evaluate_tabular_candidate(
    candidate_name: str,
    estimator,
    train_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    calibration_method: str | None = None,
    resampling_method: str | None = None,
    sample_weight_strategy: str | None = None,
) -> tuple[float, list[float], list[int], LargeScaleMetrics, pd.DataFrame]:
    x_train = train_frame[feature_columns].copy()
    y_train = train_frame["label"].astype(int).copy()
    if resampling_method:
        x_train, y_train = _resample_training_frame(x_train, y_train, method=resampling_method)
    fit_kwargs = {}
    if sample_weight_strategy == "balanced":
        fit_kwargs["sample_weight"] = _balanced_sample_weights(y_train)
    try:
        estimator.fit(x_train, y_train, **fit_kwargs)
    except TypeError:
        estimator.fit(x_train, y_train)

    threshold_labels = train_frame["label"].astype(int)
    threshold_probabilities = estimator.predict_proba(train_frame[feature_columns])[:, 1].tolist()
    validation_probabilities = estimator.predict_proba(validation_frame[feature_columns])[:, 1].tolist()

    if calibration_method and not calibration_frame.empty:
        calibration_probabilities = estimator.predict_proba(calibration_frame[feature_columns])[:, 1].tolist()
        calibrator = fit_probability_calibrator(
            calibration_frame["label"].astype(int),
            calibration_probabilities,
            method=calibration_method,
        )
        if calibrator is not None:
            threshold_labels = calibration_frame["label"].astype(int)
            threshold_probabilities = calibrator.transform(calibration_probabilities).tolist()
            validation_probabilities = calibrator.transform(validation_probabilities).tolist()

    threshold = _optimal_f1_threshold(threshold_labels, threshold_probabilities)
    predictions = [1 if score >= threshold else 0 for score in validation_probabilities]
    metrics = _compute_metrics(validation_frame["label"].astype(int), predictions, validation_probabilities)
    return float(threshold), validation_probabilities, predictions, metrics, validation_frame.copy()


def _evaluate_sequence_candidate(
    candidate_name: str,
    architecture: str,
    train_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[float, list[float], list[int], LargeScaleMetrics, pd.DataFrame] | None:
    if not torch_available():
        return None

    train_sequences = build_temporal_sequence_dataset(train_frame, feature_columns, sequence_length=sequence_length)
    calibration_sequences = build_temporal_sequence_dataset(
        calibration_frame,
        feature_columns,
        sequence_length=sequence_length,
    )
    validation_sequences = build_temporal_sequence_dataset(
        validation_frame,
        feature_columns,
        sequence_length=sequence_length,
    )
    if len(train_sequences.labels) == 0 or len(validation_sequences.labels) == 0:
        return None

    classifier = TemporalSequenceRiskClassifier(
        input_size=len(feature_columns),
        hidden_size=32,
        num_layers=1,
        dropout=0.10,
        architecture=architecture,
    )
    classifier.fit(train_sequences.sequences, train_sequences.labels, epochs=10, learning_rate=0.0015, batch_size=256)

    threshold_labels = pd.Series(train_sequences.labels)
    threshold_probabilities = classifier.predict_proba(train_sequences.sequences).tolist()
    if len(calibration_sequences.labels) > 0 and len(np.unique(calibration_sequences.labels)) > 1:
        threshold_labels = pd.Series(calibration_sequences.labels)
        threshold_probabilities = classifier.predict_proba(calibration_sequences.sequences).tolist()

    validation_probabilities = classifier.predict_proba(validation_sequences.sequences).tolist()
    threshold = _optimal_f1_threshold(threshold_labels, threshold_probabilities)
    predictions = [1 if score >= threshold else 0 for score in validation_probabilities]
    metrics = _compute_metrics(pd.Series(validation_sequences.labels), predictions, validation_probabilities)
    return float(threshold), validation_probabilities, predictions, metrics, validation_sequences.rows.copy()


def _evaluate_gnn_candidate(
    train_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> tuple[float, list[float], list[int], LargeScaleMetrics, pd.DataFrame] | None:
    if not torch_available():
        return None

    reduced_train = _reduce_temporal_graph_frame(train_frame)
    reduced_calibration = _reduce_temporal_graph_frame(calibration_frame)
    reduced_validation = _reduce_temporal_graph_frame(validation_frame)
    combined = pd.concat([reduced_train, reduced_calibration, reduced_validation], ignore_index=True).copy()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["region_id", "date"]).reset_index(drop=True)

    train_keys = {(str(row["sample_id"]), str(row["date"])) for _, row in reduced_train.assign(date=reduced_train["date"].astype(str)).iterrows()}
    calibration_keys = {(str(row["sample_id"]), str(row["date"])) for _, row in reduced_calibration.assign(date=reduced_calibration["date"].astype(str)).iterrows()}
    validation_keys = {(str(row["sample_id"]), str(row["date"])) for _, row in reduced_validation.assign(date=reduced_validation["date"].astype(str)).iterrows()}

    combined_date_keys = combined["date"].dt.strftime("%Y-%m-%d")
    train_indices = [
        index for index, (sample_id, date_text) in enumerate(zip(combined["sample_id"], combined_date_keys, strict=True))
        if (str(sample_id), str(date_text)) in train_keys
    ]
    calibration_indices = [
        index for index, (sample_id, date_text) in enumerate(zip(combined["sample_id"], combined_date_keys, strict=True))
        if (str(sample_id), str(date_text)) in calibration_keys
    ]
    validation_indices = [
        index for index, (sample_id, date_text) in enumerate(zip(combined["sample_id"], combined_date_keys, strict=True))
        if (str(sample_id), str(date_text)) in validation_keys
    ]
    if not train_indices or not validation_indices:
        return None

    records = _build_region_records_from_frame(combined)
    graph = build_spatial_graph(records)
    model = GraphNeuralRiskModel(hidden_dim=24)
    early_stop_indices = calibration_indices if calibration_indices else validation_indices
    result = model.fit(
        frame=combined,
        adjacency=graph.normalized_adjacency,
        epochs=80,
        learning_rate=0.01,
        validation_fraction=0.25,
        random_state=42,
        train_indices=train_indices,
        validation_indices=early_stop_indices,
    )
    probabilities = result["probabilities"].tolist()
    threshold_source_indices = calibration_indices if calibration_indices else train_indices
    threshold = _optimal_f1_threshold(
        combined.iloc[threshold_source_indices]["label"].astype(int),
        [probabilities[index] for index in threshold_source_indices],
    )
    validation_probabilities = [probabilities[index] for index in validation_indices]
    predictions = [1 if score >= threshold else 0 for score in validation_probabilities]
    validation_rows = combined.iloc[validation_indices].copy().reset_index(drop=True)
    metrics = _compute_metrics(validation_rows["label"].astype(int), predictions, validation_probabilities)
    return float(threshold), validation_probabilities, predictions, metrics, validation_rows


def benchmark_reduced_temporal_gnn(
    dataset_csv_path: str = "data/regional/kerala_uttarakhand_risk_timeseries.csv",
    validation_fraction: float = 0.20,
    calibration_fraction: float = 0.20,
    negative_stride: int = 30,
    hidden_dim: int = 16,
    epochs: int = 8,
    learning_rate: float = 0.01,
) -> dict[str, object]:
    if not torch_available():
        raise ImportError("PyTorch is not installed. Install `torch` to benchmark the graph GNN.")

    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    if hasattr(torch, "multiprocessing"):
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except (AttributeError, RuntimeError):
            pass

    frame = augment_large_scale_features(pd.read_csv(dataset_csv_path))
    train_frame, calibration_frame, validation_frame = temporal_train_calibration_validation_split(
        frame,
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
    )
    reduced_train = _reduce_temporal_graph_frame(train_frame, negative_stride=negative_stride)
    reduced_calibration = _reduce_temporal_graph_frame(calibration_frame, negative_stride=negative_stride)
    reduced_validation = _reduce_temporal_graph_frame(validation_frame, negative_stride=negative_stride)

    combined = pd.concat([reduced_train, reduced_calibration, reduced_validation], ignore_index=True).copy()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["region_id", "date"]).reset_index(drop=True)

    train_indices = _lookup_temporal_indices(combined, reduced_train)
    calibration_indices = _lookup_temporal_indices(combined, reduced_calibration)
    validation_indices = _lookup_temporal_indices(combined, reduced_validation)
    if not train_indices or not validation_indices:
        raise RuntimeError("Reduced temporal GNN benchmark could not build train/validation indices.")

    records = _build_region_records_from_frame(combined)
    graph = build_spatial_graph(records)
    model = GraphNeuralRiskModel(hidden_dim=hidden_dim)
    result = model.fit(
        combined,
        graph.normalized_adjacency,
        epochs=epochs,
        learning_rate=learning_rate,
        train_indices=train_indices,
        validation_indices=calibration_indices or validation_indices,
    )
    probabilities = result["probabilities"].tolist()
    threshold_source = calibration_indices if calibration_indices else train_indices
    threshold = _optimal_f1_threshold(
        combined.iloc[threshold_source]["label"].astype(int),
        [probabilities[index] for index in threshold_source],
    )
    validation_probabilities = [probabilities[index] for index in validation_indices]
    validation_rows = combined.iloc[validation_indices].copy().reset_index(drop=True)
    predictions = [1 if score >= threshold else 0 for score in validation_probabilities]
    metrics = _compute_metrics(validation_rows["label"].astype(int), predictions, validation_probabilities)
    return {
        "model_name": "graph_gnn_reduced_temporal",
        "threshold": float(threshold),
        "train_rows": len(train_indices),
        "calibration_rows": len(calibration_indices),
        "validation_rows": len(validation_indices),
        "negative_stride": negative_stride,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "metrics": asdict(metrics),
    }


def _reduce_temporal_graph_frame(frame: pd.DataFrame, negative_stride: int = 14) -> pd.DataFrame:
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    keep_parts: list[pd.DataFrame] = []
    for _, group in working.sort_values(["region_id", "date"]).groupby("region_id", dropna=False):
        positives = group[group["label"].astype(int) == 1]
        negatives = group[group["label"].astype(int) == 0]
        sampled_negatives = negatives.iloc[::negative_stride].copy()
        if not negatives.empty:
            sampled_negatives = (
                pd.concat([sampled_negatives, negatives.tail(1)], ignore_index=False)
                .drop_duplicates(subset=["sample_id"])
            )
        reduced_group = (
            pd.concat([positives, sampled_negatives], ignore_index=False)
            .drop_duplicates(subset=["sample_id"])
            .sort_values("date")
        )
        keep_parts.append(reduced_group)
    if not keep_parts:
        return working.iloc[0:0].copy()
    reduced = pd.concat(keep_parts, ignore_index=True)
    reduced["date"] = reduced["date"].dt.strftime("%Y-%m-%d")
    return reduced.reset_index(drop=True)


def _lookup_temporal_indices(combined: pd.DataFrame, subset: pd.DataFrame) -> list[int]:
    subset_copy = subset.copy()
    subset_copy["date"] = pd.to_datetime(subset_copy["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    subset_keys = {(str(row.sample_id), str(row.date)) for row in subset_copy.itertuples()}
    combined_dates = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return [
        index
        for index, (sample_id, date_text) in enumerate(zip(combined["sample_id"], combined_dates, strict=True))
        if (str(sample_id), str(date_text)) in subset_keys
    ]


def _build_region_records_from_frame(frame: pd.DataFrame) -> list[RegionRecord]:
    records: list[RegionRecord] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            RegionRecord(
                region_id=str(row["region_id"]),
                state=str(row["state"]),
                district=str(row["district"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                rainfall_24h_mm=float(row["rainfall_24h_mm"]),
                rainfall_7d_mm=float(row["rainfall_7d_mm"]),
                slope_deg=float(row["slope_deg"]),
                elevation_m=float(row["elevation_m"]),
                soil_wetness_index=float(row["soil_wetness_index"]),
                ndvi=float(row["ndvi"]),
                temperature_c=float(row["temperature_c"]),
                neighbor_risk_mean=float(row["neighbor_risk_mean"]),
                neighbor_count=int(row["neighbor_count"]),
            )
        )
    return records


def _build_candidate_models(label_source: str, include_sequence_models: bool = True) -> list[dict[str, object]]:
    if label_source == "inventory":
        candidates: list[dict[str, object]] = [
            {
                "name": "logistic_regression_balanced",
                "type": "tabular",
                "builder": lambda: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            },
            {
                "name": "logistic_regression_balanced_calibrated",
                "type": "tabular",
                "builder": lambda: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
                "calibration_method": "sigmoid",
            },
            {
                "name": "gradient_boosting",
                "type": "tabular",
                "builder": lambda: GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            },
            {
                "name": "gradient_boosting_calibrated",
                "type": "tabular",
                "builder": lambda: GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
                "calibration_method": "isotonic",
            },
            {
                "name": "graph_gnn",
                "type": "gnn",
            },
        ]
        if include_sequence_models:
            candidates.extend(
                [
                    {"name": "gru_sequence_classifier", "type": "sequence", "architecture": "gru"},
                    {"name": "lstm_sequence_classifier", "type": "sequence", "architecture": "lstm"},
                    {"name": "temporal_cnn_classifier", "type": "sequence", "architecture": "tcn"},
                    {"name": "temporal_transformer_classifier", "type": "sequence", "architecture": "transformer"},
                ]
            )
        candidates.extend(_optional_advanced_candidates())
        return candidates
    return [
        {
            "name": "gradient_boosting",
            "type": "tabular",
            "builder": lambda: GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        }
    ]


def _optional_advanced_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [
        {
            "name": "hist_gradient_boosting_cost_sensitive",
            "type": "tabular",
            "builder": lambda: HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=220,
                random_state=42,
            ),
            "sample_weight_strategy": "balanced",
        },
        {
            "name": "logistic_regression_smote",
            "type": "tabular",
            "builder": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
            "calibration_method": "sigmoid",
            "resampling_method": "smote",
        },
        {
            "name": "logistic_regression_random_oversample",
            "type": "tabular",
            "builder": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
            "calibration_method": "sigmoid",
            "resampling_method": "random_oversample",
        },
    ]

    if xgboost_available():
        from xgboost import XGBClassifier

        candidates.append(
            {
                "name": "xgboost_cost_sensitive",
                "type": "tabular",
                "builder": lambda: XGBClassifier(
                    n_estimators=260,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
                "sample_weight_strategy": "balanced",
            }
        )
    if lightgbm_available():
        from lightgbm import LGBMClassifier

        candidates.append(
            {
                "name": "lightgbm_cost_sensitive",
                "type": "tabular",
                "builder": lambda: LGBMClassifier(
                    n_estimators=260,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    is_unbalance=True,
                    verbose=-1,
                ),
            }
        )
    return candidates


def _resample_training_frame(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = y_train.astype(int)
    if labels.nunique() < 2:
        return x_train, labels
    if method == "smote" and imblearn_available():
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=42)
        resampled_x, resampled_y = sampler.fit_resample(x_train, labels)
        return pd.DataFrame(resampled_x, columns=x_train.columns), pd.Series(resampled_y)

    positive = x_train.loc[labels == 1]
    negative = x_train.loc[labels == 0]
    if positive.empty or negative.empty:
        return x_train, labels
    target_size = max(len(negative), len(positive))
    sampled_positive = positive.sample(n=target_size, replace=True, random_state=42)
    sampled_negative = negative.sample(n=target_size, replace=len(negative) < target_size, random_state=42)
    resampled_x = pd.concat([sampled_negative, sampled_positive], ignore_index=True)
    resampled_y = pd.Series([0] * len(sampled_negative) + [1] * len(sampled_positive), dtype=int)
    return resampled_x, resampled_y


def _balanced_sample_weights(labels: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=int)
    positive = max(int(values.sum()), 1)
    negative = max(int(len(values) - values.sum()), 1)
    positive_weight = negative / positive
    return np.where(values == 1, positive_weight, 1.0).astype(float)
