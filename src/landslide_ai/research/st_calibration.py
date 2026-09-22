from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from landslide_ai.graph.spatiotemporal import SpatioTemporalGraphRiskModel, build_spatiotemporal_dataset
from landslide_ai.research.fair_graph_benchmark import _flatten_spatiotemporal_dataset, _load_validation_indices
from landslide_ai.research.temporal_models import fit_probability_calibrator


@dataclass(slots=True)
class CalibrationRow:
    method: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float


def generate_st_calibration_artifact(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    st_artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    output_json_path: str | Path = "artifacts/st_gnn_calibration.json",
) -> list[CalibrationRow]:
    frame = pd.read_csv(dataset_csv_path)
    dataset = build_spatiotemporal_dataset(frame, sequence_length=10)
    model = SpatioTemporalGraphRiskModel(sequence_length=10)
    model.load(st_artifact_path)
    probabilities, _ = model.predict_dataset(dataset, mc_passes=1)
    flat = _flatten_spatiotemporal_dataset(dataset)
    validation_indices = _load_validation_indices(st_artifact_path)
    validation_dates = [dataset.sample_dates[index] for index in validation_indices]
    split = max(1, len(validation_dates) // 2)
    calibration_dates = set(validation_dates[:split])
    evaluation_dates = set(validation_dates[split:]) or set(validation_dates[:split])
    calibration_mask = flat["sample_date"].isin(calibration_dates).to_numpy()
    evaluation_mask = flat["sample_date"].isin(evaluation_dates).to_numpy()

    y_cal = flat.loc[calibration_mask, "label"].astype(int).to_numpy()
    y_eval = flat.loc[evaluation_mask, "label"].astype(int).to_numpy()
    p_all = probabilities.reshape(-1)
    p_cal = p_all[calibration_mask]
    p_eval = p_all[evaluation_mask]

    rows = [
        _evaluate("none", y_cal, y_eval, p_cal, p_eval),
    ]
    for method in ["sigmoid", "isotonic"]:
        calibrator = fit_probability_calibrator(y_cal, p_cal, method=method)
        if calibrator is None:
            continue
        rows.append(_evaluate(method, y_cal, y_eval, calibrator.transform(p_cal), calibrator.transform(p_eval)))

    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(item) for item in rows], indent=2), encoding="utf-8")
    return rows


def _evaluate(method: str, y_cal: np.ndarray, y_eval: np.ndarray, p_cal: np.ndarray, p_eval: np.ndarray) -> CalibrationRow:
    threshold = _best_threshold(y_cal, p_cal)
    preds = (p_eval >= threshold).astype(int)
    return CalibrationRow(
        method=method,
        threshold=float(threshold),
        precision=float(precision_score(y_eval, preds, zero_division=0)),
        recall=float(recall_score(y_eval, preds, zero_division=0)),
        f1=float(f1_score(y_eval, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_eval, p_eval)) if len(np.unique(y_eval)) > 1 else 0.0,
        pr_auc=float(average_precision_score(y_eval, p_eval)) if len(np.unique(y_eval)) > 1 else 0.0,
        brier_score=float(brier_score_loss(y_eval, p_eval)),
    )


def _best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 37):
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
