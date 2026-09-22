from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from landslide_ai.graph.gnn import build_graph_feature_frame
from landslide_ai.graph.spatiotemporal import build_spatiotemporal_dataset, SpatioTemporalGraphRiskModel
from landslide_ai.models.trained_risk import TRAINED_RISK_FEATURE_COLUMNS, TrainedRiskModel
from landslide_ai.research.fair_graph_benchmark import (
    _flatten_spatiotemporal_dataset,
    _load_validation_indices,
)
from landslide_ai.training.train_regional_real_model import _build_runtime_feature_frame


@dataclass(slots=True)
class ReliabilitySeries:
    model_name: str
    brier_score: float
    bin_true_rate: list[float]
    bin_pred_rate: list[float]


def generate_reliability_artifact(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    risk_model_path: str | Path = "artifacts/risk_model.pkl",
    st_artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    output_json_path: str | Path = "artifacts/reliability_curves.json",
) -> dict[str, object]:
    frame = pd.read_csv(dataset_csv_path)
    model = TrainedRiskModel()
    model.load(risk_model_path)
    flat_frame = _build_runtime_feature_frame(build_graph_feature_frame(frame.copy()))
    features = list(model.metadata.get("feature_columns", TRAINED_RISK_FEATURE_COLUMNS)) if model.metadata else list(TRAINED_RISK_FEATURE_COLUMNS)
    traditional_probs = model.estimator.predict_proba(flat_frame[features])[:, 1]
    traditional_series = _build_series(
        "traditional_runtime",
        y_true=frame["label"].astype(int).to_numpy(),
        probabilities=traditional_probs,
    )

    dataset = build_spatiotemporal_dataset(frame, sequence_length=10)
    validation_indices = _load_validation_indices(st_artifact_path)
    st_model = SpatioTemporalGraphRiskModel(sequence_length=10)
    st_model.load(st_artifact_path)
    st_probs, _ = st_model.predict_dataset(dataset, mc_passes=1)
    flat_dataset = _flatten_spatiotemporal_dataset(dataset)
    validation_mask = flat_dataset["sample_index"].isin(validation_indices).to_numpy()
    st_series = _build_series(
        "spatiotemporal_gnn",
        y_true=flat_dataset.loc[validation_mask, "label"].astype(int).to_numpy(),
        probabilities=st_probs.reshape(-1)[validation_mask],
    )

    payload = {"series": [asdict(traditional_series), asdict(st_series)]}
    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_series(model_name: str, y_true: np.ndarray, probabilities: np.ndarray) -> ReliabilitySeries:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    true_rate, pred_rate = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    brier = float(np.mean((probabilities - y_true) ** 2))
    return ReliabilitySeries(
        model_name=model_name,
        brier_score=brier,
        bin_true_rate=[round(float(value), 6) for value in true_rate],
        bin_pred_rate=[round(float(value), 6) for value in pred_rate],
    )
