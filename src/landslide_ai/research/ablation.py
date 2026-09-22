from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from landslide_ai.research.large_scale import (
    ALL_LARGE_SCALE_FEATURE_COLUMNS,
    LargeScaleMetrics,
    _evaluate_tabular_candidate,
    augment_large_scale_features,
    temporal_train_calibration_validation_split,
)


ABLATION_GROUPS = {
    "full_system": [],
    "without_graph": ["neighbor_risk_mean", "neighbor_count", "neighbor_pressure"],
    "without_temporal": [
        "rainfall_3d_mm",
        "rainfall_7d_mm",
        "rainfall_14d_mm",
        "rainfall_ratio_24h_to_7d",
        "rainfall_acceleration",
        "rainfall_3d_to_14d_ratio",
        "rainfall_7d_to_14d_ratio",
        "rainfall_intensity_x_acceleration",
        "temperature_rainfall_interaction",
    ],
    "without_vegetation": ["ndvi", "ndvi_loss", "vegetation_vulnerability"],
}


@dataclass(slots=True)
class AblationRow:
    ablation_name: str
    feature_count: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float


def run_ablation_study(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    output_json_path: str | Path = "artifacts/ablation_study.json",
) -> list[AblationRow]:
    frame = augment_large_scale_features(pd.read_csv(dataset_csv_path))
    train_frame, calibration_frame, validation_frame = temporal_train_calibration_validation_split(frame)

    rows: list[AblationRow] = []
    for ablation_name, excluded in ABLATION_GROUPS.items():
        feature_columns = [
            column
            for column in ALL_LARGE_SCALE_FEATURE_COLUMNS
            if column in frame.columns and column not in excluded
        ]
        threshold, _, _, metrics, _ = _evaluate_tabular_candidate(
            candidate_name=f"logistic_{ablation_name}",
            estimator=make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
            train_frame=train_frame,
            calibration_frame=calibration_frame,
            validation_frame=validation_frame,
            feature_columns=feature_columns,
            calibration_method="sigmoid",
        )
        rows.append(
            AblationRow(
                ablation_name=ablation_name,
                feature_count=len(feature_columns),
                threshold=float(threshold),
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                roc_auc=metrics.roc_auc,
                pr_auc=metrics.pr_auc,
                brier_score=metrics.brier_score,
            )
        )

    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(item) for item in rows], indent=2), encoding="utf-8")
    return rows
