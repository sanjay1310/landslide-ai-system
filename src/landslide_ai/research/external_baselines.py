from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from landslide_ai.research.large_scale import (
    ALL_LARGE_SCALE_FEATURE_COLUMNS,
    ModelComparisonRow,
    _evaluate_tabular_candidate,
    augment_large_scale_features,
    temporal_train_calibration_validation_split,
)
from landslide_ai.utils.optional_dependencies import lightgbm_available, xgboost_available


def run_external_baseline_comparison(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    output_json_path: str | Path = "artifacts/external_baseline_comparison.json",
) -> list[ModelComparisonRow]:
    frame = augment_large_scale_features(pd.read_csv(dataset_csv_path))
    train_frame, calibration_frame, validation_frame = temporal_train_calibration_validation_split(frame)
    feature_columns = [column for column in ALL_LARGE_SCALE_FEATURE_COLUMNS if column in frame.columns]

    candidates: list[tuple[str, object, dict[str, object]]] = [
        (
            "logistic_regression_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=42),
            ),
            {"calibration_method": "sigmoid"},
        ),
        (
            "hist_gradient_boosting_cost_sensitive",
            HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=220, random_state=42),
            {"sample_weight_strategy": "balanced"},
        ),
    ]

    if xgboost_available():
        from xgboost import XGBClassifier

        candidates.append(
            (
                "xgboost_cost_sensitive",
                XGBClassifier(
                    n_estimators=260,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
                {"sample_weight_strategy": "balanced"},
            )
        )
    if lightgbm_available():
        from lightgbm import LGBMClassifier

        candidates.append(
            (
                "lightgbm_cost_sensitive",
                LGBMClassifier(
                    n_estimators=260,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    is_unbalance=True,
                    verbose=-1,
                ),
                {},
            )
        )

    rows: list[ModelComparisonRow] = []
    for name, estimator, options in candidates:
        threshold, probabilities, predictions, metrics, _ = _evaluate_tabular_candidate(
            candidate_name=name,
            estimator=estimator,
            train_frame=train_frame,
            calibration_frame=calibration_frame,
            validation_frame=validation_frame,
            feature_columns=feature_columns,
            calibration_method=options.get("calibration_method"),
            resampling_method=options.get("resampling_method"),
            sample_weight_strategy=options.get("sample_weight_strategy"),
        )
        del probabilities, predictions
        rows.append(
            ModelComparisonRow(
                model_name=name,
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
