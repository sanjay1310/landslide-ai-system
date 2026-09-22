from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from landslide_ai.utils.optional_dependencies import shap_available


@dataclass(slots=True)
class FeatureImportanceRow:
    feature: str
    importance: float


def compute_global_feature_importance(
    estimator,
    feature_frame: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    feature_columns: list[str],
    top_k: int = 12,
) -> list[FeatureImportanceRow]:
    model = _unwrap_estimator(estimator)
    if hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
        return _rows_from_scores(feature_columns, importance, top_k=top_k)
    if hasattr(model, "feature_importances_"):
        importance = np.abs(np.asarray(model.feature_importances_, dtype=float).reshape(-1))
        return _rows_from_scores(feature_columns, importance, top_k=top_k)

    importance = permutation_importance(
        estimator,
        feature_frame[feature_columns],
        np.asarray(labels, dtype=int),
        n_repeats=5,
        random_state=42,
        scoring="average_precision",
    ).importances_mean
    return _rows_from_scores(feature_columns, importance, top_k=top_k)


def compute_shap_or_fallback_importance(
    estimator,
    feature_frame: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    feature_columns: list[str],
    top_k: int = 12,
) -> tuple[str, list[FeatureImportanceRow]]:
    if shap_available():
        try:
            import shap

            sample_frame = feature_frame[feature_columns].head(min(len(feature_frame), 200))
            explainer = shap.Explainer(estimator.predict_proba, sample_frame)
            values = explainer(sample_frame)
            if values.values.ndim == 3:
                importance = np.abs(values.values[:, :, 1]).mean(axis=0)
            else:
                importance = np.abs(values.values).mean(axis=0)
            return "shap", _rows_from_scores(feature_columns, importance, top_k=top_k)
        except Exception:
            pass
    return "model_native", compute_global_feature_importance(
        estimator,
        feature_frame=feature_frame,
        labels=labels,
        feature_columns=feature_columns,
        top_k=top_k,
    )


def explain_tabular_district_row(
    estimator,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    row_index: int,
    top_k: int = 5,
) -> dict[str, object]:
    model = _unwrap_estimator(estimator)
    row = feature_frame.iloc[row_index]
    if hasattr(model, "coef_"):
        transformed = _transform_features(estimator, feature_frame[feature_columns].iloc[[row_index]])
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        contributions = transformed.reshape(-1) * coefficients
        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        return {
            "method": "linear_contribution",
            "top_features": [feature_columns[index] for index in order],
            "top_feature_scores": [round(float(contributions[index]), 4) for index in order],
            "row_region_id": row.get("region_id", ""),
            "row_date": str(row.get("date", "")),
        }

    global_rows = compute_global_feature_importance(
        estimator,
        feature_frame=feature_frame,
        labels=np.zeros(len(feature_frame), dtype=int),
        feature_columns=feature_columns,
        top_k=top_k,
    )
    return {
        "method": "global_importance_proxy",
        "top_features": [item.feature for item in global_rows],
        "top_feature_scores": [round(float(item.importance), 4) for item in global_rows],
        "row_region_id": row.get("region_id", ""),
        "row_date": str(row.get("date", "")),
    }


def _rows_from_scores(
    feature_columns: list[str],
    scores: np.ndarray,
    top_k: int,
) -> list[FeatureImportanceRow]:
    order = np.argsort(np.abs(scores))[::-1][:top_k]
    return [
        FeatureImportanceRow(
            feature=feature_columns[index],
            importance=round(float(scores[index]), 6),
        )
        for index in order
    ]


def _unwrap_estimator(estimator):
    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.values())
        return steps[-1]
    return estimator


def _transform_features(estimator, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "named_steps"):
        transformed = frame.copy()
        for step_name, step in estimator.named_steps.items():
            if step_name == list(estimator.named_steps.keys())[-1]:
                break
            transformed = step.transform(transformed)
        return np.asarray(transformed, dtype=float)
    return np.asarray(frame, dtype=float)
