from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from landslide_ai.research.ablation import ABLATION_GROUPS
from landslide_ai.research.explainability import (
    compute_global_feature_importance,
    explain_tabular_district_row,
)
from landslide_ai.research.large_scale import _build_candidate_models


def test_ablation_groups_cover_requested_components() -> None:
    assert "without_graph" in ABLATION_GROUPS
    assert "without_temporal" in ABLATION_GROUPS
    assert "without_vegetation" in ABLATION_GROUPS


def test_candidate_models_include_new_sequence_baselines() -> None:
    candidates = _build_candidate_models(label_source="inventory", include_sequence_models=True)
    names = {item["name"] for item in candidates}
    assert "temporal_cnn_classifier" in names
    assert "temporal_transformer_classifier" in names
    assert "logistic_regression_random_oversample" in names


def test_linear_explainability_outputs_sorted_feature_rows() -> None:
    frame = pd.DataFrame(
        {
            "rainfall_24h_mm": [10.0, 20.0, 30.0, 40.0],
            "soil_wetness_index": [0.2, 0.3, 0.8, 0.9],
            "ndvi": [0.7, 0.6, 0.4, 0.3],
            "label": [0, 0, 1, 1],
            "region_id": ["A", "B", "C", "D"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )
    features = ["rainfall_24h_mm", "soil_wetness_index", "ndvi"]
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, solver="liblinear", random_state=42),
    )
    estimator.fit(frame[features], frame["label"])

    importance = compute_global_feature_importance(
        estimator,
        feature_frame=frame,
        labels=frame["label"],
        feature_columns=features,
        top_k=2,
    )
    assert len(importance) == 2
    assert importance[0].importance >= importance[1].importance

    district = explain_tabular_district_row(
        estimator,
        feature_frame=frame,
        feature_columns=features,
        row_index=2,
        top_k=2,
    )
    assert district["method"] == "linear_contribution"
    assert len(district["top_features"]) == 2
    assert district["row_region_id"] == "C"
