from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from landslide_ai.graph.spatiotemporal import run_spatiotemporal_inference_from_frame
from landslide_ai.research.ablation import run_ablation_study
from landslide_ai.research.explainability import (
    compute_shap_or_fallback_importance,
    explain_tabular_district_row,
)
from landslide_ai.research.large_scale import (
    ALL_LARGE_SCALE_FEATURE_COLUMNS,
    augment_large_scale_features,
    evaluate_large_scale_risk_dataset,
    temporal_train_calibration_validation_split,
)


def run_advanced_research_suite(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    spatiotemporal_artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    output_json_path: str | Path = "artifacts/advanced_research_suite.json",
    output_markdown_path: str | Path = "docs/ADVANCED_RESEARCH_SUITE.md",
) -> dict[str, object]:
    evaluation = evaluate_large_scale_risk_dataset(
        dataset_csv_path=str(dataset_csv_path),
        include_sequence_models=True,
        sequence_length=10,
    )
    ablations = run_ablation_study(dataset_csv_path=dataset_csv_path)

    frame = augment_large_scale_features(pd.read_csv(dataset_csv_path))
    train_frame, _, validation_frame = temporal_train_calibration_validation_split(frame)
    feature_columns = [column for column in ALL_LARGE_SCALE_FEATURE_COLUMNS if column in frame.columns]
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
    )
    estimator.fit(train_frame[feature_columns], train_frame["label"].astype(int))
    explanation_method, global_importance = compute_shap_or_fallback_importance(
        estimator,
        feature_frame=validation_frame,
        labels=validation_frame["label"].astype(int),
        feature_columns=feature_columns,
        top_k=10,
    )
    district_explanation = explain_tabular_district_row(
        estimator,
        feature_frame=validation_frame,
        feature_columns=feature_columns,
        row_index=0,
        top_k=5,
    )

    st_frame = run_spatiotemporal_inference_from_frame(
        frame=pd.read_csv(dataset_csv_path),
        artifact_path=spatiotemporal_artifact_path,
        sequence_length=10,
    )
    st_example = st_frame.iloc[0].to_dict() if not st_frame.empty else {}

    payload = {
        "evaluation": {
            "model_name": evaluation.model_name,
            "decision_threshold": evaluation.decision_threshold,
            "metrics": asdict(evaluation.metrics),
            "model_comparisons": [asdict(item) for item in evaluation.model_comparisons],
        },
        "ablations": [asdict(item) for item in ablations],
        "explainability": {
            "global_method": explanation_method,
            "global_top_features": [asdict(item) for item in global_importance],
            "district_level_tabular": district_explanation,
            "district_level_spatiotemporal": st_example,
        },
    }

    json_target = Path(output_json_path)
    markdown_target = Path(output_markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_target.write_text(_to_markdown(payload), encoding="utf-8")
    return payload


def _to_markdown(payload: dict[str, object]) -> str:
    evaluation = payload["evaluation"]
    explainability = payload["explainability"]
    return (
        "# Advanced Research Suite\n\n"
        "## Best Evaluation\n"
        f"- Best model: {evaluation['model_name']}\n"
        f"- Decision threshold: {evaluation['decision_threshold']:.4f}\n"
        f"- Metrics: `{json.dumps(evaluation['metrics'])}`\n\n"
        "## Ablation Study\n"
        f"`{json.dumps(payload['ablations'], indent=2)}`\n\n"
        "## Explainability\n"
        f"- Global importance method: {explainability['global_method']}\n"
        f"- Global top features: `{json.dumps(explainability['global_top_features'])}`\n"
        f"- District explanation: `{json.dumps(explainability['district_level_tabular'])}`\n"
        f"- ST-GNN district explanation: `{json.dumps(explainability['district_level_spatiotemporal'])}`\n"
    )
