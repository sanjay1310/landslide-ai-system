from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from landslide_ai.agents.graph_agent import GraphAgent
from landslide_ai.agents.vision_agent import VisionAgent
from landslide_ai.dashboard.view_models import build_state_summary
from landslide_ai.data.loader import load_region_records, load_training_frame
from landslide_ai.models.baseline import BaselineRiskModel
from landslide_ai.models.trained_risk import TRAINED_RISK_FEATURE_COLUMNS, TrainedRiskModel
from landslide_ai.pipeline.feature_engineering import build_features
from landslide_ai.pipeline.orchestrator import LandslideSystemOrchestrator


GRAPH_ABLATION_COLUMNS = ["graph_influence", "propagated_graph_signal"]


@dataclass(slots=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


@dataclass(slots=True)
class ModelComparisonResult:
    name: str
    feature_columns: list[str]
    metrics: BinaryMetrics
    validation_rows: int


@dataclass(slots=True)
class ScenarioExampleResult:
    scenario_name: str
    target_scope: str
    mean_risk_delta: float
    high_risk_delta: int
    top_movers: list[dict[str, object]]


@dataclass(slots=True)
class ResearchEvaluationResult:
    dataset_summary: dict[str, object]
    experimental_setup: dict[str, object]
    model_comparisons: list[ModelComparisonResult]
    scenario_examples: list[ScenarioExampleResult]
    state_analysis: list[dict[str, object]]
    district_hotspots: list[dict[str, object]]
    limitations: list[str]


def _safe_roc_auc(y_true: pd.Series, y_score: list[float]) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0


def _metrics_from_predictions(y_true: pd.Series, y_pred: list[int], y_score: list[float]) -> BinaryMetrics:
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=_safe_roc_auc(y_true, y_score),
    )


def build_research_frame(csv_path: str | Path) -> pd.DataFrame:
    training_frame = load_training_frame(csv_path)
    records = load_region_records(csv_path)
    original_graph_flag = os.environ.get("LANDSLIDE_ENABLE_GRAPH_MODEL")
    os.environ["LANDSLIDE_ENABLE_GRAPH_MODEL"] = "false"
    graph_agent = GraphAgent()
    if original_graph_flag is None:
        os.environ.pop("LANDSLIDE_ENABLE_GRAPH_MODEL", None)
    else:
        os.environ["LANDSLIDE_ENABLE_GRAPH_MODEL"] = original_graph_flag
    vision_agent = VisionAgent()
    graph_map = graph_agent.evaluate_network(records)

    rows: list[dict[str, object]] = []
    for record in records:
        terrain_change_signal, vegetation_signal = vision_agent.evaluate(record)
        graph_insights = graph_map.get(record.region_id)
        graph_signal = graph_insights.local_influence if graph_insights else graph_agent.evaluate(record)
        propagated_signal = graph_insights.propagated_influence if graph_insights else graph_signal
        features = build_features(
            record=record,
            terrain_change_signal=terrain_change_signal,
            vegetation_signal=vegetation_signal,
            graph_signal=graph_signal,
        )
        feature_row = TrainedRiskModel.build_feature_row(features, propagated_signal)
        feature_row.update(
            {
                "region_id": record.region_id,
                "state": record.state,
                "district": record.district,
                "label": int(training_frame.loc[training_frame["region_id"] == record.region_id, "label"].iloc[0]),
                "hazard_score": float(
                    training_frame.loc[training_frame["region_id"] == record.region_id, "hazard_score"].iloc[0]
                )
                if "hazard_score" in training_frame.columns
                else None,
            }
        )
        rows.append(feature_row)
    return pd.DataFrame(rows)


def summarize_dataset(csv_path: str | Path, research_frame: pd.DataFrame | None = None) -> dict[str, object]:
    frame = load_training_frame(csv_path)
    working = research_frame if research_frame is not None else build_research_frame(csv_path)
    class_counts = frame["label"].value_counts().sort_index().to_dict()
    return {
        "row_count": int(len(frame)),
        "state_count": int(frame["state"].nunique()),
        "district_count": int(frame["district"].nunique()),
        "positive_labels": int(class_counts.get(1, 0)),
        "negative_labels": int(class_counts.get(0, 0)),
        "feature_count": len(TRAINED_RISK_FEATURE_COLUMNS),
        "states": sorted(frame["state"].astype(str).unique().tolist()),
        "mean_risk_feature_values": {
            column: round(float(working[column].mean()), 3)
            for column in TRAINED_RISK_FEATURE_COLUMNS
        },
    }


def evaluate_rule_based_model(
    csv_path: str | Path,
    research_frame: pd.DataFrame | None = None,
    decision_threshold: float = 0.50,
) -> ModelComparisonResult:
    frame = research_frame if research_frame is not None else build_research_frame(csv_path)
    baseline_model = BaselineRiskModel()
    scores: list[float] = []
    for row in frame.to_dict(orient="records"):
        risk_score, _, _ = baseline_model.predict(
            features=type(
                "_Features",
                (),
                {
                    "rainfall_intensity": row["rainfall_intensity"],
                    "antecedent_rainfall": row["antecedent_rainfall"],
                    "slope_factor": row["slope_factor"],
                    "soil_wetness": row["soil_wetness"],
                    "vegetation_stress": row["vegetation_stress"],
                    "graph_influence": row["graph_influence"],
                },
            )(),
            propagated_graph_signal=float(row["propagated_graph_signal"]),
        )
        scores.append(float(risk_score))
    predictions = [1 if score >= decision_threshold else 0 for score in scores]
    metrics = _metrics_from_predictions(frame["label"], predictions, scores)
    return ModelComparisonResult(
        name="rule_based_baseline",
        feature_columns=TRAINED_RISK_FEATURE_COLUMNS,
        metrics=metrics,
        validation_rows=len(frame),
    )


def evaluate_trained_model(
    csv_path: str | Path,
    feature_columns: list[str] | None = None,
    model_type: str = "gradient_boosting",
    validation_fraction: float = 0.25,
    random_state: int = 42,
    name: str = "trained_risk_model",
) -> ModelComparisonResult:
    frame = build_research_frame(csv_path)
    selected_features = feature_columns or TRAINED_RISK_FEATURE_COLUMNS
    x = frame[selected_features]
    y = frame["label"].astype(int)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y,
    )
    estimator = TrainedRiskModel.create_estimator(model_type)
    estimator.fit(x_train, y_train)
    probabilities = estimator.predict_proba(x_val)[:, 1].tolist()
    predictions = estimator.predict(x_val).tolist()
    metrics = _metrics_from_predictions(y_val, predictions, probabilities)
    return ModelComparisonResult(
        name=name,
        feature_columns=list(selected_features),
        metrics=metrics,
        validation_rows=len(x_val),
    )


def evaluate_graph_ablation(
    csv_path: str | Path,
    model_type: str = "gradient_boosting",
    validation_fraction: float = 0.25,
    random_state: int = 42,
) -> list[ModelComparisonResult]:
    with_graph = evaluate_trained_model(
        csv_path=csv_path,
        model_type=model_type,
        validation_fraction=validation_fraction,
        random_state=random_state,
        name="trained_risk_with_graph",
    )
    without_graph = evaluate_trained_model(
        csv_path=csv_path,
        feature_columns=[column for column in TRAINED_RISK_FEATURE_COLUMNS if column not in GRAPH_ABLATION_COLUMNS],
        model_type=model_type,
        validation_fraction=validation_fraction,
        random_state=random_state,
        name="trained_risk_without_graph",
    )
    return [with_graph, without_graph]


def evaluate_scenarios(csv_path: str | Path) -> list[ScenarioExampleResult]:
    original_trained_flag = os.environ.get("LANDSLIDE_ENABLE_TRAINED_RISK_MODEL")
    original_graph_flag = os.environ.get("LANDSLIDE_ENABLE_GRAPH_MODEL")
    os.environ["LANDSLIDE_ENABLE_TRAINED_RISK_MODEL"] = "false"
    os.environ["LANDSLIDE_ENABLE_GRAPH_MODEL"] = "false"
    orchestrator = LandslideSystemOrchestrator()
    if original_trained_flag is None:
        os.environ.pop("LANDSLIDE_ENABLE_TRAINED_RISK_MODEL", None)
    else:
        os.environ["LANDSLIDE_ENABLE_TRAINED_RISK_MODEL"] = original_trained_flag
    if original_graph_flag is None:
        os.environ.pop("LANDSLIDE_ENABLE_GRAPH_MODEL", None)
    else:
        os.environ["LANDSLIDE_ENABLE_GRAPH_MODEL"] = original_graph_flag
    records = load_region_records(csv_path)
    baseline = orchestrator.run_records(records)
    baseline_frame = pd.DataFrame(
        [{"region_id": item.region_id, "risk_score": item.risk_score, "risk_level": item.risk_level} for item in baseline]
    )
    definitions = [
        {
            "scenario_name": "Kerala heavy rainfall stress test",
            "target_scope": "Kerala",
            "kwargs": {
                "rainfall_multiplier": 1.30,
                "antecedent_rainfall_multiplier": 1.20,
                "soil_wetness_delta": 0.08,
                "target_state": "Kerala",
            },
        },
        {
            "scenario_name": "Himalayan slope saturation test",
            "target_scope": "Uttarakhand",
            "kwargs": {
                "rainfall_multiplier": 1.20,
                "antecedent_rainfall_multiplier": 1.25,
                "soil_wetness_delta": 0.10,
                "neighbor_risk_delta": 0.05,
                "target_state": "Uttarakhand",
            },
        },
    ]
    results: list[ScenarioExampleResult] = []
    for definition in definitions:
        assessed = orchestrator.run_scenario(records, **definition["kwargs"])
        assessed_frame = pd.DataFrame(
            [
                {
                    "region_id": item.region_id,
                    "district": item.district,
                    "state": item.state,
                    "risk_score": item.risk_score,
                    "forecast_risk": item.forecast_risk,
                    "primary_driver": item.primary_driver,
                    "risk_level": item.risk_level,
                }
                for item in assessed
            ]
        )
        merged = assessed_frame.merge(
            baseline_frame.rename(columns={"risk_score": "base_risk_score", "risk_level": "base_risk_level"}),
            on="region_id",
            how="left",
        )
        merged["risk_delta"] = merged["risk_score"] - merged["base_risk_score"]
        top_movers = (
            merged.sort_values("risk_delta", ascending=False)
            .head(5)[["district", "state", "risk_delta", "forecast_risk", "primary_driver"]]
            .round(3)
            .to_dict(orient="records")
        )
        results.append(
            ScenarioExampleResult(
                scenario_name=definition["scenario_name"],
                target_scope=definition["target_scope"],
                mean_risk_delta=round(float(merged["risk_delta"].mean()), 3),
                high_risk_delta=int((merged["risk_level"] == "High").sum() - (merged["base_risk_level"] == "High").sum()),
                top_movers=top_movers,
            )
        )
    return results


def evaluate_state_and_district_analysis(csv_path: str | Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    original_trained_flag = os.environ.get("LANDSLIDE_ENABLE_TRAINED_RISK_MODEL")
    original_graph_flag = os.environ.get("LANDSLIDE_ENABLE_GRAPH_MODEL")
    os.environ["LANDSLIDE_ENABLE_TRAINED_RISK_MODEL"] = "false"
    os.environ["LANDSLIDE_ENABLE_GRAPH_MODEL"] = "false"
    orchestrator = LandslideSystemOrchestrator()
    if original_trained_flag is None:
        os.environ.pop("LANDSLIDE_ENABLE_TRAINED_RISK_MODEL", None)
    else:
        os.environ["LANDSLIDE_ENABLE_TRAINED_RISK_MODEL"] = original_trained_flag
    if original_graph_flag is None:
        os.environ.pop("LANDSLIDE_ENABLE_GRAPH_MODEL", None)
    else:
        os.environ["LANDSLIDE_ENABLE_GRAPH_MODEL"] = original_graph_flag
    records = load_region_records(csv_path)
    assessments = orchestrator.run_records(records)
    frame = pd.DataFrame(
        [
            {
                "Region ID": item.region_id,
                "State": item.state,
                "District": item.district,
                "Risk Score": round(item.risk_score, 3),
                "Forecast Risk": round(item.forecast_risk, 3),
                "Uncertainty": round(item.uncertainty_score, 3),
                "Primary Driver": item.primary_driver,
            }
            for item in assessments
        ]
    )
    state_summary = build_state_summary(frame).to_dict(orient="records")
    district_hotspots = (
        frame.sort_values(["Risk Score", "Forecast Risk"], ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    return state_summary, district_hotspots


def default_limitations() -> list[str]:
    return [
        "The district dataset is modest in size and partially seed-generated, so results should be treated as research-oriented rather than production-validated.",
        "The current labels are proxies for landslide susceptibility and do not represent a complete historical landslide inventory.",
        "Forecasting and graph artifacts depend on local environment compatibility and should be retrained in the target runtime before deployment claims are made.",
        "The dashboard is suitable for comparative analysis and scenario exploration, but not yet for official disaster warning operations.",
    ]


def run_full_research_evaluation(csv_path: str | Path = "data/india/india_regions.csv") -> ResearchEvaluationResult:
    research_frame = build_research_frame(csv_path)
    dataset_summary = summarize_dataset(csv_path, research_frame=research_frame)
    setup = {
        "dataset_path": str(csv_path),
        "validation_split": "stratified holdout",
        "validation_fraction": 0.25,
        "trained_model_type": "gradient_boosting",
        "rule_based_threshold": 0.50,
        "graph_ablation_columns": GRAPH_ABLATION_COLUMNS,
        "evaluation_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    }
    model_comparisons = [
        evaluate_rule_based_model(csv_path, research_frame=research_frame),
        *evaluate_graph_ablation(csv_path),
    ]
    scenario_examples = evaluate_scenarios(csv_path)
    state_analysis, district_hotspots = evaluate_state_and_district_analysis(csv_path)
    return ResearchEvaluationResult(
        dataset_summary=dataset_summary,
        experimental_setup=setup,
        model_comparisons=model_comparisons,
        scenario_examples=scenario_examples,
        state_analysis=state_analysis,
        district_hotspots=district_hotspots,
        limitations=default_limitations(),
    )


def evaluation_to_json_dict(result: ResearchEvaluationResult) -> dict[str, object]:
    return {
        "dataset_summary": result.dataset_summary,
        "experimental_setup": result.experimental_setup,
        "model_comparisons": [
            {
                "name": item.name,
                "feature_columns": item.feature_columns,
                "validation_rows": item.validation_rows,
                "metrics": asdict(item.metrics),
            }
            for item in result.model_comparisons
        ],
        "scenario_examples": [asdict(item) for item in result.scenario_examples],
        "state_analysis": result.state_analysis,
        "district_hotspots": result.district_hotspots,
        "limitations": result.limitations,
    }


def write_research_outputs(
    result: ResearchEvaluationResult,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    payload = evaluation_to_json_dict(result)
    json_target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    comparison_lines = "\n".join(
        [
            (
                f"| {item.name} | {item.validation_rows} | {item.metrics.accuracy:.3f} | "
                f"{item.metrics.precision:.3f} | {item.metrics.recall:.3f} | "
                f"{item.metrics.f1:.3f} | {item.metrics.roc_auc:.3f} |"
            )
            for item in result.model_comparisons
        ]
    )
    scenario_lines = "\n".join(
        [
            f"### {item.scenario_name}\n"
            f"- Target scope: {item.target_scope}\n"
            f"- Mean risk delta: {item.mean_risk_delta:.3f}\n"
            f"- Change in high-risk districts: {item.high_risk_delta}\n"
            f"- Top movers: {json.dumps(item.top_movers, indent=2)}\n"
            for item in result.scenario_examples
        ]
    )
    markdown_target.write_text(
        (
            "# Research Evaluation Summary\n\n"
            "## Dataset Summary\n"
            f"- Rows: {result.dataset_summary['row_count']}\n"
            f"- States: {result.dataset_summary['state_count']}\n"
            f"- Districts: {result.dataset_summary['district_count']}\n"
            f"- Positive labels: {result.dataset_summary['positive_labels']}\n"
            f"- Negative labels: {result.dataset_summary['negative_labels']}\n\n"
            "## Experimental Setup\n"
            f"- Dataset: `{result.experimental_setup['dataset_path']}`\n"
            f"- Validation split: {result.experimental_setup['validation_split']} ({result.experimental_setup['validation_fraction']:.2f})\n"
            f"- Trained model: {result.experimental_setup['trained_model_type']}\n"
            f"- Metrics: {', '.join(result.experimental_setup['evaluation_metrics'])}\n\n"
            "## Model Comparisons\n"
            "| Model | Validation Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            f"{comparison_lines}\n\n"
            "## Scenario Examples\n"
            f"{scenario_lines}\n"
            "## State Analysis\n"
            f"{json.dumps(result.state_analysis[:8], indent=2)}\n\n"
            "## Top District Hotspots\n"
            f"{json.dumps(result.district_hotspots[:10], indent=2)}\n\n"
            "## Limitations\n"
            + "\n".join([f"- {item}" for item in result.limitations])
            + "\n"
        ),
        encoding="utf-8",
    )
