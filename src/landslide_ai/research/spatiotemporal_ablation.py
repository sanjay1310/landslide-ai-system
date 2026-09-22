from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from landslide_ai.research.fair_graph_benchmark import generate_fair_graph_benchmark
from landslide_ai.training.train_spatiotemporal_gnn import train_spatiotemporal_graph_model


@dataclass(slots=True)
class SpatioTemporalAblationRow:
    name: str
    use_graph: bool
    use_temporal: bool
    include_vegetation: bool
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float


ABLATION_CONFIGS = [
    {"name": "full_stgnn", "use_graph": True, "use_temporal": True, "include_vegetation": True},
    {"name": "without_graph", "use_graph": False, "use_temporal": True, "include_vegetation": True},
    {"name": "without_temporal", "use_graph": True, "use_temporal": False, "include_vegetation": True},
    {"name": "without_vegetation", "use_graph": True, "use_temporal": True, "include_vegetation": False},
]


def run_spatiotemporal_ablation_study(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    output_json_path: str | Path = "artifacts/spatiotemporal_ablation.json",
    hidden_size: int = 48,
    epochs: int = 8,
    learning_rate: float = 0.001,
) -> list[SpatioTemporalAblationRow]:
    rows: list[SpatioTemporalAblationRow] = []
    for config in ABLATION_CONFIGS:
        artifact_path = Path("artifacts") / f"{config['name']}.pt"
        train_spatiotemporal_graph_model(
            dataset_csv_path=dataset_csv_path,
            artifact_path=artifact_path,
            sequence_length=10,
            hidden_size=hidden_size,
            dropout=0.15,
            epochs=epochs,
            learning_rate=learning_rate,
            architecture="gru",
            focal_gamma=1.5,
            early_stopping_patience=4,
            positive_class_weight_scale=1.2,
            selection_metric="rare_event_score",
            use_graph=config["use_graph"],
            use_temporal=config["use_temporal"],
            include_vegetation=config["include_vegetation"],
        )
        benchmark_path = Path("artifacts") / f"{config['name']}_fair_benchmark.json"
        fair_result = generate_fair_graph_benchmark(
            st_artifact_path=artifact_path,
            output_path=benchmark_path,
        )
        del fair_result
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))["spatiotemporal_gnn_same_scope"]
        rows.append(
            SpatioTemporalAblationRow(
                name=config["name"],
                use_graph=config["use_graph"],
                use_temporal=config["use_temporal"],
                include_vegetation=config["include_vegetation"],
                threshold=float(payload["threshold"]),
                accuracy=float(payload["accuracy"]),
                precision=float(payload["precision"]),
                recall=float(payload["recall"]),
                f1=float(payload["f1"]),
                roc_auc=float(payload["roc_auc"]),
                pr_auc=float(payload["pr_auc"]),
                brier_score=float(payload["brier_score"]),
            )
        )

    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(item) for item in rows], indent=2), encoding="utf-8")
    return rows
