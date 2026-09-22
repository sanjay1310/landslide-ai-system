from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from landslide_ai.training.train_spatiotemporal_gnn import train_spatiotemporal_graph_model


@dataclass(slots=True)
class SpatioTemporalFastAblationRow:
    name: str
    use_graph: bool
    use_temporal: bool
    include_vegetation: bool
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_roc_auc: float
    validation_pr_auc: float
    validation_uncertainty_mean: float


CONFIGS = [
    {"name": "full_stgnn", "use_graph": True, "use_temporal": True, "include_vegetation": True},
    {"name": "without_graph", "use_graph": False, "use_temporal": True, "include_vegetation": True},
    {"name": "without_temporal", "use_graph": True, "use_temporal": False, "include_vegetation": True},
    {"name": "without_vegetation", "use_graph": True, "use_temporal": True, "include_vegetation": False},
]


def run_fast_spatiotemporal_ablation(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    output_json_path: str | Path = "artifacts/spatiotemporal_ablation_fast.json",
    epochs: int = 4,
    hidden_size: int = 32,
    learning_rate: float = 0.0012,
) -> list[SpatioTemporalFastAblationRow]:
    rows: list[SpatioTemporalFastAblationRow] = []
    for config in CONFIGS:
        artifact_path = Path("artifacts") / f"{config['name']}_fast.pt"
        result = train_spatiotemporal_graph_model(
            dataset_csv_path=dataset_csv_path,
            artifact_path=artifact_path,
            sequence_length=10,
            hidden_size=hidden_size,
            dropout=0.15,
            epochs=epochs,
            learning_rate=learning_rate,
            architecture="gru",
            focal_gamma=1.5,
            early_stopping_patience=2,
            positive_class_weight_scale=1.2,
            selection_metric="rare_event_score",
            use_graph=config["use_graph"],
            use_temporal=config["use_temporal"],
            include_vegetation=config["include_vegetation"],
        )
        metadata = json.loads(artifact_path.with_name(f"{artifact_path.stem}_metadata.json").read_text(encoding="utf-8"))
        rows.append(
            SpatioTemporalFastAblationRow(
                name=config["name"],
                use_graph=config["use_graph"],
                use_temporal=config["use_temporal"],
                include_vegetation=config["include_vegetation"],
                validation_precision=float(metadata.get("validation_precision", 0.0)),
                validation_recall=float(metadata.get("validation_recall", 0.0)),
                validation_f1=float(metadata.get("validation_f1", 0.0)),
                validation_roc_auc=float(metadata.get("validation_roc_auc", 0.0)),
                validation_pr_auc=float(metadata.get("validation_pr_auc", 0.0)),
                validation_uncertainty_mean=float(metadata.get("validation_uncertainty_mean", 0.0)),
            )
        )

    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(item) for item in rows], indent=2), encoding="utf-8")
    return rows
