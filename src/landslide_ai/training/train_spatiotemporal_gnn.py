from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from landslide_ai.graph.spatiotemporal import (
    SpatioTemporalGraphRiskModel,
    build_spatiotemporal_dataset,
)
from landslide_ai.mlops import runtime_versions, write_experiment_log


@dataclass(slots=True)
class SpatioTemporalTrainingResult:
    artifact_path: Path
    metadata_path: Path
    experiment_log_path: Path
    sample_count: int
    sample_dates: int
    node_count: int
    validation_f1: float
    validation_roc_auc: float
    validation_pr_auc: float
    validation_uncertainty_mean: float


def train_spatiotemporal_graph_model(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    sequence_length: int = 7,
    hidden_size: int = 32,
    dropout: float = 0.15,
    epochs: int = 18,
    learning_rate: float = 0.0015,
    architecture: str = "gru",
    focal_gamma: float = 1.5,
    early_stopping_patience: int = 4,
    positive_class_weight_scale: float = 1.0,
    selection_metric: str = "rare_event_score",
    use_graph: bool = True,
    use_temporal: bool = True,
    include_vegetation: bool = True,
    experiment_dir: str | Path = "artifacts/experiments",
) -> SpatioTemporalTrainingResult:
    frame = pd.read_csv(dataset_csv_path)
    dataset = build_spatiotemporal_dataset(frame, sequence_length=sequence_length)
    if len(dataset.sample_dates) == 0:
        raise ValueError("Spatio-temporal dataset is empty. Ensure the regional dataset has enough dated rows per region.")

    model = SpatioTemporalGraphRiskModel(
        hidden_size=hidden_size,
        dropout=dropout,
        sequence_length=sequence_length,
        architecture=architecture,
        focal_gamma=focal_gamma,
        use_graph=use_graph,
        use_temporal=use_temporal,
        include_vegetation=include_vegetation,
    )
    metrics = model.fit(
        dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        positive_class_weight_scale=positive_class_weight_scale,
        selection_metric=selection_metric,
    )
    artifact = Path(artifact_path)
    metadata_path = artifact.with_name(f"{artifact.stem}_metadata.json")
    metadata = {
        "model_type": "spatiotemporal_graph_gnn",
        "source_csv": str(dataset_csv_path),
        "feature_columns": dataset.feature_columns,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "architecture": architecture,
        "focal_gamma": focal_gamma,
        "early_stopping_patience": early_stopping_patience,
        "positive_class_weight_scale": positive_class_weight_scale,
        "selection_metric": selection_metric,
        "use_graph": use_graph,
        "use_temporal": use_temporal,
        "include_vegetation": include_vegetation,
        "sample_count": int(len(dataset.sequences)),
        "sample_dates": int(len(dataset.sample_dates)),
        "node_count": int(len(dataset.region_ids)),
        **runtime_versions(),
        "python_version": runtime_versions()["python"],
        "scikit_learn_version": runtime_versions()["scikit_learn"],
        **metrics,
    }
    model.save(artifact, metadata=metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    experiment_log_path = write_experiment_log(
        experiment_name="spatiotemporal_gnn",
        parameters={
            "dataset_csv_path": str(dataset_csv_path),
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "architecture": architecture,
            "focal_gamma": focal_gamma,
            "early_stopping_patience": early_stopping_patience,
            "positive_class_weight_scale": positive_class_weight_scale,
            "selection_metric": selection_metric,
            "use_graph": use_graph,
            "use_temporal": use_temporal,
            "include_vegetation": include_vegetation,
        },
        metrics=metrics,
        artifacts={"artifact_path": str(artifact), "metadata_path": str(metadata_path)},
        output_dir=experiment_dir,
    )
    return SpatioTemporalTrainingResult(
        artifact_path=artifact,
        metadata_path=metadata_path,
        experiment_log_path=experiment_log_path,
        sample_count=int(len(dataset.sequences)),
        sample_dates=int(len(dataset.sample_dates)),
        node_count=int(len(dataset.region_ids)),
        validation_f1=float(metrics["validation_f1"]),
        validation_roc_auc=float(metrics["validation_roc_auc"]),
        validation_pr_auc=float(metrics["validation_pr_auc"]),
        validation_uncertainty_mean=float(metrics["validation_uncertainty_mean"]),
    )
