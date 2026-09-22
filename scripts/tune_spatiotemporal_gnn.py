from __future__ import annotations

import json
from pathlib import Path
import shutil

from landslide_ai.research.fair_graph_benchmark import generate_fair_graph_benchmark
from landslide_ai.training.train_spatiotemporal_gnn import train_spatiotemporal_graph_model


CONFIGS = [
    {
        "name": "gru_seq10_h48_d015_e8_lr10e4",
        "sequence_length": 10,
        "hidden_size": 48,
        "dropout": 0.15,
        "epochs": 8,
        "learning_rate": 0.0010,
        "architecture": "gru",
        "positive_class_weight_scale": 1.2,
    },
    {
        "name": "gru_seq14_h64_d010_e8_lr8e4",
        "sequence_length": 14,
        "hidden_size": 64,
        "dropout": 0.10,
        "epochs": 8,
        "learning_rate": 0.0008,
        "architecture": "gru",
        "positive_class_weight_scale": 1.35,
    },
    {
        "name": "lstm_seq14_h48_d015_e8_lr8e4",
        "sequence_length": 14,
        "hidden_size": 48,
        "dropout": 0.15,
        "epochs": 8,
        "learning_rate": 0.0008,
        "architecture": "lstm",
        "positive_class_weight_scale": 1.2,
    },
    {
        "name": "lstm_seq21_h64_d020_e10_lr6e4",
        "sequence_length": 21,
        "hidden_size": 64,
        "dropout": 0.20,
        "epochs": 10,
        "learning_rate": 0.0006,
        "architecture": "lstm",
        "positive_class_weight_scale": 1.4,
    },
]


def main() -> None:
    benchmark_path = Path("artifacts/fair_graph_sequence_benchmark.json")
    baseline = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else {}
    baseline_row = baseline.get("spatiotemporal_gnn_same_scope", {})
    best = {
        "name": "current_saved",
        "score": composite_score(baseline_row),
        "metrics": baseline_row,
        "artifact_path": Path("artifacts/spatiotemporal_gnn.pt"),
        "metadata_path": Path("artifacts/spatiotemporal_gnn_metadata.json"),
    }

    sweep_results: list[dict[str, object]] = []
    for config in CONFIGS:
        artifact_path = Path("artifacts") / f"{config['name']}.pt"
        result = train_spatiotemporal_graph_model(
            artifact_path=artifact_path,
            sequence_length=config["sequence_length"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            architecture=config.get("architecture", "gru"),
            positive_class_weight_scale=config.get("positive_class_weight_scale", 1.0),
            selection_metric="rare_event_score",
        )
        fair_result = generate_fair_graph_benchmark(
            st_artifact_path=artifact_path,
            output_path=Path("artifacts") / f"{config['name']}_fair_benchmark.json",
        )
        fair_metrics = json.loads(fair_result.output_path.read_text(encoding="utf-8"))["spatiotemporal_gnn_same_scope"]
        sweep_row = {
            **config,
            "validation_f1": result.validation_f1,
            "validation_roc_auc": result.validation_roc_auc,
            "validation_pr_auc": result.validation_pr_auc,
            "fair_metrics": fair_metrics,
            "score": composite_score(fair_metrics),
            "artifact_path": str(artifact_path),
        }
        sweep_results.append(sweep_row)
        if sweep_row["score"] > best["score"]:
            best = {
                "name": config["name"],
                "score": sweep_row["score"],
                "metrics": fair_metrics,
                "artifact_path": artifact_path,
                "metadata_path": artifact_path.with_name(f"{artifact_path.stem}_metadata.json"),
            }

    if best["name"] != "current_saved":
        shutil.copy2(best["artifact_path"], Path("artifacts/spatiotemporal_gnn.pt"))
        shutil.copy2(best["metadata_path"], Path("artifacts/spatiotemporal_gnn_metadata.json"))
        generate_fair_graph_benchmark(
            st_artifact_path="artifacts/spatiotemporal_gnn.pt",
            output_path="artifacts/fair_graph_sequence_benchmark.json",
        )

    summary = {
        "baseline_metrics": baseline_row,
        "best_model": {
            "name": best["name"],
            "score": best["score"],
            "metrics": best["metrics"],
        },
        "candidates": sweep_results,
    }
    summary_path = Path("artifacts/spatiotemporal_tuning_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("ST-GNN tuning complete")
    print("=" * 24)
    print(f"Best model: {best['name']}")
    print(f"Summary: {summary_path}")
    print(json.dumps(best["metrics"], indent=2))


def composite_score(metrics: dict[str, object]) -> float:
    if not metrics:
        return -1.0
    roc_auc = float(metrics.get("roc_auc", 0.0))
    pr_auc = float(metrics.get("pr_auc", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    return (1.4 * pr_auc) + (1.2 * f1) + (0.8 * roc_auc) + (0.55 * precision) + (0.3 * recall)


if __name__ == "__main__":
    main()
