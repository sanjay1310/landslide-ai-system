from __future__ import annotations

from landslide_ai.graph.gnn import torch_available
from landslide_ai.training.train_gnn import train_graph_risk_model


def main() -> None:
    if not torch_available():
        print("PyTorch is not installed. Install `torch` to train the graph model.")
        return

    result = train_graph_risk_model("data/india/india_graph_training.csv")
    print("Graph neural model training complete")
    print(f"Samples: {result.sample_count}")
    print(f"Train Accuracy: {result.accuracy:.3f}")
    print(f"Validation Accuracy: {result.validation_accuracy:.3f}")
    print(f"Train F1: {result.f1:.3f}")
    print(f"Validation F1: {result.validation_f1:.3f}")
    print(f"Artifact: {result.artifact_path}")
    print(result.report_text)


if __name__ == "__main__":
    main()
