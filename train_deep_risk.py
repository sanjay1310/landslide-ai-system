from __future__ import annotations

from landslide_ai.models.deep_risk import torch_available
from landslide_ai.training.train_deep_risk import train_deep_risk_model_from_csv


def main() -> None:
    if not torch_available():
        print("PyTorch is not installed. Install `torch` to train the deep risk model.")
        return
    result = train_deep_risk_model_from_csv("data/india/india_regions.csv")
    print("Deep Risk Model Training")
    print("=" * 24)
    print(f"Samples: {result.sample_count}")
    print(f"Features: {', '.join(result.feature_columns)}")
    print(f"Epochs: {result.epochs}")
    print(f"Train Accuracy: {result.train_accuracy:.3f}")
    print(f"Validation Accuracy: {result.validation_accuracy:.3f}")
    print(f"Train F1: {result.train_f1:.3f}")
    print(f"Validation F1: {result.validation_f1:.3f}")
    print(f"Validation Precision: {result.validation_precision:.3f}")
    print(f"Validation Recall: {result.validation_recall:.3f}")
    print(f"Validation ROC-AUC: {result.validation_roc_auc:.3f}")
    print(f"Artifact: {result.model_path}")


if __name__ == "__main__":
    main()
