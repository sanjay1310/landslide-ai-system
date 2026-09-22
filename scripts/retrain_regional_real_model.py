from __future__ import annotations

from landslide_ai.training.train_regional_real_model import train_regional_real_runtime_model


def main() -> None:
    result = train_regional_real_runtime_model()
    print("Regional real model training complete")
    print(f"Model: {result.model_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Samples: {result.sample_count}")
    print(f"Train rows: {result.train_rows}")
    print(f"Validation rows: {result.validation_rows}")
    print(f"ROC-AUC: {result.roc_auc:.3f}")
    print(f"PR-AUC: {result.pr_auc:.3f}")
    print(f"Decision threshold: {result.decision_threshold:.3f}")


if __name__ == "__main__":
    main()
