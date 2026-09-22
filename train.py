from landslide_ai.training.train_baseline import train_random_forest_from_csv


def main() -> None:
    result = train_random_forest_from_csv("data/india/india_regions.csv")
    print("Trained Landslide Risk Model")
    print("=" * 29)
    print(f"Model type: {result.model_type}")
    print(f"Samples: {result.sample_count}")
    print(f"Features: {', '.join(result.feature_columns)}")
    print(f"Train accuracy: {result.train_accuracy:.3f}")
    print(f"Validation accuracy: {result.validation_accuracy:.3f}")
    print(f"Train F1: {result.train_f1:.3f}")
    print(f"Validation F1: {result.validation_f1:.3f}")
    print(f"Saved artifact: {result.model_path}")
    print(result.report_text)


if __name__ == "__main__":
    main()
