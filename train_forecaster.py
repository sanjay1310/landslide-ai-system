from landslide_ai.training.train_forecaster import (
    forecast_predictions_frame,
    train_rainfall_forecaster_from_windows,
)


def main() -> None:
    result = train_rainfall_forecaster_from_windows("data/india/rainfall_windows.csv")
    print("Rainfall Forecast Training")
    print("=" * 25)
    print(f"Samples: {result.sample_count}")
    print(f"Active samples: {result.active_sample_count}")
    print(f"Features: {', '.join(result.feature_columns)}")
    print(f"Target: {result.target_column}")
    print(f"Validation split: {result.validation_split_type}")
    print(f"Train MAE: {result.train_mae:.2f}")
    print(f"Train RMSE: {result.train_rmse:.2f}")
    print(f"Train R2: {result.train_r2:.2f}")
    print(f"Validation MAE: {result.validation_mae:.2f}")
    print(f"Validation RMSE: {result.validation_rmse:.2f}")
    print(f"Validation R2: {result.validation_r2:.2f}")
    print(f"Validation Active-Window MAE: {result.validation_active_mae:.2f}")
    print(f"Validation Active-Window RMSE: {result.validation_active_rmse:.2f}")
    print(f"Saved model: {result.model_path}")
    print()
    predictions = forecast_predictions_frame(result).sort_values("absolute_error", ascending=False)
    print("Hardest forecast windows (top 20 by absolute error)")
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
