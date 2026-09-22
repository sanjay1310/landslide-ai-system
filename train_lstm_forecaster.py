from landslide_ai.forecasting.lstm import torch_available
from landslide_ai.training.train_lstm_forecaster import (
    lstm_predictions_frame,
    train_lstm_forecaster_from_windows,
)


def main() -> None:
    if not torch_available():
        print("PyTorch is not installed. Install `torch` to run LSTM forecasting.")
        return

    result = train_lstm_forecaster_from_windows("data/india/rainfall_windows.csv")
    print("Rainfall LSTM Forecast Training")
    print("=" * 30)
    print(f"Samples: {result.sample_count}")
    print(f"Features: {', '.join(result.feature_columns)}")
    print(f"Target: {result.target_column}")
    print(f"MAE: {result.mae:.2f}")
    print(f"RMSE: {result.rmse:.2f}")
    print(f"R2: {result.r2:.2f}")
    print(f"Final loss: {result.losses[-1]:.4f}")
    print()
    print(lstm_predictions_frame(result).to_string(index=False))


if __name__ == "__main__":
    main()
