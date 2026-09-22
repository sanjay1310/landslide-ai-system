from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from landslide_ai.forecasting.baseline import ForecastPrediction
from landslide_ai.forecasting.lstm import RainfallLSTMForecaster, torch_available


@dataclass(slots=True)
class LSTMForecastTrainingResult:
    sample_count: int
    feature_columns: list[str]
    target_column: str
    mae: float
    rmse: float
    r2: float
    losses: list[float]
    predictions: list[ForecastPrediction]


def train_lstm_forecaster_from_windows(
    csv_path: str,
    epochs: int = 250,
    learning_rate: float = 0.01,
) -> LSTMForecastTrainingResult:
    if not torch_available():
        raise ImportError("PyTorch is not installed. Install `torch` before running LSTM training.")

    import torch

    frame = pd.read_csv(csv_path)
    feature_columns = [column for column in frame.columns if column.startswith("rainfall_t_minus_")]
    target_column = next(column for column in frame.columns if column.startswith("rainfall_t_plus_"))

    x_values = frame[feature_columns].values.astype("float32")
    y_values = frame[target_column].values.astype("float32")

    x_train = torch.tensor(x_values).unsqueeze(-1)
    y_train = torch.tensor(y_values)

    forecaster = RainfallLSTMForecaster(input_size=1, hidden_size=32, num_layers=1)
    losses = forecaster.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    predicted_tensor = forecaster.predict(x_train)
    predicted = predicted_tensor.detach().cpu().numpy().tolist()

    predictions = [
        ForecastPrediction(
            region_id=str(row["region_id"]),
            reference_date=str(row["reference_date"]),
            actual_rainfall=float(actual),
            predicted_rainfall=float(prediction),
            absolute_error=abs(float(actual) - float(prediction)),
        )
        for (_, row), actual, prediction in zip(
            frame.iterrows(),
            frame[target_column].tolist(),
            predicted,
            strict=True,
        )
    ]

    return LSTMForecastTrainingResult(
        sample_count=len(frame),
        feature_columns=feature_columns,
        target_column=target_column,
        mae=float(mean_absolute_error(frame[target_column], predicted)),
        rmse=float(mean_squared_error(frame[target_column], predicted) ** 0.5),
        r2=float(r2_score(frame[target_column], predicted)),
        losses=losses,
        predictions=predictions,
    )


def lstm_predictions_frame(result: LSTMForecastTrainingResult) -> pd.DataFrame:
    return pd.DataFrame([asdict(prediction) for prediction in result.predictions])
