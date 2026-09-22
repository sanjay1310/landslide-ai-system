from __future__ import annotations

from pathlib import Path

import pandas as pd

from landslide_ai.forecasting.time_series import build_rainfall_windows


def build_forecast_windows_from_daily_weather(
    daily_weather_csv_path: str | Path,
    output_csv_path: str | Path,
    window_size: int = 3,
    forecast_horizon: int = 1,
) -> Path:
    frame = pd.read_csv(daily_weather_csv_path)
    dataset = build_rainfall_windows(
        frame=frame.rename(columns={"rainfall_mm": "rainfall_mm"}),
        region_column="region_id",
        date_column="date",
        rainfall_column="rainfall_mm",
        window_size=window_size,
        forecast_horizon=forecast_horizon,
    )
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.windows.to_csv(output_path, index=False)
    return output_path
