from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ForecastWindowDataset:
    windows: pd.DataFrame
    target_column: str
    feature_columns: list[str]


def build_rainfall_windows(
    frame: pd.DataFrame,
    region_column: str = "region_id",
    date_column: str = "date",
    rainfall_column: str = "rainfall_mm",
    window_size: int = 3,
    forecast_horizon: int = 1,
) -> ForecastWindowDataset:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1")

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column].astype(str), errors="coerce")
    working = working.sort_values([region_column, date_column]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    feature_columns = [f"rainfall_t_minus_{offset}" for offset in range(window_size, 0, -1)]
    target_column = f"rainfall_t_plus_{forecast_horizon}"

    for region_id, group in working.groupby(region_column):
        series = group[rainfall_column].tolist()
        dates = group[date_column].tolist()
        for index in range(window_size, len(group) - forecast_horizon + 1):
            reference_date = dates[index - 1]
            row: dict[str, object] = {
                region_column: region_id,
                "reference_date": reference_date,
            }
            history = series[index - window_size : index]
            for column_name, value in zip(feature_columns, history, strict=True):
                row[column_name] = value
            history_series = pd.Series(history, dtype="float64")
            row["rainfall_history_sum"] = float(history_series.sum())
            row["rainfall_history_mean"] = float(history_series.mean())
            row["rainfall_history_max"] = float(history_series.max())
            row["rainfall_history_min"] = float(history_series.min())
            row["rainfall_history_std"] = float(history_series.std(ddof=0))
            row["rainfall_last_value"] = float(history[-1])
            row["rainfall_first_value"] = float(history[0])
            row["rainfall_trend"] = float(history[-1] - history[0])
            row["rainfall_range"] = float(history_series.max() - history_series.min())
            row["wet_day_count"] = int((history_series >= 1.0).sum())
            row["heavy_rain_day_count"] = int((history_series >= 35.0).sum())
            row["recent_share"] = float(history[-1] / max(history_series.sum(), 1.0))
            if pd.notna(reference_date):
                reference_ts = pd.Timestamp(reference_date)
                day_of_year = float(reference_ts.dayofyear)
                month = float(reference_ts.month)
            else:
                day_of_year = 1.0
                month = 1.0
            row["reference_day_sin"] = float(np.sin((2.0 * np.pi * day_of_year) / 365.0))
            row["reference_day_cos"] = float(np.cos((2.0 * np.pi * day_of_year) / 365.0))
            row["reference_month_sin"] = float(np.sin((2.0 * np.pi * month) / 12.0))
            row["reference_month_cos"] = float(np.cos((2.0 * np.pi * month) / 12.0))
            row[target_column] = series[index + forecast_horizon - 1]
            rows.append(row)

    return ForecastWindowDataset(
        windows=pd.DataFrame(rows),
        target_column=target_column,
        feature_columns=feature_columns,
    )
