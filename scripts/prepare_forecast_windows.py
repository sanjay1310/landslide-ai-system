import pandas as pd

from landslide_ai.forecasting.time_series import build_rainfall_windows


def main() -> None:
    frame = pd.read_csv("data/raw/india_rainfall_timeseries.csv")
    dataset = build_rainfall_windows(frame, window_size=3, forecast_horizon=1)
    output_path = "data/india/rainfall_windows.csv"
    dataset.windows.to_csv(output_path, index=False)
    print("Prepared rainfall forecast windows")
    print(f"Rows: {len(dataset.windows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
