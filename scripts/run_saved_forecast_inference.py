from landslide_ai.services.system_service import LandslideSystemService


def main() -> None:
    service = LandslideSystemService(".")
    forecast_frame = service.forecast_from_windows("data/india/rainfall_windows.csv")
    print("Saved forecast inference")
    print("=" * 23)
    print(forecast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
