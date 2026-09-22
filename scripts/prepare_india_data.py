from landslide_ai.ingestion.india_pipeline import prepare_india_region_dataset


def main() -> None:
    result = prepare_india_region_dataset(
        rainfall_csv="data/raw/india_rainfall.csv",
        terrain_csv="data/raw/india_terrain.csv",
        output_csv="data/india/india_regions.csv",
    )
    print("Prepared India region dataset")
    print(f"Rows: {result.row_count}")
    print(f"Output: {result.output_path}")


if __name__ == "__main__":
    main()
