from __future__ import annotations

from landslide_ai.ingestion.dem import fetch_dem_elevation_for_regions
from landslide_ai.research.real_regions import default_regional_paths


def main() -> None:
    paths = default_regional_paths(".")
    result = fetch_dem_elevation_for_regions(
        region_csv_path=paths.region_csv,
        output_csv_path="data/regional/kerala_uttarakhand_dem.csv",
        dataset="srtm90m",
    )
    print("Regional DEM fetch complete")
    print(f"Output: {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Dataset: {result.dataset}")


if __name__ == "__main__":
    main()
