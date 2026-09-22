from __future__ import annotations

from landslide_ai.ingestion.ndvi import fetch_regional_ndvi_from_sentinel
from landslide_ai.research.real_regions import default_regional_paths


def main() -> None:
    paths = default_regional_paths(".")
    result = fetch_regional_ndvi_from_sentinel(
        region_csv_path=paths.region_csv,
        output_csv_path="data/regional/kerala_uttarakhand_ndvi.csv",
        start_date="2026-02-01",
        end_date="2026-04-10",
    )
    print("Regional NDVI fetch complete")
    print(f"Output: {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Collection: {result.collection}")


if __name__ == "__main__":
    main()
