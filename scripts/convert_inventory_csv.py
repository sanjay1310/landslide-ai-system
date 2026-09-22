from __future__ import annotations

import argparse

from landslide_ai.ingestion.inventory import write_standardized_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a raw landslide inventory CSV into project format.")
    parser.add_argument("input_csv", help="Path to the raw inventory CSV")
    parser.add_argument("output_csv", help="Path to write the standardized inventory CSV")
    parser.add_argument("--source", default="external_inventory", help="Source label to store in the output CSV")
    args = parser.parse_args()

    output_path = write_standardized_inventory(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        source_name=args.source,
    )
    print("Standardized landslide inventory")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
