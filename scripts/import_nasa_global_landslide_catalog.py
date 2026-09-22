from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from landslide_ai.ingestion.inventory import (
    merge_mapped_inventory_frames,
    prepare_nasa_global_landslide_catalog_frame,
    write_coordinate_mapped_inventory,
)


TARGET_STATES = ("Kerala", "Uttarakhand")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map NASA Global Landslide Catalog events into the Kerala/Uttarakhand district inventory."
    )
    parser.add_argument(
        "--input-csv",
        default="data/inventory/nasa_global_landslide_catalog.csv",
        help="Path to the raw NASA Global Landslide Catalog CSV.",
    )
    parser.add_argument(
        "--existing-inventory",
        default="data/inventory/kerala_uttarakhand_inventory_mapped.csv",
        help="Path to the existing mapped project inventory CSV.",
    )
    parser.add_argument(
        "--mapped-output",
        default="data/inventory/kerala_uttarakhand_inventory_nasa_mapped.csv",
        help="Path to write the mapped NASA-only inventory CSV.",
    )
    parser.add_argument(
        "--merged-output",
        default="data/inventory/kerala_uttarakhand_inventory_merged.csv",
        help="Path to write the merged project plus NASA inventory CSV.",
    )
    parser.add_argument(
        "--region-metadata",
        default="data/raw/india_rainfall.csv",
        help="Path to the region metadata CSV with district centroids.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv)
    prepared = prepare_nasa_global_landslide_catalog_frame(raw, allowed_states=TARGET_STATES)

    with tempfile.TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir) / "nasa_glc_prepared.csv"
        prepared.to_csv(prepared_path, index=False)
        mapped_path = write_coordinate_mapped_inventory(
            input_csv_path=prepared_path,
            output_csv_path=args.mapped_output,
            region_metadata_csv_path=args.region_metadata,
            source_name="nasa_glc",
            allowed_states=TARGET_STATES,
        )

    mapped = pd.read_csv(mapped_path)
    existing_path = Path(args.existing_inventory)
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame(columns=mapped.columns)
    merged = merge_mapped_inventory_frames([existing, mapped])
    merged_output = Path(args.merged_output)
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_output, index=False)

    merged_keys = set(zip(merged["state"], merged["district"], merged["event_date"]))
    existing_keys = set(zip(existing.get("state", []), existing.get("district", []), existing.get("event_date", [])))
    mapped_keys = set(zip(mapped["state"], mapped["district"], mapped["event_date"]))
    new_unique_events = len(merged_keys - existing_keys)
    overlap_events = len(mapped_keys & existing_keys)

    print("NASA Global Landslide Catalog import complete")
    print(f"Prepared NASA candidate rows: {len(prepared)}")
    print(f"Mapped NASA rows: {len(mapped)}")
    print(f"Existing inventory rows: {len(existing)}")
    print(f"Merged inventory rows: {len(merged)}")
    print(f"New unique district-date events added: {new_unique_events}")
    print(f"NASA overlap with existing district-date events: {overlap_events}")
    print(f"NASA mapped inventory: {mapped_path}")
    print(f"Merged inventory: {merged_output}")


if __name__ == "__main__":
    main()
