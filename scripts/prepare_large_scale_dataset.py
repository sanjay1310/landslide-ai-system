from __future__ import annotations

import os

from landslide_ai.research.large_scale import prepare_large_scale_dataset


def main() -> None:
    output_path = prepare_large_scale_dataset(
        landslide_inventory_csv_path=os.getenv("LANDSLIDE_EVENT_INVENTORY_CSV"),
    )
    print("Prepared large-scale risk dataset")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
