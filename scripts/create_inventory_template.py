from __future__ import annotations

from landslide_ai.ingestion.inventory import create_inventory_template


def main() -> None:
    path = create_inventory_template("data/inventory/india_landslide_inventory_template.csv")
    print("Created inventory template")
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
