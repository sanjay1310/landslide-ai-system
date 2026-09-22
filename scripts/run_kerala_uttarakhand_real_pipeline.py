from __future__ import annotations

import os

from landslide_ai.research.large_scale import write_large_scale_evaluation
from landslide_ai.research.real_regions import run_target_state_real_evaluation


def main() -> None:
    inventory_path = os.getenv("LANDSLIDE_EVENT_INVENTORY_CSV")
    paths, result = run_target_state_real_evaluation(
        project_root=".",
        landslide_inventory_csv_path=inventory_path,
    )
    write_large_scale_evaluation(
        result,
        json_path="artifacts/kerala_uttarakhand_real_evaluation.json",
        markdown_path="docs/KERALA_UTTARAKHAND_REAL_EVALUATION.md",
    )
    print("Kerala/Uttarakhand real-data pipeline complete")
    print(f"Regions: {paths.region_csv}")
    print(f"NASA daily: {paths.nasa_daily_csv}")
    print(f"Risk dataset: {paths.risk_dataset_csv}")
    print("JSON: artifacts/kerala_uttarakhand_real_evaluation.json")
    print("Markdown: docs/KERALA_UTTARAKHAND_REAL_EVALUATION.md")


if __name__ == "__main__":
    main()
