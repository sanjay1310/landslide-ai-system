from __future__ import annotations

import os

from landslide_ai.research.large_scale import (
    evaluate_large_scale_risk_dataset,
    prepare_large_scale_dataset,
    write_large_scale_evaluation,
)


def main() -> None:
    prepare_large_scale_dataset(
        landslide_inventory_csv_path=os.getenv("LANDSLIDE_EVENT_INVENTORY_CSV"),
    )
    result = evaluate_large_scale_risk_dataset()
    write_large_scale_evaluation(result)
    print("Large-scale temporal risk evaluation complete")
    print("Dataset: data/india/india_risk_timeseries.csv")
    print("JSON: artifacts/large_scale_risk_evaluation.json")
    print("Markdown: docs/LARGE_SCALE_RESEARCH_EVALUATION.md")


if __name__ == "__main__":
    main()
