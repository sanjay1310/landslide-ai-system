from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from landslide_ai.ingestion.inventory import write_coordinate_mapped_inventory
from landslide_ai.research.large_scale import evaluate_large_scale_risk_dataset, write_large_scale_evaluation
from landslide_ai.research.real_regions import run_target_state_real_evaluation


TARGET_STATES = ("Kerala", "Uttarakhand")


def _derive_nasa_date_range(inventory_csv_path: str | Path, padding_days: int = 14) -> tuple[str, str]:
    frame = pd.read_csv(inventory_csv_path)
    dates = pd.to_datetime(frame["event_date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Inventory file does not contain valid event_date values.")
    start_date = (dates.min() - pd.Timedelta(days=padding_days)).strftime("%Y%m%d")
    end_date = (dates.max() + pd.Timedelta(days=padding_days)).strftime("%Y%m%d")
    return start_date, end_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Kerala/Uttarakhand real-data dataset using raw inventory events and NASA POWER rainfall."
    )
    parser.add_argument(
        "--raw-inventory",
        required=True,
        help="Path to the raw landslide inventory CSV containing state, district, event_date, latitude, and longitude.",
    )
    parser.add_argument(
        "--source",
        default="raw_inventory",
        help="Source name to store in the cleaned inventory.",
    )
    parser.add_argument(
        "--mapped-inventory-output",
        default="data/inventory/kerala_uttarakhand_inventory_mapped.csv",
        help="Path to write the mapped inventory CSV.",
    )
    parser.add_argument(
        "--evaluation-json",
        default="artifacts/kerala_uttarakhand_real_evaluation.json",
        help="Path to write the evaluation JSON.",
    )
    parser.add_argument(
        "--evaluation-markdown",
        default="docs/KERALA_UTTARAKHAND_REAL_EVALUATION.md",
        help="Path to write the evaluation Markdown.",
    )
    args = parser.parse_args()

    mapped_inventory_path = write_coordinate_mapped_inventory(
        input_csv_path=args.raw_inventory,
        output_csv_path=args.mapped_inventory_output,
        region_metadata_csv_path="data/raw/india_rainfall.csv",
        source_name=args.source,
        allowed_states=TARGET_STATES,
    )
    start_date, end_date = _derive_nasa_date_range(mapped_inventory_path)
    os.environ["LANDSLIDE_EVENT_INVENTORY_CSV"] = str(Path(mapped_inventory_path).resolve())
    os.environ["LANDSLIDE_NASA_POWER_START_DATE"] = start_date
    os.environ["LANDSLIDE_NASA_POWER_END_DATE"] = end_date

    paths, result = run_target_state_real_evaluation(
        project_root=".",
        landslide_inventory_csv_path=str(mapped_inventory_path),
        states=TARGET_STATES,
    )
    write_large_scale_evaluation(
        result,
        json_path=args.evaluation_json,
        markdown_path=args.evaluation_markdown,
    )

    print("Kerala/Uttarakhand real-data build complete")
    print(f"Mapped inventory: {mapped_inventory_path}")
    print(f"NASA date range: {start_date} to {end_date}")
    print(f"Region file: {paths.region_csv}")
    print(f"NASA daily rainfall: {paths.nasa_daily_csv}")
    print(f"Merged risk dataset: {paths.risk_dataset_csv}")
    print(f"Evaluation JSON: {args.evaluation_json}")
    print(f"Evaluation Markdown: {args.evaluation_markdown}")
    print(f"Label source: {result.label_source}")
    print(
        "Metrics: "
        f"accuracy={result.metrics.accuracy:.3f}, "
        f"precision={result.metrics.precision:.3f}, "
        f"recall={result.metrics.recall:.3f}, "
        f"f1={result.metrics.f1:.3f}, "
        f"roc_auc={result.metrics.roc_auc:.3f}"
    )


if __name__ == "__main__":
    main()
