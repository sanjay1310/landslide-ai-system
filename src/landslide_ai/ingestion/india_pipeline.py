from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class IndiaDataPreparationResult:
    output_path: Path
    row_count: int
    columns: list[str]


def prepare_india_region_dataset(
    rainfall_csv: str | Path,
    terrain_csv: str | Path,
    output_csv: str | Path,
) -> IndiaDataPreparationResult:
    rainfall = pd.read_csv(rainfall_csv)
    terrain = pd.read_csv(terrain_csv)

    merged = rainfall.merge(terrain, on=["region_id", "state", "district"], how="inner")
    merged.to_csv(output_csv, index=False)

    return IndiaDataPreparationResult(
        output_path=Path(output_csv),
        row_count=len(merged),
        columns=list(merged.columns),
    )
