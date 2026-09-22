from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from landslide_ai.graph.spatiotemporal import run_spatiotemporal_inference_from_frame


def generate_case_study_artifact(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    st_artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    output_json_path: str | Path = "artifacts/case_studies.json",
) -> list[dict[str, object]]:
    frame = pd.read_csv(dataset_csv_path)
    cache_frame = _load_cached_spatiotemporal_frame()
    explanation = cache_frame if cache_frame is not None else run_spatiotemporal_inference_from_frame(frame, st_artifact_path, sequence_length=10)
    merged = explanation.merge(
        frame[["region_id", "date", "label", "rainfall_24h_mm", "rainfall_7d_mm", "slope_deg", "ndvi"]].assign(
            date=lambda df: pd.to_datetime(df["date"]).dt.date.astype(str)
        ),
        on=["region_id", "date"],
        how="left",
    )
    top_rows = (
        merged.sort_values(["spatiotemporal_gnn_probability", "label"], ascending=[False, False])
        .drop_duplicates(subset=["district"])
        .head(3)
    )
    case_studies: list[dict[str, object]] = []
    for row in top_rows.to_dict(orient="records"):
        district_history = (
            merged.loc[merged["district"] == row["district"], ["date", "spatiotemporal_gnn_probability", "rainfall_7d_mm", "label"]]
            .sort_values("date", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
        case_studies.append(
            {
                "district": row["district"],
                "state": row["state"],
                "reference_date": row["date"],
                "st_gnn_score": row["spatiotemporal_gnn_probability"],
                "uncertainty_score": row["uncertainty_score"],
                "confidence_level": row["confidence_level"],
                "recommended_interpretation": row["recommended_interpretation"],
                "top_driver": row["top_driver"],
                "top_feature_drivers": row["top_feature_drivers"],
                "top_neighbor_influences": row["top_neighbor_influences"],
                "rainfall_24h_mm": row.get("rainfall_24h_mm"),
                "rainfall_7d_mm": row.get("rainfall_7d_mm"),
                "slope_deg": row.get("slope_deg"),
                "ndvi": row.get("ndvi"),
                "label": row.get("label"),
                "recent_history": district_history,
            }
        )
    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(case_studies, indent=2, default=str), encoding="utf-8")
    return case_studies


def _load_cached_spatiotemporal_frame() -> pd.DataFrame | None:
    cache_dir = Path("artifacts/cache")
    candidates = sorted(cache_dir.glob("spatiotemporal_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return pd.read_csv(candidates[0])
