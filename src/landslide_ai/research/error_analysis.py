from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from landslide_ai.graph.spatiotemporal import run_spatiotemporal_inference_from_frame


def generate_error_analysis_artifact(
    dataset_csv_path: str | Path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv",
    st_artifact_path: str | Path = "artifacts/spatiotemporal_gnn.pt",
    threshold: float = 0.6650654077529907,
    output_json_path: str | Path = "artifacts/error_analysis.json",
) -> dict[str, object]:
    frame = pd.read_csv(dataset_csv_path)
    cache_frame = _load_cached_spatiotemporal_frame()
    explanation = cache_frame if cache_frame is not None else run_spatiotemporal_inference_from_frame(frame, st_artifact_path, sequence_length=10)
    merged = explanation.merge(
        frame[["region_id", "date", "label", "state", "district", "rainfall_7d_mm", "slope_deg", "ndvi"]].assign(
            date=lambda df: pd.to_datetime(df["date"]).dt.date.astype(str)
        ),
        on=["region_id", "date", "state", "district"],
        how="left",
    )
    merged["season"] = pd.to_datetime(merged["date"]).dt.month.map(_season_name)
    merged["prediction"] = (merged["spatiotemporal_gnn_probability"] >= threshold).astype(int)
    false_positives = merged[(merged["prediction"] == 1) & (merged["label"] == 0)].copy()
    false_negatives = merged[(merged["prediction"] == 0) & (merged["label"] == 1)].copy()
    payload = {
        "threshold": threshold,
        "false_positive_count": int(len(false_positives)),
        "false_negative_count": int(len(false_negatives)),
        "false_positive_states": false_positives["state"].value_counts().head(8).to_dict(),
        "false_negative_states": false_negatives["state"].value_counts().head(8).to_dict(),
        "false_positive_seasons": false_positives["season"].value_counts().to_dict(),
        "false_negative_seasons": false_negatives["season"].value_counts().to_dict(),
        "top_false_positives": false_positives.sort_values("spatiotemporal_gnn_probability", ascending=False).head(8).round(4).to_dict(orient="records"),
        "top_false_negatives": false_negatives.sort_values("spatiotemporal_gnn_probability", ascending=True).head(8).round(4).to_dict(orient="records"),
    }
    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _season_name(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Pre-monsoon"
    if month in {6, 7, 8, 9}:
        return "Monsoon"
    return "Post-monsoon"


def _load_cached_spatiotemporal_frame() -> pd.DataFrame | None:
    cache_dir = Path("artifacts/cache")
    candidates = sorted(cache_dir.glob("spatiotemporal_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    frame = pd.read_csv(candidates[0])
    return frame
