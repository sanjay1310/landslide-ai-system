from __future__ import annotations

import pandas as pd

from landslide_ai.models.schemas import RiskAssessment


def assessments_to_frame(results: list[RiskAssessment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Region ID": result.region_id,
                "State": result.state,
                "District": result.district,
                "Risk Level": result.risk_level,
                "Risk Score": round(result.risk_score, 3),
                "Forecast Risk": round(result.forecast_risk, 3),
                "Primary Driver": result.primary_driver,
                "Uncertainty": round(result.uncertainty_score, 3),
                "Calibrated Risk": round(result.calibrated_risk_score, 3),
                "Confidence Level": result.confidence_level,
                "Alert Band": result.alert_band,
                "Recommended Action": result.recommended_action,
                "Graph Exposure": round(result.graph_exposure, 3),
                "Propagated Graph Exposure": round(result.propagated_graph_exposure, 3),
                "Rainfall Exposure": round(result.rainfall_exposure, 3),
                "Terrain Exposure": round(result.terrain_exposure, 3),
                "Vegetation Exposure": round(result.vegetation_exposure, 3),
                "Latitude": result.latitude,
                "Longitude": result.longitude,
                "Advisory": result.advisory,
            }
            for result in results
        ]
    )


def build_tooltip_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["District"]
        + ", "
        + frame["State"]
        + " | Risk: "
        + frame["Risk Score"].astype(str)
    )


def risk_to_fill_color(score: float) -> list[int]:
    return [255, int(220 - (score * 120)), 60, 180]


def build_driver_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("Primary Driver", dropna=False)["Region ID"]
        .count()
        .reset_index(name="Region Count")
        .sort_values("Region Count", ascending=False)
        .reset_index(drop=True)
    )


def build_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("State", dropna=False)
        .agg(
            {
                "Risk Score": "mean",
                "Forecast Risk": "mean",
                "Uncertainty": "mean",
                "Region ID": "count",
            }
        )
        .rename(columns={"Region ID": "Region Count"})
        .round(3)
        .reset_index()
        .sort_values("Risk Score", ascending=False)
        .reset_index(drop=True)
    )


def build_risk_band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("Risk Level", dropna=False)["Region ID"]
        .count()
        .reset_index(name="Region Count")
        .sort_values("Region Count", ascending=False)
        .reset_index(drop=True)
    )


def build_scenario_delta_frame(base_frame: pd.DataFrame, scenario_frame: pd.DataFrame) -> pd.DataFrame:
    merged = scenario_frame.merge(
        base_frame[["Region ID", "Risk Score", "Forecast Risk"]].rename(
            columns={
                "Risk Score": "Base Risk Score",
                "Forecast Risk": "Base Forecast Risk",
            }
        ),
        on="Region ID",
        how="left",
    )
    merged["Risk Delta"] = (merged["Risk Score"] - merged["Base Risk Score"]).round(3)
    merged["Forecast Delta"] = (merged["Forecast Risk"] - merged["Base Forecast Risk"]).round(3)
    merged["Absolute Risk Delta"] = merged["Risk Delta"].abs().round(3)
    return merged.sort_values(["Absolute Risk Delta", "Risk Delta"], ascending=False).reset_index(drop=True)


def build_top_regions(frame: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    return (
        frame.sort_values(["Risk Score", "Forecast Risk"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
