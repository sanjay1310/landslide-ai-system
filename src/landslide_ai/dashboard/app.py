from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import sklearn
import streamlit as st

from landslide_ai.alerts.generator import build_alerts
from landslide_ai.data.geojson_loader import load_geojson
from landslide_ai.data.india_sources import default_india_dataset_paths, dataset_status
from landslide_ai.dashboard.view_models import (
    assessments_to_frame,
    build_driver_summary,
    build_risk_band_summary,
    build_scenario_delta_frame,
    build_state_summary,
    build_tooltip_label,
    build_top_regions,
    risk_to_fill_color,
)
from landslide_ai.forecasting.lstm import torch_available
from landslide_ai.reporting.exporters import (
    assessment_summary_to_json_bytes,
    alerts_to_csv_bytes,
    assessments_to_csv_bytes,
    forecast_summary_to_json_bytes,
    lstm_forecast_summary_to_json_bytes,
)
from landslide_ai.services.forecast_service import ForecastInferenceService, default_forecast_artifacts
from landslide_ai.services.system_service import LandslideSystemService
from landslide_ai.training.train_forecaster import train_rainfall_forecaster_from_windows
from landslide_ai.training.train_gnn import train_graph_risk_model
from landslide_ai.training.train_lstm_forecaster import train_lstm_forecaster_from_windows


st.set_page_config(
    page_title="Landslide Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink-strong: #163247;
        --ink-soft: #4c5d66;
        --panel-bg: rgba(255,255,255,0.92);
        --panel-border: rgba(18, 52, 71, 0.10);
        --accent-blue: #1a5c7a;
        --accent-orange: #d96d2d;
        --accent-sand: #f3e2cf;
        --accent-green: #2f7d5d;
        --accent-red: #b54b3f;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(217, 109, 45, 0.16), transparent 24%),
            radial-gradient(circle at 82% 8%, rgba(26, 92, 122, 0.16), transparent 26%),
            linear-gradient(180deg, #faf6ef 0%, #efe5d6 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }
    .hero-shell {
        background:
            radial-gradient(circle at 20% 20%, rgba(255,255,255,0.14), transparent 22%),
            linear-gradient(135deg, #102c3e 0%, #1a5c7a 48%, #d96d2d 100%);
        color: #f6efe6;
        padding: 1.9rem 2rem 1.5rem 2rem;
        border-radius: 28px;
        box-shadow: 0 24px 52px rgba(22, 56, 79, 0.16);
        margin-bottom: 1.1rem;
    }
    .hero-kicker {
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.75rem;
        opacity: 0.82;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.45rem;
        font-weight: 700;
        line-height: 1.05;
        margin-bottom: 0.45rem;
        max-width: 920px;
    }
    .hero-copy {
        font-size: 1rem;
        max-width: 860px;
        opacity: 0.92;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.8fr 1fr;
        gap: 1rem;
        align-items: start;
    }
    .hero-side {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 22px;
        padding: 1rem 1rem 0.8rem 1rem;
    }
    .hero-side-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.7rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    .hero-side-value {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .hero-side-copy {
        font-size: 0.88rem;
        opacity: 0.9;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.8rem;
    }
    .chip {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.20);
        color: #fff7ee;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.8rem;
    }
    .metric-card {
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-top: 5px solid var(--accent-blue);
        padding: 0.95rem 1.05rem;
        border-radius: 18px;
        box-shadow: 0 12px 24px rgba(22, 56, 79, 0.08);
        min-height: 122px;
        backdrop-filter: blur(8px);
    }
    .metric-card.risk {
        border-top-color: var(--accent-red);
    }
    .metric-card.forecast {
        border-top-color: var(--accent-orange);
    }
    .metric-card.alert {
        border-top-color: var(--accent-green);
    }
    .metric-card.graph {
        border-top-color: var(--accent-blue);
    }
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        color: #61727b;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        font-size: 1.95rem;
        font-weight: 700;
        color: var(--ink-strong);
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .metric-sub {
        font-size: 0.88rem;
        color: var(--ink-soft);
    }
    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.74rem;
        color: #966038;
        margin-bottom: 0.25rem;
    }
    .panel-card {
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        padding: 1rem 1rem 0.65rem 1rem;
        box-shadow: 0 10px 24px rgba(24, 61, 83, 0.08);
    }
    .insight-band {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0 1rem 0;
    }
    .insight-card {
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(24, 61, 83, 0.10);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 10px 22px rgba(24, 61, 83, 0.07);
    }
    .insight-card strong {
        display: block;
        color: #153348;
        margin-bottom: 0.25rem;
    }
    .insight-card span {
        color: #53626a;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background: rgba(255,255,255,0.86);
        border-radius: 14px;
        border: 1px solid rgba(22, 56, 79, 0.10);
        padding: 0 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #163247 !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-weight: 600;
        color: var(--ink-strong) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #fffaf2 !important;
    }
    .block-container [data-testid="stMarkdownContainer"] p,
    .block-container [data-testid="stMarkdownContainer"] li,
    .block-container [data-testid="stMarkdownContainer"] h1,
    .block-container [data-testid="stMarkdownContainer"] h2,
    .block-container [data-testid="stMarkdownContainer"] h3,
    .block-container [data-testid="stMarkdownContainer"] h4,
    .block-container label,
    .block-container [data-testid="stWidgetLabel"] {
        color: var(--ink-strong) !important;
    }
    .block-container [data-testid="stMarkdownContainer"] small,
    .block-container [data-testid="stMarkdownContainer"] span,
    .block-container [data-testid="stCaptionContainer"] {
        color: var(--ink-soft) !important;
    }
    .panel-card,
    .panel-card p,
    .panel-card h1,
    .panel-card h2,
    .panel-card h3,
    .panel-card h4,
    .panel-card strong,
    .panel-card span {
        color: var(--ink-strong) !important;
    }
    [data-baseweb="select"] > div {
        color: var(--ink-strong);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fffaf2 0%, #f2eadf 100%);
        border-right: 1px solid rgba(20, 50, 70, 0.08);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: var(--ink-strong);
    }
    @media (max-width: 1100px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .insight-band {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric_card(label: str, value: str, subtext: str, tone: str = "graph") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_round(value: object, digits: int = 3) -> object:
    if value is None or value == "":
        return "n/a"
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _confidence_rank(label: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(label), 0)


def _model_disagreement_summary(
    traditional_score: float,
    graph_score: float | None,
    spatiotemporal_score: float | None,
) -> str:
    available_scores = {
        "traditional": float(traditional_score),
        "graph": None if graph_score is None or pd.isna(graph_score) else float(graph_score),
        "spatiotemporal": None if spatiotemporal_score is None or pd.isna(spatiotemporal_score) else float(spatiotemporal_score),
    }
    compact = {name: score for name, score in available_scores.items() if score is not None}
    if len(compact) < 2:
        return "Only one model path is available for this district, so disagreement analysis is limited."
    spread = max(compact.values()) - min(compact.values())
    if spread <= 0.08:
        return "The models are broadly aligned, which makes this district-level picture more stable."
    highest_model = max(compact.items(), key=lambda item: item[1])[0]
    if highest_model == "spatiotemporal":
        return "The ST-GNN is materially higher than the other models, suggesting temporal rainfall memory is adding extra concern."
    if highest_model == "graph":
        return "The static graph GNN is materially higher than the others, suggesting stronger spatial spillover or neighborhood pressure."
    return "The traditional model is higher than the graph models here, suggesting the local environmental features are stronger than the current graph signal."


@st.cache_resource(show_spinner=False)
def get_system_service() -> LandslideSystemService:
    return LandslideSystemService(".")


@st.cache_data(show_spinner=False)
def get_geojson(path: str) -> dict[str, object]:
    return load_geojson(Path(path))


@st.cache_data(show_spinner=False)
def get_assessment_bundle(
    csv_path: str,
    rainfall_multiplier: float,
    antecedent_multiplier: float,
    soil_delta: float,
    ndvi_delta: float,
    neighbor_delta: float,
    scenario_state: str,
) -> dict[str, object]:
    service = get_system_service()
    base_results = service.assess_csv(csv_path)
    scenario_results = service.assess_scenario(
        csv_path=csv_path,
        rainfall_multiplier=rainfall_multiplier,
        antecedent_rainfall_multiplier=antecedent_multiplier,
        soil_wetness_delta=soil_delta,
        ndvi_delta=ndvi_delta,
        neighbor_risk_delta=neighbor_delta,
        target_state=None if scenario_state == "All" else scenario_state,
    )
    base_table = assessments_to_frame(base_results)
    scenario_table = assessments_to_frame(scenario_results)
    alert_frame = pd.DataFrame([asdict(alert) for alert in build_alerts(scenario_results)])
    return {
        "base_table": base_table,
        "scenario_table": scenario_table,
        "alert_frame": alert_frame,
    }


@st.cache_data(show_spinner=False)
def get_forecast_bundle(windows_path: str) -> dict[str, object]:
    forecast_path = Path(windows_path)
    if not forecast_path.exists():
        return {"status": "missing"}

    try:
        frame = pd.read_csv(forecast_path)
    except pd.errors.EmptyDataError:
        return {"status": "empty"}

    if frame.empty:
        return {"status": "empty"}

    service = ForecastInferenceService(default_forecast_artifacts(Path(".")))
    if service.available():
        try:
            predicted = service.predict_frame(frame)
            return {
                "status": "artifact",
                "frame": predicted,
                "source": str(service.artifacts.model_path),
                "metadata": service.metadata or {},
            }
        except Exception as exc:
            return {
                "status": "artifact_error",
                "frame": frame,
                "source": str(service.artifacts.model_path),
                "message": str(exc),
            }

    return {
        "status": "raw",
        "frame": frame,
    }


@st.cache_data(show_spinner=False)
def get_graph_bundle(csv_path: str) -> dict[str, object]:
    service = get_system_service()
    graph_status = service.graph_runtime_status()
    artifact_path = graph_status["artifact_path"]
    if not graph_status["configured_enabled"]:
        return {"status": "disabled", "artifact_path": str(artifact_path)}
    if not graph_status["torch_available"]:
        return {"status": "torch_missing", "artifact_path": str(artifact_path)}
    if not graph_status["artifact_exists"]:
        return {"status": "artifact_missing", "artifact_path": str(artifact_path)}
    try:
        frame = service.graph_inference_from_csv(csv_path, artifact_path)
        return {"status": "ready", "frame": frame, "artifact_path": str(artifact_path)}
    except Exception as exc:
        return {
            "status": "error",
            "artifact_path": str(artifact_path),
            "message": str(exc),
        }


@st.cache_data(show_spinner=False)
def get_spatiotemporal_bundle(csv_path: str) -> dict[str, object]:
    service = get_system_service()
    runtime_status = service.spatiotemporal_runtime_status()
    artifact_path = runtime_status["artifact_path"]
    if not runtime_status["configured_enabled"]:
        return {"status": "disabled", "artifact_path": str(artifact_path)}
    if not runtime_status["torch_available"]:
        return {"status": "torch_missing", "artifact_path": str(artifact_path)}
    if not runtime_status["artifact_exists"]:
        return {"status": "artifact_missing", "artifact_path": str(artifact_path)}
    try:
        frame = service.spatiotemporal_graph_inference_from_csv(csv_path, artifact_path)
        return {"status": "ready", "frame": frame, "artifact_path": str(artifact_path)}
    except Exception as exc:
        return {
            "status": "error",
            "artifact_path": str(artifact_path),
            "message": str(exc),
        }


@st.cache_data(show_spinner=False)
def run_knowledge_query(query: str, top_k: int) -> dict[str, object]:
    return get_system_service().knowledge_query(query=query, top_k=top_k)


@st.cache_data(show_spinner=False)
def get_data_provenance(region_csv_path: str, imd_csv_path: str, nasa_csv_path: str) -> dict[str, object]:
    provenance = {
        "region_data_mode": "unknown",
        "nasa_mode": "missing",
        "imd_mode": "missing",
        "imd_rows": 0,
    }
    region_path = Path(region_csv_path)
    if region_path.exists():
        region_frame = pd.read_csv(region_path)
        if "source_mode" in region_frame.columns and not region_frame.empty:
            provenance["region_data_mode"] = ", ".join(sorted(region_frame["source_mode"].astype(str).unique().tolist()))
        else:
            provenance["region_data_mode"] = "seeded"

    nasa_path = Path(nasa_csv_path)
    if nasa_path.exists():
        nasa_frame = pd.read_csv(nasa_path)
        if "source_mode" in nasa_frame.columns and not nasa_frame.empty:
            provenance["nasa_mode"] = ", ".join(sorted(nasa_frame["source_mode"].astype(str).unique().tolist()))

    imd_path = Path(imd_csv_path)
    if imd_path.exists():
        try:
            imd_frame = pd.read_csv(imd_path)
            provenance["imd_rows"] = len(imd_frame)
            provenance["imd_mode"] = "overlay_ready" if len(imd_frame) > 0 else "placeholder"
        except Exception:
            provenance["imd_mode"] = "unreadable"

    return provenance


@st.cache_data(show_spinner=False)
def get_model_metadata_bundle() -> dict[str, object]:
    service = get_system_service()
    runtime_versions = {
        "python": __import__("platform").python_version(),
        "scikit_learn": sklearn.__version__,
        "torch": None,
    }
    if torch_available():
        try:
            import torch

            runtime_versions["torch"] = torch.__version__
        except Exception:
            runtime_versions["torch"] = "unavailable"

    def load_json_if_exists(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    risk_metadata = load_json_if_exists(service.config.risk_model_artifact.with_name("risk_model_metadata.json"))
    forecast_metadata = load_json_if_exists(Path("artifacts/rainfall_baseline_metadata.json"))
    graph_runtime = service.graph_runtime_status()
    st_graph_runtime = service.spatiotemporal_runtime_status()
    st_graph_metadata = load_json_if_exists(Path(st_graph_runtime["artifact_path"]).with_name("spatiotemporal_gnn_metadata.json"))

    return {
        "runtime_versions": runtime_versions,
        "risk_model": {
            "artifact_path": str(service.config.risk_model_artifact),
            "artifact_exists": service.config.risk_model_artifact.exists(),
            "metadata": risk_metadata,
            "version_match": risk_metadata.get("scikit_learn_version") == runtime_versions["scikit_learn"]
            if risk_metadata
            else None,
        },
        "forecast_model": {
            "artifact_path": str(Path("artifacts/rainfall_baseline.pkl")),
            "artifact_exists": Path("artifacts/rainfall_baseline.pkl").exists(),
            "metadata": forecast_metadata,
            "version_match": forecast_metadata.get("scikit_learn_version") == runtime_versions["scikit_learn"]
            if forecast_metadata
            else None,
        },
        "graph_model": {
            "artifact_path": graph_runtime["artifact_path"],
            "artifact_exists": graph_runtime["artifact_exists"],
            "runtime_ready": graph_runtime["runtime_ready"],
            "torch_version": runtime_versions["torch"],
        },
        "spatiotemporal_graph_model": {
            "artifact_path": st_graph_runtime["artifact_path"],
            "artifact_exists": st_graph_runtime["artifact_exists"],
            "runtime_ready": st_graph_runtime["runtime_ready"],
            "metadata": st_graph_metadata,
            "version_match": st_graph_metadata.get("scikit_learn_version") == runtime_versions["scikit_learn"]
            if st_graph_metadata
            else None,
        },
    }


@st.cache_data(show_spinner=False)
def get_traditional_vs_gnn_bundle() -> dict[str, object]:
    summary_path = Path("artifacts/merged_real_benchmark_summary.json")
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_spatiotemporal_metrics_bundle() -> dict[str, object]:
    metadata_path = Path("artifacts/spatiotemporal_gnn_metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_advanced_research_bundle() -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, path in {
        "ablation": Path("artifacts/ablation_study.json"),
        "spatiotemporal_ablation": Path("artifacts/spatiotemporal_ablation.json"),
        "spatiotemporal_ablation_fast": Path("artifacts/spatiotemporal_ablation_fast.json"),
        "explainability": Path("artifacts/explainability_snapshot.json"),
        "reliability": Path("artifacts/reliability_curves.json"),
        "st_calibration": Path("artifacts/st_gnn_calibration.json"),
        "error_analysis": Path("artifacts/error_analysis.json"),
        "case_studies": Path("artifacts/case_studies.json"),
        "external_baselines": Path("artifacts/external_baseline_comparison.json"),
        "advanced_markdown": Path("docs/ADVANCED_RESEARCH_SUITE.md"),
        "large_scale": Path("artifacts/large_scale_risk_evaluation.json"),
    }.items():
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                payload[key] = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload[key] = path.read_text(encoding="utf-8")
        except Exception:
            continue
    return payload


def build_map_layers(
    map_frame: pd.DataFrame,
    map_mode: str,
    district_geojson_path: Path,
    state_geojson_path: Path,
    state_summary: pd.DataFrame,
) -> list[pdk.Layer]:
    if map_mode == "District boundaries" and district_geojson_path.exists():
        geojson = get_geojson(str(district_geojson_path))
        boundary_frame = map_frame[["Region ID", "State", "District", "Risk Score", "Forecast Risk"]]
        risk_by_region = boundary_frame.set_index("Region ID").to_dict(orient="index")
        risk_by_location = {
            (row["State"].strip().lower(), row["District"].strip().lower()): row
            for row in boundary_frame.to_dict(orient="records")
        }
        for feature in geojson.get("features", []):
            properties = feature.setdefault("properties", {})
            region_risk = risk_by_region.get(properties.get("region_id"))
            if not region_risk:
                state_name = str(properties.get("State") or properties.get("NAME_1") or "").strip()
                district_name = str(properties.get("District") or properties.get("NAME_2") or "").strip()
                region_risk = risk_by_location.get((state_name.lower(), district_name.lower()))
            if region_risk:
                properties["District"] = region_risk["District"]
                properties["State"] = region_risk["State"]
                properties["Risk Score"] = region_risk["Risk Score"]
                properties["Forecast Risk"] = region_risk["Forecast Risk"]
        return [
            pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                opacity=0.64,
                stroked=True,
                filled=True,
                get_fill_color="[226, 103 - properties['Risk Score'] * 70, 52, 180]",
                get_line_color=[25, 44, 59, 120],
                pickable=True,
            )
        ]

    if map_mode == "State choropleth" and state_geojson_path.exists():
        geojson = get_geojson(str(state_geojson_path))
        state_lookup = {
            row["State"].strip().lower(): row
            for row in state_summary.to_dict(orient="records")
        }
        for feature in geojson.get("features", []):
            properties = feature.setdefault("properties", {})
            state_name = str(properties.get("State") or properties.get("NAME_1") or "").strip().lower()
            row = state_lookup.get(state_name)
            if row:
                properties["State"] = row["State"]
                properties["Risk Score"] = row["Risk Score"]
                properties["Forecast Risk"] = row["Forecast Risk"]
                properties["Uncertainty"] = row["Uncertainty"]
                properties["Region Count"] = row["Region Count"]
        return [
            pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                opacity=0.58,
                stroked=True,
                filled=True,
                get_fill_color="[22, 56 + properties['Risk Score'] * 120, 79 + properties['Forecast Risk'] * 60, 180]",
                get_line_color=[18, 34, 46, 140],
                pickable=True,
            )
        ]

    return [
        pdk.Layer(
            "HeatmapLayer",
            data=map_frame,
            get_position=["lon", "lat"],
            get_weight="weight",
            radiusPixels=58,
            intensity=1.0,
            threshold=0.1,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=map_frame,
            get_position=["lon", "lat"],
            get_radius=22000,
            get_fill_color="fill_color",
            get_line_color=[20, 41, 54, 140],
            line_width_min_pixels=1,
            pickable=True,
        ),
    ]


def build_tooltip_html(map_mode: str) -> str:
    if map_mode == "State choropleth":
        return (
            "<b>{State}</b><br/>Avg risk: {Risk Score}"
            "<br/>Forecast: {Forecast Risk}<br/>Regions: {Region Count}"
        )
    return "<b>{District}</b><br/>{State}<br/>Risk: {Risk Score}<br/>Forecast: {Forecast Risk}"


paths = default_india_dataset_paths(Path("."))
status = dataset_status(Path("."))
csv_path = str(paths.region_csv) if status["region_csv_exists"] else "data/sample_regions.csv"
windows_csv_path = "data/india/rainfall_windows.csv"
imd_csv_path = "data/raw/imd/imd_latest_normalized.csv"
nasa_latest_csv_path = "data/raw/nasa_power/nasa_power_latest.csv"
regional_spatiotemporal_csv_path = "data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv"

records_frame = pd.read_csv(csv_path)
state_choices = sorted(records_frame["state"].astype(str).unique().tolist())
default_focus_state = "Kerala" if "Kerala" in state_choices else state_choices[0]
provenance = get_data_provenance(csv_path, imd_csv_path, nasa_latest_csv_path)
model_metadata = get_model_metadata_bundle()
benchmark_bundle = get_traditional_vs_gnn_bundle()
spatiotemporal_metrics = get_spatiotemporal_metrics_bundle()
advanced_research_bundle = get_advanced_research_bundle()

with st.sidebar:
    st.markdown("### Scenario Controls")
    st.caption("Tune rainfall, soil, vegetation, and spatial spillover without recomputing the whole stack from scratch.")
    scenario_state = st.selectbox(
        "Scenario scope",
        ["All"] + state_choices,
        index=(["All"] + state_choices).index(default_focus_state),
    )
    rainfall_multiplier = st.slider("24h rainfall multiplier", 0.5, 2.0, 1.0, 0.05)
    antecedent_multiplier = st.slider("7d rainfall multiplier", 0.5, 2.0, 1.0, 0.05)
    soil_delta = st.slider("Soil wetness delta", -0.30, 0.30, 0.0, 0.01)
    ndvi_delta = st.slider("NDVI delta", -0.30, 0.30, 0.0, 0.01)
    neighbor_delta = st.slider("Network spillover delta", -0.30, 0.30, 0.0, 0.01)
    st.markdown("---")
    st.markdown("### View Controls")
    selected_state = st.selectbox(
        "Display state filter",
        ["All"] + state_choices,
        index=(["All"] + state_choices).index(default_focus_state),
    )
    map_mode_options = ["Heat points"]
    if status["district_geojson_exists"]:
        map_mode_options.append("District boundaries")
    if status["state_geojson_exists"]:
        map_mode_options.append("State choropleth")
    preferred_map_mode = "District boundaries" if "District boundaries" in map_mode_options else map_mode_options[0]
    selected_map_mode = st.radio("Map mode", map_mode_options, index=map_mode_options.index(preferred_map_mode))
    with st.expander("Dataset status", expanded=False):
        st.write(
            {
                "active_region_csv": csv_path,
                "india_region_csv_found": status["region_csv_exists"],
                "district_geojson_found": status["district_geojson_exists"],
                "state_geojson_found": status["state_geojson_exists"],
                "forecast_windows_found": Path(windows_csv_path).exists(),
                "region_data_mode": provenance["region_data_mode"],
                "nasa_mode": provenance["nasa_mode"],
                "imd_mode": provenance["imd_mode"],
                "imd_rows": provenance["imd_rows"],
            }
        )
    with st.expander("Model artifact status", expanded=False):
        st.write(
            {
                "runtime_versions": model_metadata["runtime_versions"],
                "risk_model": {
                    "artifact_exists": model_metadata["risk_model"]["artifact_exists"],
                    "artifact_path": model_metadata["risk_model"]["artifact_path"],
                    "training_sklearn_version": model_metadata["risk_model"]["metadata"].get("scikit_learn_version", "unknown"),
                    "version_match": model_metadata["risk_model"]["version_match"],
                },
                "forecast_model": {
                    "artifact_exists": model_metadata["forecast_model"]["artifact_exists"],
                    "artifact_path": model_metadata["forecast_model"]["artifact_path"],
                    "training_sklearn_version": model_metadata["forecast_model"]["metadata"].get("scikit_learn_version", "unknown"),
                    "version_match": model_metadata["forecast_model"]["version_match"],
                },
                "graph_model": model_metadata["graph_model"],
                "spatiotemporal_graph_model": model_metadata["spatiotemporal_graph_model"],
            }
        )
    with st.expander("Live refresh", expanded=False):
        st.caption("Pull the latest weather and satellite snapshots and clear cached dashboard bundles.")
        if st.button("Refresh real-world sources", width="stretch"):
            with st.spinner("Refreshing NASA, Sentinel, and regional weather inputs..."):
                refresh_payload = get_system_service().refresh_real_data_sources()
                get_assessment_bundle.clear()
                get_forecast_bundle.clear()
                get_graph_bundle.clear()
                get_spatiotemporal_bundle.clear()
                get_data_provenance.clear()
                st.session_state["refresh_payload"] = refresh_payload
            st.success("Refresh complete. The dashboard will use the updated inputs on the next rerun.")
        if "refresh_payload" in st.session_state:
            st.write(st.session_state["refresh_payload"])

bundle = get_assessment_bundle(
    csv_path=csv_path,
    rainfall_multiplier=rainfall_multiplier,
    antecedent_multiplier=antecedent_multiplier,
    soil_delta=soil_delta,
    ndvi_delta=ndvi_delta,
    neighbor_delta=neighbor_delta,
    scenario_state=scenario_state,
)

base_table = bundle["base_table"].copy()
table = bundle["scenario_table"].copy()
alert_frame = bundle["alert_frame"].copy()
state_summary = build_state_summary(table)
driver_summary = build_driver_summary(table)
risk_band_summary = build_risk_band_summary(table)
scenario_delta = build_scenario_delta_frame(base_table, table)
top_regions = build_top_regions(table)
graph_bundle = get_graph_bundle(csv_path)

if graph_bundle["status"] == "ready":
    graph_frame = graph_bundle["frame"][["region_id", "graph_gnn_probability"]].rename(
        columns={
            "region_id": "Region ID",
            "graph_gnn_probability": "Graph GNN Probability",
        }
    )
    base_table = base_table.merge(graph_frame, on="Region ID", how="left")
    table = table.merge(graph_frame, on="Region ID", how="left")
    scenario_delta = scenario_delta.merge(graph_frame, on="Region ID", how="left")
    top_regions = build_top_regions(table)
else:
    table["Graph GNN Probability"] = pd.NA
    base_table["Graph GNN Probability"] = pd.NA
    scenario_delta["Graph GNN Probability"] = pd.NA
    top_regions = build_top_regions(table)

if selected_state != "All":
    table = table[table["State"] == selected_state].reset_index(drop=True)
    scenario_delta = scenario_delta[scenario_delta["State"] == selected_state].reset_index(drop=True)
    if not alert_frame.empty:
        alert_frame = alert_frame[alert_frame["state"] == selected_state].reset_index(drop=True)

chart_theme = {
    "axis": {
        "labelColor": "#334650",
        "titleColor": "#163247",
        "gridColor": "#d9e2e8",
        "tickColor": "#9eb0bb",
        "domainColor": "#b7c5cd",
    },
    "view": {"stroke": "#dbe5ea"},
    "legend": {"labelColor": "#334650", "titleColor": "#163247"},
    "title": {"color": "#163247", "fontSize": 16, "anchor": "start"},
}

top_row = table.sort_values(["Risk Score", "Forecast Risk"], ascending=False).iloc[0]
map_frame = table.rename(columns={"Latitude": "lat", "Longitude": "lon"}).copy()
map_frame["weight"] = map_frame["Risk Score"] * 100
map_frame["fill_color"] = map_frame["Risk Score"].apply(risk_to_fill_color)
map_frame["tooltip_label"] = build_tooltip_label(map_frame)
high_alert_count = int((table["Risk Level"] == "High").sum())
forecast_gap = (table["Forecast Risk"] - table["Risk Score"]).mean()
hotspot_state = str(state_summary.iloc[0]["State"]) if not state_summary.empty else "n/a"
graph_ready = graph_bundle["status"] == "ready"
spatiotemporal_ready = bool(model_metadata["spatiotemporal_graph_model"]["artifact_exists"])
data_story = (
    "Latest NASA rainfall inputs are available and fused with terrain and vegetation signals."
    if "downloaded" in str(provenance["nasa_mode"])
    else "This run is using prepared regional data with fallback weather inputs for demo stability."
)

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Landslide Intelligence Platform</div>
                <div class="hero-title">India Landslide Risk Dashboard for Monitoring, Forecasting, and Spatial Analysis</div>
                <div class="hero-copy">
                    Track district hot spots, compare scenario changes, inspect neighbor spillover, and review forecast outputs
                    from one clear operational workspace.
                </div>
                <div class="chip-row">
                    <div class="chip">Scenario: {scenario_state}</div>
                    <div class="chip">View: {selected_state}</div>
                    <div class="chip">Map: {selected_map_mode}</div>
                    <div class="chip">NASA: {provenance["nasa_mode"]}</div>
                    <div class="chip">IMD: {provenance["imd_mode"]}</div>
                    <div class="chip">Dataset: {provenance["region_data_mode"]}</div>
                    <div class="chip">State GIS: {"Connected" if status["state_geojson_exists"] else "Missing"}</div>
                    <div class="chip">sklearn: {model_metadata["runtime_versions"]["scikit_learn"]}</div>
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-side-label">Current View</div>
                <div class="hero-side-value">{len(table)}</div>
                <div class="hero-side-copy">Districts visible in the current filtered view.</div>
                <div class="chip-row">
                    <div class="chip">Top state hotspot: {hotspot_state}</div>
                    <div class="chip">Graph model: {"Ready" if graph_ready else "Offline"}</div>
                    <div class="chip">ST-GNN: {"Ready" if spatiotemporal_ready else "Offline"}</div>
                </div>
            </div>
        </div>
        <div class="insight-band">
            <div class="insight-card">
                <strong>Data Source</strong>
                <span>{data_story}</span>
            </div>
            <div class="insight-card">
                <strong>Where To Focus</strong>
                <span>Use the state filter and scenario controls to quickly compare districts with stronger rainfall and terrain stress.</span>
            </div>
            <div class="insight-card">
                <strong>Model View</strong>
                <span>This dashboard combines trained risk scoring, forecast support, and graph spillover signals in one place.</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
with metric_col_1:
    render_metric_card("Top Risk District", str(top_row["District"]), str(top_row["State"]), tone="risk")
with metric_col_2:
    render_metric_card("Highest Current Risk", f"{top_row['Risk Score']:.2f}", f"Forecast {top_row['Forecast Risk']:.2f}", tone="risk")
with metric_col_3:
    render_metric_card("High-Risk Districts", str(high_alert_count), "Districts currently classified as High", tone="alert")
with metric_col_4:
    render_metric_card("Average Forecast Change", f"{forecast_gap:+.2f}", "Forecast risk minus current risk", tone="forecast")
with metric_col_5:
    if pd.notna(top_row.get("Graph GNN Probability")):
        render_metric_card("Graph Hotspot Score", f"{float(top_row['Graph GNN Probability']):.2f}", f"Main driver: {top_row['Primary Driver']}", tone="graph")
    else:
        render_metric_card("Main Driver", str(top_row["Primary Driver"]), f"Uncertainty {top_row['Uncertainty']:.2f}", tone="graph")

overview_tab, comparison_tab, alerts_tab, forecast_tab, details_tab, knowledge_tab, exports_tab = st.tabs(
    ["Overview", "Traditional vs GNN vs ST-GNN", "Alerts", "Forecast", "District Detail", "Knowledge Q&A", "Exports"]
)

with overview_tab:
    st.markdown('<div class="section-label">Map Overview</div>', unsafe_allow_html=True)
    overview_left, overview_right = st.columns([1.7, 1])
    with overview_left:
        layers = build_map_layers(
            map_frame=map_frame,
            map_mode=selected_map_mode,
            district_geojson_path=paths.district_geojson,
            state_geojson_path=paths.state_geojson,
            state_summary=state_summary,
        )
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=22.9734, longitude=78.6569, zoom=4.2),
                layers=layers,
                tooltip={"html": build_tooltip_html(selected_map_mode)},
            ),
            width="stretch",
        )
    with overview_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### State Summary")
        st.dataframe(state_summary, width="stretch", height=230)
        st.markdown("#### Risk Levels")
        st.dataframe(risk_band_summary, width="stretch", height=180)
        st.markdown("#### Main Drivers")
        st.dataframe(driver_summary, width="stretch", height=180)
        if graph_bundle["status"] == "ready":
            st.caption(f"Graph artifact: `{graph_bundle['artifact_path']}`")
        elif graph_bundle["status"] == "disabled":
            st.caption("Graph runtime is disabled in configuration. Set `LANDSLIDE_ENABLE_GRAPH_MODEL=true` to enable live GNN inference.")
        elif graph_bundle["status"] == "artifact_missing":
            st.caption("Graph artifact not found yet. Train with `PYTHONPATH=src python3 train_gnn.py`.")
        elif graph_bundle["status"] == "error":
            st.caption(f"Graph inference unavailable: {graph_bundle['message']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Artifact Metadata")
        metadata_frame = pd.DataFrame(
            [
                {
                    "Model": "Risk model",
                    "Artifact ready": model_metadata["risk_model"]["artifact_exists"],
                    "Training sklearn": model_metadata["risk_model"]["metadata"].get("scikit_learn_version", "unknown"),
                    "Runtime sklearn": model_metadata["runtime_versions"]["scikit_learn"],
                    "Version match": model_metadata["risk_model"]["version_match"],
                },
                {
                    "Model": "Forecast model",
                    "Artifact ready": model_metadata["forecast_model"]["artifact_exists"],
                    "Training sklearn": model_metadata["forecast_model"]["metadata"].get("scikit_learn_version", "unknown"),
                    "Runtime sklearn": model_metadata["runtime_versions"]["scikit_learn"],
                    "Version match": model_metadata["forecast_model"]["version_match"],
                },
                {
                    "Model": "Graph model",
                    "Artifact ready": model_metadata["graph_model"]["artifact_exists"],
                    "Training sklearn": "n/a",
                    "Runtime sklearn": model_metadata["runtime_versions"]["scikit_learn"],
                    "Version match": model_metadata["graph_model"]["runtime_ready"],
                },
                {
                    "Model": "Spatio-temporal GNN",
                    "Artifact ready": model_metadata["spatiotemporal_graph_model"]["artifact_exists"],
                    "Training sklearn": model_metadata["spatiotemporal_graph_model"]["metadata"].get("scikit_learn_version", "unknown"),
                    "Runtime sklearn": model_metadata["runtime_versions"]["scikit_learn"],
                    "Version match": model_metadata["spatiotemporal_graph_model"]["version_match"],
                },
            ]
        )
        st.dataframe(metadata_frame, width="stretch", height=185)

    st.markdown("#### Graph Model Guidance")
    st.markdown(
        """
        <div class="panel-card">
            <strong>What the GNN score means</strong><br/>
            The learned graph hotspot score estimates district risk after combining local environmental features with spatial neighborhood structure.
            Use it as a graph-aware hotspot indicator, not as a standalone operational probability.<br/><br/>
            <strong>How it differs from heuristic graph exposure</strong><br/>
            Graph exposure and propagated graph exposure are rule-based spatial signals derived from nearby districts. The GNN score is the learned graph model output and can highlight nonlinear graph patterns the heuristic path cannot capture.<br/><br/>
            <strong>When to trust it</strong><br/>
            Trust the GNN most when it agrees with rainfall, terrain, and propagated graph exposure. Treat isolated high GNN values as investigation prompts rather than final alerts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Spatio-Temporal GNN Guidance")
    st.markdown(
        """
        <div class="panel-card">
            <strong>What the spatio-temporal score means</strong><br/>
            The spatio-temporal GNN blends each district's recent rainfall sequence with graph message passing across nearby districts to estimate district-date risk under real inventory-linked supervision.<br/><br/>
            <strong>Why it differs from the graph hotspot score</strong><br/>
            The graph hotspot score is a static graph classifier on a reduced graph slice. The spatio-temporal GNN adds sequence memory, returns uncertainty-aware confidence labels, and explains which features and neighboring districts most influenced the prediction.<br/><br/>
            <strong>When to trust it</strong><br/>
            Trust it most when the predicted score is elevated and the confidence level is High or Medium. Low-confidence outputs should be treated as early-warning prompts rather than final operational alerts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    analytics_left, analytics_right = st.columns(2)
    with analytics_left:
        identity_line = (
            alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
            .mark_line(color="#8fa2ad", strokeDash=[5, 5], opacity=0.8)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
            )
        )
        risk_points = (
            alt.Chart(table)
            .mark_circle(opacity=0.88, stroke="#ffffff", strokeWidth=1)
            .encode(
                x=alt.X("Risk Score:Q", scale=alt.Scale(domain=[0, 1]), title="Current risk score"),
                y=alt.Y("Forecast Risk:Q", scale=alt.Scale(domain=[0, 1]), title="Forecast risk score"),
                size=alt.Size("Uncertainty:Q", scale=alt.Scale(range=[90, 520]), title="Uncertainty"),
                color=alt.Color(
                    "Risk Level:N",
                    scale=alt.Scale(
                        domain=["Low", "Moderate", "High"],
                        range=["#7aa6c2", "#edae49", "#c8553d"],
                    ),
                    title="Risk level",
                ),
                tooltip=[
                    "District",
                    "State",
                    alt.Tooltip("Risk Score:Q", format=".2f"),
                    alt.Tooltip("Forecast Risk:Q", format=".2f"),
                    alt.Tooltip("Uncertainty:Q", format=".2f"),
                    "Primary Driver",
                ],
            )
        )
        risk_scatter = (
            (identity_line + risk_points)
            .properties(height=320, title="Current Risk vs Forecast Risk")
            .configure(**chart_theme)
        )
        st.altair_chart(risk_scatter, width="stretch")
    with analytics_right:
        if graph_bundle["status"] == "ready":
            graph_ready_frame = table.dropna(subset=["Graph GNN Probability"]).copy()
            graph_compare = (
                alt.Chart(graph_ready_frame)
                .mark_circle(size=125, opacity=0.88, stroke="#ffffff", strokeWidth=1)
                .encode(
                    x=alt.X(
                        "Propagated Graph Exposure:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="Propagated graph exposure",
                    ),
                    y=alt.Y(
                        "Graph GNN Probability:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="Learned graph hotspot score",
                    ),
                    color=alt.Color(
                        "Risk Score:Q",
                        scale=alt.Scale(domain=[0, 1], range=["#b8d8e8", "#d96d2d", "#b54b3f"]),
                        title="Risk score",
                    ),
                    tooltip=[
                        "District",
                        "State",
                        alt.Tooltip("Graph Exposure:Q", format=".2f", title="Local graph exposure"),
                        alt.Tooltip("Propagated Graph Exposure:Q", format=".2f"),
                        alt.Tooltip("Graph GNN Probability:Q", format=".2f"),
                        alt.Tooltip("Risk Score:Q", format=".2f"),
                    ],
                )
                .properties(height=320, title="Graph Exposure vs Learned Graph Hotspot Score")
                .configure(**chart_theme)
            )
            st.altair_chart(graph_compare, width="stretch")
        else:
            state_bar = (
                alt.Chart(state_summary)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#16384f")
                .encode(
                    x=alt.X("State:N", sort="-y"),
                    y=alt.Y("Risk Score:Q"),
                    tooltip=["State", "Risk Score", "Forecast Risk", "Uncertainty", "Region Count"],
                )
                .properties(height=320, title="Average Risk by State")
                .configure(**chart_theme)
            )
            st.altair_chart(state_bar, width="stretch")

    movers_left, movers_right = st.columns([1.25, 1])
    with movers_left:
        scenario_top = scenario_delta.head(10).copy()
        delta_chart = (
            alt.Chart(scenario_top)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("District:N", sort="-y", title="District"),
                y=alt.Y("Risk Delta:Q", title="Change in risk score"),
                color=alt.condition(
                    alt.datum["Risk Delta"] >= 0,
                    alt.value("#d96d2d"),
                    alt.value("#4f88a8"),
                ),
                tooltip=[
                    "District",
                    "State",
                    alt.Tooltip("Risk Delta:Q", format="+.3f"),
                    alt.Tooltip("Forecast Delta:Q", format="+.3f"),
                    alt.Tooltip("Absolute Risk Delta:Q", format=".3f"),
                    "Primary Driver",
                ],
            )
            .properties(height=300, title="Largest Scenario Changes")
            .configure(**chart_theme)
        )
        st.altair_chart(delta_chart, width="stretch")
    with movers_right:
        if graph_bundle["status"] == "ready":
            top_graph_chart = (
                alt.Chart(top_regions.dropna(subset=["Graph GNN Probability"]).head(8))
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#2b637f")
                .encode(
                    y=alt.Y("District:N", sort="-x"),
                    x=alt.X("Graph GNN Probability:Q", title="Graph hotspot score"),
                    tooltip=[
                        "District",
                        "State",
                        alt.Tooltip("Graph GNN Probability:Q", format=".2f"),
                        alt.Tooltip("Risk Score:Q", format=".2f"),
                        "Primary Driver",
                    ],
                )
                .properties(height=300, title="Top Graph Hotspots")
                .configure(**chart_theme)
            )
            st.altair_chart(top_graph_chart, width="stretch")
        else:
            top_region_chart = (
                alt.Chart(top_regions)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#28566d")
                .encode(
                    y=alt.Y("District:N", sort="-x"),
                    x=alt.X("Risk Score:Q"),
                    tooltip=["District", "State", "Risk Score", "Forecast Risk", "Primary Driver"],
                )
                .properties(height=300, title="Highest-Risk Districts")
                .configure(**chart_theme)
            )
            st.altair_chart(top_region_chart, width="stretch")

with comparison_tab:
    st.markdown('<div class="section-label">Traditional vs GNN vs Spatio-Temporal GNN</div>', unsafe_allow_html=True)
    if benchmark_bundle:
        inventory = benchmark_bundle.get("inventory", {})
        traditional = benchmark_bundle.get("traditional_best", {})
        traditional_calibrated = benchmark_bundle.get("traditional_calibrated", {})
        graph_metrics = benchmark_bundle.get("graph_gnn", {})
        spatiotemporal_summary = benchmark_bundle.get("spatiotemporal_gnn", spatiotemporal_metrics)
        fair_scope = benchmark_bundle.get("fair_same_scope_benchmark", {})
        fair_traditional = fair_scope.get("traditional_same_scope", {})
        fair_graph = fair_scope.get("graph_gnn_same_scope", {})
        fair_spatiotemporal = fair_scope.get("spatiotemporal_gnn_same_scope", {})

        cmp_col_1, cmp_col_2, cmp_col_3, cmp_col_4 = st.columns(4)
        with cmp_col_1:
            render_metric_card(
                "Merged Positive Windows",
                str(inventory.get("merged_positive_windows", "n/a")),
                f"up from {inventory.get('original_positive_windows', 'n/a')}",
                tone="alert",
            )
        with cmp_col_2:
            render_metric_card(
                "Fair ST-GNN ROC-AUC",
                f"{_safe_round(fair_spatiotemporal.get('roc_auc'), 3)}",
                f"PR-AUC {_safe_round(fair_spatiotemporal.get('pr_auc'), 3)}",
                tone="forecast",
            )
        with cmp_col_3:
            render_metric_card(
                "Fair ST-GNN F1",
                f"{_safe_round(fair_spatiotemporal.get('f1'), 3)}",
                f"Precision {_safe_round(fair_spatiotemporal.get('precision'), 3)}",
                tone="graph",
            )
        with cmp_col_4:
            render_metric_card(
                "Fair Eval Rows",
                str(fair_scope.get("evaluation_rows", "n/a")),
                f"{fair_scope.get('node_count', 'n/a')} districts",
                tone="graph",
            )

        st.markdown("#### Fair Same-Scope Benchmark")
        fair_rows = pd.DataFrame(
            [
                {
                    "Approach": "Traditional runtime model",
                    "Threshold": _safe_round(fair_traditional.get("threshold"), 3),
                    "Accuracy": _safe_round(fair_traditional.get("accuracy"), 4),
                    "Precision": _safe_round(fair_traditional.get("precision"), 4),
                    "Recall": _safe_round(fair_traditional.get("recall"), 4),
                    "F1": _safe_round(fair_traditional.get("f1"), 4),
                    "ROC-AUC": _safe_round(fair_traditional.get("roc_auc"), 4),
                    "PR-AUC": _safe_round(fair_traditional.get("pr_auc"), 4),
                },
                {
                    "Approach": "Static graph GNN",
                    "Threshold": _safe_round(fair_graph.get("threshold"), 3),
                    "Accuracy": _safe_round(fair_graph.get("accuracy"), 4),
                    "Precision": _safe_round(fair_graph.get("precision"), 4),
                    "Recall": _safe_round(fair_graph.get("recall"), 4),
                    "F1": _safe_round(fair_graph.get("f1"), 4),
                    "ROC-AUC": _safe_round(fair_graph.get("roc_auc"), 4),
                    "PR-AUC": _safe_round(fair_graph.get("pr_auc"), 4),
                },
                {
                    "Approach": "Spatio-temporal GNN",
                    "Threshold": _safe_round(fair_spatiotemporal.get("threshold"), 3),
                    "Accuracy": _safe_round(fair_spatiotemporal.get("accuracy"), 4),
                    "Precision": _safe_round(fair_spatiotemporal.get("precision"), 4),
                    "Recall": _safe_round(fair_spatiotemporal.get("recall"), 4),
                    "F1": _safe_round(fair_spatiotemporal.get("f1"), 4),
                    "ROC-AUC": _safe_round(fair_spatiotemporal.get("roc_auc"), 4),
                    "PR-AUC": _safe_round(fair_spatiotemporal.get("pr_auc"), 4),
                },
            ]
        )
        st.dataframe(fair_rows, width="stretch", height=170)

        st.markdown("#### Mixed-Scope Context")
        comparison_rows = pd.DataFrame(
            [
                {
                    "Approach": "Traditional best model",
                    "Scope": traditional.get("scope", "Full validation"),
                    "Accuracy": _safe_round(traditional.get("accuracy"), 4),
                    "Precision": _safe_round(traditional.get("precision"), 4),
                    "Recall": _safe_round(traditional.get("recall"), 4),
                    "F1": _safe_round(traditional.get("f1"), 4),
                    "ROC-AUC": _safe_round(traditional.get("roc_auc"), 4),
                    "PR-AUC": _safe_round(traditional.get("pr_auc"), 4),
                },
                {
                    "Approach": "Traditional calibrated",
                    "Scope": "Full merged-label temporal validation",
                    "Accuracy": _safe_round(traditional_calibrated.get("accuracy"), 4),
                    "Precision": _safe_round(traditional_calibrated.get("precision"), 4),
                    "Recall": _safe_round(traditional_calibrated.get("recall"), 4),
                    "F1": _safe_round(traditional_calibrated.get("f1"), 4),
                    "ROC-AUC": _safe_round(traditional_calibrated.get("roc_auc"), 4),
                    "PR-AUC": _safe_round(traditional_calibrated.get("pr_auc"), 4),
                },
                {
                    "Approach": "Graph GNN",
                    "Scope": graph_metrics.get("scope", "Reduced graph validation"),
                    "Accuracy": _safe_round(graph_metrics.get("accuracy"), 4),
                    "Precision": _safe_round(graph_metrics.get("precision"), 4),
                    "Recall": _safe_round(graph_metrics.get("recall"), 4),
                    "F1": _safe_round(graph_metrics.get("f1"), 4),
                    "ROC-AUC": _safe_round(graph_metrics.get("roc_auc"), 4),
                    "PR-AUC": _safe_round(graph_metrics.get("pr_auc"), 4),
                },
                {
                    "Approach": "Spatio-temporal GNN",
                    "Scope": f"District-date graph sequence benchmark ({spatiotemporal_summary.get('node_count', 'n/a')} districts)",
                    "Accuracy": _safe_round(spatiotemporal_summary.get("accuracy"), 4),
                    "Precision": _safe_round(spatiotemporal_summary.get("precision"), 4),
                    "Recall": _safe_round(spatiotemporal_summary.get("recall"), 4),
                    "F1": _safe_round(spatiotemporal_summary.get("validation_f1", spatiotemporal_summary.get("f1")), 4),
                    "ROC-AUC": _safe_round(spatiotemporal_summary.get("validation_roc_auc", spatiotemporal_summary.get("roc_auc")), 4),
                    "PR-AUC": _safe_round(spatiotemporal_summary.get("validation_pr_auc", spatiotemporal_summary.get("pr_auc")), 4),
                },
            ]
        )
        st.dataframe(comparison_rows, width="stretch", height=190)

        chart_frame = pd.DataFrame(
            [
                {
                    "Metric": "Precision",
                    "Traditional": traditional.get("precision", 0.0),
                    "GNN": graph_metrics.get("precision", 0.0),
                    "ST-GNN": spatiotemporal_summary.get("precision", 0.0),
                },
                {
                    "Metric": "Recall",
                    "Traditional": traditional.get("recall", 0.0),
                    "GNN": graph_metrics.get("recall", 0.0),
                    "ST-GNN": spatiotemporal_summary.get("recall", 0.0),
                },
                {
                    "Metric": "F1",
                    "Traditional": traditional.get("f1", 0.0),
                    "GNN": graph_metrics.get("f1", 0.0),
                    "ST-GNN": spatiotemporal_summary.get("validation_f1", spatiotemporal_summary.get("f1", 0.0)),
                },
                {
                    "Metric": "PR-AUC",
                    "Traditional": traditional.get("pr_auc", 0.0),
                    "GNN": graph_metrics.get("pr_auc", 0.0),
                    "ST-GNN": spatiotemporal_summary.get("validation_pr_auc", spatiotemporal_summary.get("pr_auc", 0.0)),
                },
            ]
        ).melt(id_vars="Metric", var_name="Model", value_name="Value")
        comparison_chart = (
            alt.Chart(chart_frame)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Metric:N", title="Metric"),
                y=alt.Y("Value:Q", title="Score"),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#1a5c7a", "#d96d2d", "#548c5c"])),
                xOffset="Model:N",
                tooltip=["Metric", "Model", alt.Tooltip("Value:Q", format=".3f")],
            )
            .properties(height=320, title="Merged Real-Data Traditional vs GNN vs Spatio-Temporal GNN")
            .configure(**chart_theme)
        )
        st.altair_chart(comparison_chart, width="stretch")

        st.markdown(
            """
            <div class="panel-card">
                <strong>Interpretation</strong><br/>
                The fair same-scope benchmark is now the main comparison to trust. On that shared district-date graph-sequence evaluation, the upgraded spatio-temporal GNN became the strongest overall model and beat both the traditional runtime model and the static graph GNN on ROC-AUC, PR-AUC, and F1 while also avoiding the threshold instability seen in the other two approaches.
                The mixed-scope rows below are still useful context for the full merged temporal benchmark, but they should be interpreted as supporting evidence rather than the fairest head-to-head comparison.
            </div>
            """,
            unsafe_allow_html=True,
        )

        reliability_payload = advanced_research_bundle.get("reliability", {})
        if reliability_payload:
            reliability_rows = []
            for series in reliability_payload.get("series", []):
                for predicted, observed in zip(series.get("bin_pred_rate", []), series.get("bin_true_rate", []), strict=True):
                    reliability_rows.append(
                        {
                            "Model": series.get("model_name", "unknown"),
                            "Predicted Risk": predicted,
                            "Observed Rate": observed,
                            "Brier Score": series.get("brier_score", 0.0),
                        }
                    )
            if reliability_rows:
                st.markdown("#### Reliability Curves")
                reliability_frame = pd.DataFrame(reliability_rows)
                reliability_chart = (
                    alt.Chart(reliability_frame)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Predicted Risk:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("Observed Rate:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Model:N", scale=alt.Scale(range=["#d96d2d", "#1a5c7a"])),
                        tooltip=["Model", alt.Tooltip("Predicted Risk:Q", format=".3f"), alt.Tooltip("Observed Rate:Q", format=".3f"), alt.Tooltip("Brier Score:Q", format=".4f")],
                    )
                    .properties(height=260, title="Probability Reliability")
                    .configure(**chart_theme)
                )
                st.altair_chart(reliability_chart, width="stretch")

        calibration_rows = advanced_research_bundle.get("st_calibration", [])
        if calibration_rows:
            st.markdown("#### ST-GNN Calibration")
            calibration_frame = pd.DataFrame(calibration_rows).rename(
                columns={
                    "method": "Calibration",
                    "threshold": "Threshold",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC",
                    "brier_score": "Brier",
                }
            )
            preferred_columns = [column for column in ["Calibration", "Threshold", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"] if column in calibration_frame.columns]
            st.dataframe(calibration_frame[preferred_columns], width="stretch", height=185)
            best_row = min(calibration_rows, key=lambda row: float(row.get("brier_score", 1.0)))
            st.caption(
                "Sigmoid calibration currently gives the strongest probability reliability for the ST-GNN, "
                f"reducing Brier score to {best_row.get('brier_score', 0.0):.4f}."
            )

        ablation_rows = (
            advanced_research_bundle.get("spatiotemporal_ablation")
            or advanced_research_bundle.get("spatiotemporal_ablation_fast")
            or advanced_research_bundle.get("ablation")
        )
        if ablation_rows:
            st.markdown("#### Ablation Study")
            ablation_frame = pd.DataFrame(ablation_rows)
            rename_map = {
                "name": "Ablation",
                "ablation_name": "Ablation",
                "validation_precision": "Precision",
                "validation_recall": "Recall",
                "validation_f1": "F1",
                "validation_roc_auc": "ROC-AUC",
                "validation_pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
            }
            ablation_frame = ablation_frame.rename(columns=rename_map)
            preferred_columns = [column for column in ["Ablation", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"] if column in ablation_frame.columns]
            st.dataframe(ablation_frame[preferred_columns], width="stretch", height=220)

        external_baselines = advanced_research_bundle.get("external_baselines", [])
        if external_baselines:
            st.markdown("#### External Baselines")
            baseline_frame = pd.DataFrame(external_baselines).rename(
                columns={
                    "model_name": "Model",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC",
                    "brier_score": "Brier",
                }
            )
            preferred_columns = [column for column in ["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"] if column in baseline_frame.columns]
            st.dataframe(baseline_frame[preferred_columns], width="stretch", height=220)
    else:
        st.info("Merged benchmark summary artifact not found yet. Run the merged real-data evaluation to populate this section.")

with alerts_tab:
    st.markdown('<div class="section-label">Alerts</div>', unsafe_allow_html=True)
    if not alert_frame.empty:
        high_alerts = int((alert_frame["severity"] == "High").sum())
        moderate_alerts = int((alert_frame["severity"] == "Moderate").sum())
        critical_bands = int((alert_frame["alert_band"] == "Critical").sum()) if "alert_band" in alert_frame.columns else high_alerts
        alert_kpi_1, alert_kpi_2, alert_kpi_3 = st.columns(3)
        with alert_kpi_1:
            render_metric_card("High Alerts", str(high_alerts), "Immediate attention needed", tone="alert")
        with alert_kpi_2:
            render_metric_card("Critical Bands", str(critical_bands), "Confidence-aware operational escalations", tone="forecast")
        with alert_kpi_3:
            render_metric_card("Alert Coverage", str(len(alert_frame)), "Districts with alert messages", tone="graph")
        st.markdown(
            """
            <div class="panel-card">
                <strong>Calibrated alerting</strong><br/>
                Alerts now combine current risk, forecast risk, and model uncertainty into a calibrated operational score.
                Use <em>Critical</em> alerts for escalation, <em>Watch</em> for intensified monitoring, and <em>Review</em> when the signal is meaningful but confidence is still weak.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            alert_frame.sort_values(["severity", "calibrated_risk_score"], ascending=[True, False]),
            width="stretch",
            height=420,
        )
    else:
        st.success("No regions crossed the configured alert threshold under the current scenario.")

with forecast_tab:
    st.markdown('<div class="section-label">Forecast</div>', unsafe_allow_html=True)
    forecast_bundle = get_forecast_bundle(windows_csv_path)

    if forecast_bundle["status"] == "artifact":
        forecast_frame = forecast_bundle["frame"]
        forecast_metadata = forecast_bundle.get("metadata", {})
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Forecast Samples", str(len(forecast_frame)))
        metric_b.metric(
            "Observed MAE",
            f"{forecast_frame['absolute_error'].mean():.2f}" if "absolute_error" in forecast_frame.columns else "n/a",
        )
        metric_c.metric(
            "Validation MAE",
            f"{float(forecast_metadata.get('validation_mae', 0.0)):.2f}" if forecast_metadata else "n/a",
        )
        metric_d.metric("Inference Mode", "Saved Artifact")
        forecast_chart = (
            alt.Chart(forecast_frame.head(40))
            .mark_line(point=True)
            .encode(
                x=alt.X("reference_date:N", title="Reference date"),
                y=alt.Y("predicted_rainfall:Q", title="Predicted rainfall"),
                color=alt.Color("region_id:N", legend=None),
                tooltip=["region_id", "reference_date", "predicted_rainfall"],
            )
            .properties(height=320, title="Predicted Rainfall from Saved Forecast Model")
        )
        st.altair_chart(forecast_chart, width="stretch")
        st.caption(f"Loaded model artifact: `{forecast_bundle['source']}`")
        if forecast_metadata:
            st.caption(
                "Model quality: "
                f"{forecast_metadata.get('validation_split_type', 'n/a')} | "
                f"validation RMSE {float(forecast_metadata.get('validation_rmse', 0.0)):.2f} | "
                f"active-window MAE {float(forecast_metadata.get('validation_active_mae', 0.0)):.2f}"
            )
        st.dataframe(forecast_frame, width="stretch", height=320)
    elif forecast_bundle["status"] == "empty":
        st.warning("Forecast windows file exists but is empty. Re-run the ingestion pipeline to regenerate temporal features.")
    elif forecast_bundle["status"] == "missing":
        st.info("Forecast windows not found yet. Run `PYTHONPATH=src python3 scripts/run_ingestion_jobs.py` first.")
    elif forecast_bundle["status"] == "artifact_error":
        st.warning(
            "Saved forecast artifact could not be loaded in this environment. "
            "The dashboard is staying up, but forecast inference is unavailable until the artifact is retrained."
        )
        st.caption(f"Forecast artifact: `{forecast_bundle['source']}`")
        st.caption(f"Load error: {forecast_bundle['message']}")
        st.dataframe(forecast_bundle["frame"].head(40), width="stretch", height=320)
    else:
        st.warning("No saved forecast artifact found yet. Train the baseline forecaster once from the CLI for fast inference here.")

    st.markdown("#### Optional Benchmarks")
    benchmark_col_1, benchmark_col_2, benchmark_col_3 = st.columns([1, 1, 1.2])
    with benchmark_col_1:
        run_baseline_benchmark = st.button("Run baseline benchmark", width="stretch")
    with benchmark_col_2:
        run_lstm_benchmark = st.button(
            "Run one-off LSTM benchmark",
            disabled=not torch_available(),
            width="stretch",
        )
    with benchmark_col_3:
        run_graph_benchmark = st.button(
            "Run graph benchmark",
            disabled=not torch_available(),
            width="stretch",
        )

    if run_baseline_benchmark:
        try:
            benchmark = train_rainfall_forecaster_from_windows(windows_csv_path)
            st.success(
                "Baseline benchmark complete. "
                f"Validation MAE {benchmark.validation_mae:.2f}, "
                f"validation RMSE {benchmark.validation_rmse:.2f}, "
                f"validation R2 {benchmark.validation_r2:.2f}"
            )
            st.caption(
                f"Train MAE {benchmark.train_mae:.2f} | Train RMSE {benchmark.train_rmse:.2f} | Train R2 {benchmark.train_r2:.2f}"
            )
            st.download_button(
                "Download baseline benchmark summary",
                data=forecast_summary_to_json_bytes(benchmark),
                file_name="baseline_forecast_benchmark.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(str(exc))

    if run_lstm_benchmark:
        try:
            lstm_result = train_lstm_forecaster_from_windows(windows_csv_path, epochs=40)
            st.success(
                f"LSTM benchmark complete. MAE {lstm_result.mae:.2f}, RMSE {lstm_result.rmse:.2f}, R2 {lstm_result.r2:.2f}"
            )
            st.download_button(
                "Download LSTM benchmark summary",
                data=lstm_forecast_summary_to_json_bytes(lstm_result),
                file_name="lstm_forecast_benchmark.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(str(exc))

    if run_graph_benchmark:
        try:
            graph_result = train_graph_risk_model("data/india/india_graph_training.csv", epochs=220, learning_rate=0.015)
            st.success(
                "Graph benchmark complete. "
                f"Train accuracy {graph_result.accuracy:.2f}, validation accuracy {graph_result.validation_accuracy:.2f}, "
                f"validation F1 {graph_result.validation_f1:.2f}"
            )
        except Exception as exc:
            st.error(str(exc))

with details_tab:
    st.markdown('<div class="section-label">District Detail</div>', unsafe_allow_html=True)
    detail_left, detail_right = st.columns([1.3, 1])
    with detail_left:
        st.dataframe(table, width="stretch", height=440)
    with detail_right:
        selected_region = st.selectbox("Choose a district to inspect", table["Region ID"].tolist())
        selected_row = table.loc[table["Region ID"] == selected_region].iloc[0]
        selected_st_row = None
        spatiotemporal_bundle = None
        if spatiotemporal_ready:
            spatiotemporal_bundle = get_spatiotemporal_bundle(regional_spatiotemporal_csv_path)
            if spatiotemporal_bundle.get("status") == "ready":
                district_matches = spatiotemporal_bundle["frame"].loc[
                    spatiotemporal_bundle["frame"]["district"] == selected_row["District"]
                ].sort_values("date", ascending=False)
                if not district_matches.empty:
                    selected_st_row = district_matches.iloc[0]
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f"### {selected_row['District']}, {selected_row['State']}")
        st.write(selected_row["Advisory"])
        calibrated_cols = st.columns(3)
        with calibrated_cols[0]:
            render_metric_card(
                "Calibrated Risk",
                f"{float(selected_row['Calibrated Risk']):.2f}",
                f"Band {selected_row['Alert Band']}",
                tone="forecast",
            )
        with calibrated_cols[1]:
            render_metric_card(
                "Confidence",
                str(selected_row["Confidence Level"]),
                f"Uncertainty {float(selected_row['Uncertainty']):.2f}",
                tone="graph",
            )
        with calibrated_cols[2]:
            render_metric_card(
                "Recommended Action",
                str(selected_row["Alert Band"]),
                str(selected_row["Recommended Action"]),
                tone="alert",
            )
        signal_frame = pd.DataFrame(
            [
                {"Signal": "Rainfall", "Value": selected_row["Rainfall Exposure"]},
                {"Signal": "Terrain", "Value": selected_row["Terrain Exposure"]},
                {"Signal": "Vegetation", "Value": selected_row["Vegetation Exposure"]},
                {"Signal": "Graph", "Value": selected_row["Graph Exposure"]},
                {"Signal": "Propagated Graph", "Value": selected_row["Propagated Graph Exposure"]},
            ]
        )
        signal_chart = (
            alt.Chart(signal_frame)
            .mark_bar(color="#d46f2e", cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x="Signal:N",
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Signal", "Value"],
            )
            .properties(height=250, title="Risk Signal Breakdown")
        )
        st.altair_chart(signal_chart, width="stretch")
        st.json(
            {
                "primary_driver": selected_row["Primary Driver"],
                "uncertainty": round(float(selected_row["Uncertainty"]), 3),
                "calibrated_risk_score": round(float(selected_row["Calibrated Risk"]), 3),
                "confidence_level": selected_row["Confidence Level"],
                "alert_band": selected_row["Alert Band"],
                "recommended_action": selected_row["Recommended Action"],
                "graph_exposure": round(float(selected_row["Graph Exposure"]), 3),
                "propagated_graph_exposure": round(float(selected_row["Propagated Graph Exposure"]), 3),
                "graph_gnn_probability": None
                if pd.isna(selected_row.get("Graph GNN Probability"))
                else round(float(selected_row["Graph GNN Probability"]), 3),
                "risk_delta_vs_base": round(
                    float(
                        scenario_delta.loc[scenario_delta["Region ID"] == selected_region, "Risk Delta"].iloc[0]
                    ),
                    3,
                )
                if not scenario_delta[scenario_delta["Region ID"] == selected_region].empty
                else 0.0,
                "spatiotemporal_artifact_ready": spatiotemporal_ready,
            }
        )
        st.markdown("#### Model Comparison")
        model_rows = [
            {
                "Model": "Traditional runtime",
                "Score": round(float(selected_row["Risk Score"]), 3),
                "Driver": selected_row["Primary Driver"],
                "Confidence": selected_row["Confidence Level"],
            },
            {
                "Model": "Static graph GNN",
                "Score": None if pd.isna(selected_row.get("Graph GNN Probability")) else round(float(selected_row["Graph GNN Probability"]), 3),
                "Driver": "Spatial spillover",
                "Confidence": "n/a",
            },
        ]
        if selected_st_row is not None:
            model_rows.append(
                {
                    "Model": "Spatio-temporal GNN",
                    "Score": round(float(selected_st_row["spatiotemporal_gnn_probability"]), 3),
                    "Driver": selected_st_row["top_driver"],
                    "Confidence": selected_st_row["confidence_level"],
                }
            )
        model_frame = pd.DataFrame(model_rows)
        st.dataframe(model_frame, width="stretch", height=140)
        disagreement_note = _model_disagreement_summary(
            traditional_score=float(selected_row["Risk Score"]),
            graph_score=None if pd.isna(selected_row.get("Graph GNN Probability")) else float(selected_row["Graph GNN Probability"]),
            spatiotemporal_score=None if selected_st_row is None else float(selected_st_row["spatiotemporal_gnn_probability"]),
        )
        st.markdown(
            f"""
            <div class="panel-card">
                <strong>Why the models disagree</strong><br/>
                {disagreement_note}
            </div>
            """,
            unsafe_allow_html=True,
        )
        explainability_payload = advanced_research_bundle.get("explainability", {})
        if explainability_payload:
            st.markdown("#### Global Drivers")
            global_importance = explainability_payload.get("global_top_features", [])
            if global_importance:
                st.dataframe(pd.DataFrame(global_importance), width="stretch", height=180)
            district_payload = explainability_payload.get("district_level_tabular", {})
            if district_payload and district_payload.get("row_region_id") == selected_region:
                st.markdown("#### District-Level Tabular Explanation")
                st.json(district_payload)
        if selected_st_row is not None:
            st.markdown("#### Spatio-Temporal Explanation")
            st.json(
                {
                    "date": selected_st_row["date"],
                    "score": round(float(selected_st_row["spatiotemporal_gnn_probability"]), 3),
                    "uncertainty_score": round(float(selected_st_row["uncertainty_score"]), 3),
                    "confidence_level": selected_st_row["confidence_level"],
                    "recommended_interpretation": selected_st_row["recommended_interpretation"],
                    "top_feature_drivers": selected_st_row["top_feature_drivers"],
                    "top_neighbor_influences": selected_st_row["top_neighbor_influences"],
                }
            )
        case_studies = advanced_research_bundle.get("case_studies", [])
        if case_studies:
            st.markdown("#### Case Studies")
            case_frame = pd.DataFrame(
                [
                    {
                        "District": case["district"],
                        "State": case["state"],
                        "Reference Date": case["reference_date"],
                        "Score": round(float(case["st_gnn_score"]), 3),
                        "Confidence": case["confidence_level"],
                        "Top Driver": case["top_driver"],
                    }
                    for case in case_studies
                ]
            )
            st.dataframe(case_frame, width="stretch", height=160)
        error_payload = advanced_research_bundle.get("error_analysis", {})
        if error_payload:
            st.markdown("#### Error Analysis")
            error_cols = st.columns(2)
            with error_cols[0]:
                st.metric("False Positives", int(error_payload.get("false_positive_count", 0)))
            with error_cols[1]:
                st.metric("False Negatives", int(error_payload.get("false_negative_count", 0)))
            state_summary_rows = []
            for state_name, count in error_payload.get("false_positive_states", {}).items():
                state_summary_rows.append({"State": state_name, "Error Type": "False Positive", "Count": count})
            for state_name, count in error_payload.get("false_negative_states", {}).items():
                state_summary_rows.append({"State": state_name, "Error Type": "False Negative", "Count": count})
            if state_summary_rows:
                st.dataframe(pd.DataFrame(state_summary_rows), width="stretch", height=180)
        elif spatiotemporal_ready:
            st.info("No spatio-temporal sequence row was available for this district in the merged benchmark dataset.")
        elif spatiotemporal_bundle and spatiotemporal_bundle.get("status") != "ready":
            st.warning("Spatio-temporal inference is enabled, but its explanation frame could not be loaded right now.")
        st.markdown("</div>", unsafe_allow_html=True)

with knowledge_tab:
    st.markdown('<div class="section-label">Knowledge Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-card">
            Ask about landslide drivers, district risk interpretation, safety guidance, or project methodology.
            The assistant retrieves local knowledge first and then uses the LLM layer when it is enabled.
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_question = (
        f"What should authorities focus on for {top_row['District']}, {top_row['State']} given the current risk pattern?"
    )
    question = st.text_area(
        "Ask a question about the current risk view or project knowledge",
        value=st.session_state.get("knowledge_question", default_question),
        height=110,
        placeholder="Example: Why is this district high risk and what actions should local teams take?",
    )
    st.session_state["knowledge_question"] = question

    qa_col_1, qa_col_2, qa_col_3 = st.columns([0.9, 0.8, 1.2])
    with qa_col_1:
        top_k = st.slider("Retrieved context chunks", min_value=1, max_value=5, value=3)
    with qa_col_2:
        run_qa = st.button("Ask knowledge assistant", width="stretch")
    with qa_col_3:
        if st.button("Use district detail prompt", width="stretch"):
            st.session_state["knowledge_question"] = (
                f"Explain the risk for {selected_row['District']}, {selected_row['State']} and suggest practical precautions."
            )
            st.rerun()

    if run_qa:
        cleaned_question = question.strip()
        if not cleaned_question:
            st.warning("Enter a question first so the assistant has something to answer.")
        else:
            with st.spinner("Retrieving guidance and preparing an answer..."):
                knowledge_response = run_knowledge_query(cleaned_question, top_k)

            status_col_1, status_col_2, status_col_3 = st.columns(3)
            with status_col_1:
                render_metric_card(
                    "Answer Mode",
                    "LLM + Retrieval" if knowledge_response.get("llm_enabled") else "Retrieval",
                    "Retrieval runs first for grounded answers",
                    tone="graph",
                )
            with status_col_2:
                render_metric_card(
                    "Context Chunks",
                    str(knowledge_response.get("row_count", 0)),
                    "Knowledge snippets used for the answer",
                    tone="forecast",
                )
            with status_col_3:
                render_metric_card(
                    "Current Focus",
                    str(top_row["Primary Driver"]),
                    f"Top district: {top_row['District']}",
                    tone="alert",
                )

            st.markdown("#### Answer")
            st.markdown(str(knowledge_response.get("answer", "No answer generated.")))

            if not knowledge_response.get("llm_enabled"):
                st.info(
                    "LLM synthesis is currently off, so this answer is based on retrieval-first local summarization. "
                    "Enable the LLM environment variables to get full synthesized responses."
                )

            rows = knowledge_response.get("rows", [])
            if rows:
                st.markdown("#### Retrieved Context")
                context_frame = pd.DataFrame(rows)
                preferred_columns = [
                    column
                    for column in ["rank", "source", "score", "content"]
                    if column in context_frame.columns
                ]
                st.dataframe(context_frame[preferred_columns], width="stretch", height=260)
            else:
                st.warning("No matching knowledge snippets were retrieved for this question.")

with exports_tab:
    st.markdown('<div class="section-label">Exports</div>', unsafe_allow_html=True)
    export_assessments = scenario_results = table.to_dict(orient="records")
    export_alerts = alert_frame.to_dict(orient="records")
    export_col_1, export_col_2, export_col_3 = st.columns(3)
    export_col_1.download_button(
        "Download assessments CSV",
        data=assessments_to_csv_bytes(table),
        file_name="india_landslide_assessments.csv",
        mime="text/csv",
    )
    export_col_2.download_button(
        "Download alerts CSV",
        data=pd.DataFrame(export_alerts).to_csv(index=False).encode("utf-8"),
        file_name="india_landslide_alerts.csv",
        mime="text/csv",
    )
    export_col_3.download_button(
        "Download summary JSON",
        data=__import__("json").dumps(
            {
                "assessment_count": len(export_assessments),
                "alert_count": len(export_alerts),
                "top_region": export_assessments[0]["Region ID"] if export_assessments else None,
                "assessments": export_assessments,
                "alerts": export_alerts,
            },
            indent=2,
        ).encode("utf-8"),
        file_name="india_landslide_summary.json",
        mime="application/json",
    )

    st.markdown("#### Scenario Delta Export")
    st.download_button(
        "Download scenario delta CSV",
        data=scenario_delta.to_csv(index=False).encode("utf-8"),
        file_name="india_landslide_scenario_delta.csv",
        mime="text/csv",
    )
