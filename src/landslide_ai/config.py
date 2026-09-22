from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(project_root: str | Path = ".") -> None:
    env_path = Path(project_root) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    parts = [item.strip() for item in value.split(",")]
    return tuple(item for item in parts if item)


@dataclass(frozen=True, slots=True)
class AppConfig:
    project_root: Path
    api_auth_enabled: bool
    api_key: str
    api_host: str
    api_port: int
    api_cors_origins: tuple[str, ...]
    enable_llm: bool
    llm_provider: str
    llm_model: str
    openai_api_key: str
    enable_deep_risk_model: bool
    deep_risk_model_artifact: Path
    enable_trained_risk_model: bool
    risk_model_artifact: Path
    enable_graph_model: bool
    graph_model_artifact: Path
    enable_spatiotemporal_graph_model: bool
    spatiotemporal_graph_model_artifact: Path
    default_region_csv: Path
    default_rainfall_windows_csv: Path
    official_district_geojson: Path
    official_state_geojson: Path
    raw_imd_dir: Path
    raw_nasa_dir: Path
    raw_sentinel_dir: Path
    imd_source_url: str
    nasa_power_url: str
    sentinel_catalog_url: str
    nasa_power_start_date: str
    nasa_power_end_date: str
    graph_distance_scale_km: float
    graph_propagation_steps: int
    graph_propagation_alpha: float
    experiment_log_dir: Path


def load_config(project_root: str | Path = ".") -> AppConfig:
    root = Path(project_root).resolve()
    load_env_file(root)
    return AppConfig(
        project_root=root,
        api_auth_enabled=_env_bool("LANDSLIDE_API_AUTH_ENABLED", False),
        api_key=os.getenv("LANDSLIDE_API_KEY", "change-me"),
        api_host=os.getenv("LANDSLIDE_API_HOST", "0.0.0.0"),
        api_port=_env_int("LANDSLIDE_API_PORT", 8000),
        api_cors_origins=_env_list("LANDSLIDE_API_CORS_ORIGINS", ("*",)),
        enable_llm=_env_bool("LANDSLIDE_ENABLE_LLM", False),
        llm_provider=os.getenv("LANDSLIDE_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LANDSLIDE_LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("LANDSLIDE_OPENAI_API_KEY", ""),
        enable_deep_risk_model=_env_bool("LANDSLIDE_ENABLE_DEEP_RISK_MODEL", False),
        deep_risk_model_artifact=root / os.getenv("LANDSLIDE_DEEP_RISK_MODEL_ARTIFACT", "artifacts/deep_risk_model.pt"),
        enable_trained_risk_model=_env_bool("LANDSLIDE_ENABLE_TRAINED_RISK_MODEL", True),
        risk_model_artifact=root / os.getenv("LANDSLIDE_RISK_MODEL_ARTIFACT", "artifacts/risk_model.pkl"),
        enable_graph_model=_env_bool("LANDSLIDE_ENABLE_GRAPH_MODEL", False),
        graph_model_artifact=root / os.getenv("LANDSLIDE_GRAPH_MODEL_ARTIFACT", "artifacts/graph_gnn.pt"),
        enable_spatiotemporal_graph_model=_env_bool("LANDSLIDE_ENABLE_SPATIOTEMPORAL_GRAPH_MODEL", False),
        spatiotemporal_graph_model_artifact=root
        / os.getenv("LANDSLIDE_SPATIOTEMPORAL_GRAPH_MODEL_ARTIFACT", "artifacts/spatiotemporal_gnn.pt"),
        default_region_csv=root / os.getenv("LANDSLIDE_REGION_CSV", "data/india/india_regions.csv"),
        default_rainfall_windows_csv=root / os.getenv("LANDSLIDE_RAINFALL_WINDOWS_CSV", "data/india/rainfall_windows.csv"),
        official_district_geojson=root / os.getenv("LANDSLIDE_OFFICIAL_DISTRICT_GEOJSON", "data/gis/india_districts_official.geojson"),
        official_state_geojson=root / os.getenv("LANDSLIDE_OFFICIAL_STATE_GEOJSON", "data/gis/india_states_official.geojson"),
        raw_imd_dir=root / os.getenv("LANDSLIDE_RAW_IMD_DIR", "data/raw/imd"),
        raw_nasa_dir=root / os.getenv("LANDSLIDE_RAW_NASA_DIR", "data/raw/nasa_power"),
        raw_sentinel_dir=root / os.getenv("LANDSLIDE_RAW_SENTINEL_DIR", "data/raw/sentinel"),
        imd_source_url=os.getenv("LANDSLIDE_IMD_SOURCE_URL", ""),
        nasa_power_url=os.getenv("LANDSLIDE_NASA_POWER_URL", "https://power.larc.nasa.gov/api/temporal/daily/point"),
        sentinel_catalog_url=os.getenv("LANDSLIDE_SENTINEL_CATALOG_URL", ""),
        nasa_power_start_date=os.getenv("LANDSLIDE_NASA_POWER_START_DATE", "20260101"),
        nasa_power_end_date=os.getenv("LANDSLIDE_NASA_POWER_END_DATE", "20260131"),
        graph_distance_scale_km=_env_float("LANDSLIDE_GRAPH_DISTANCE_SCALE_KM", 250.0),
        graph_propagation_steps=_env_int("LANDSLIDE_GRAPH_PROPAGATION_STEPS", 2),
        graph_propagation_alpha=_env_float("LANDSLIDE_GRAPH_PROPAGATION_ALPHA", 0.65),
        experiment_log_dir=root / os.getenv("LANDSLIDE_EXPERIMENT_LOG_DIR", "artifacts/experiments"),
    )
