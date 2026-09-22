from dataclasses import dataclass


@dataclass(slots=True)
class RegionRecord:
    region_id: str
    state: str
    district: str
    latitude: float
    longitude: float
    rainfall_24h_mm: float
    rainfall_7d_mm: float
    slope_deg: float
    elevation_m: float
    soil_wetness_index: float
    ndvi: float
    temperature_c: float
    neighbor_risk_mean: float
    neighbor_count: int


@dataclass(slots=True)
class RegionFeatures:
    rainfall_intensity: float
    antecedent_rainfall: float
    slope_factor: float
    soil_wetness: float
    vegetation_stress: float
    graph_influence: float
