from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from landslide_ai.config import load_config
from landslide_ai.data.schemas import RegionRecord


@dataclass(slots=True)
class SpatialGraph:
    region_ids: list[str]
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray


def build_spatial_graph(records: list[RegionRecord], distance_scale_km: float | None = None) -> SpatialGraph:
    if distance_scale_km is None:
        distance_scale_km = load_config(".").graph_distance_scale_km
    region_ids = [record.region_id for record in records]
    count = len(records)
    adjacency = np.zeros((count, count), dtype=float)

    if count == 0:
        return SpatialGraph(region_ids=region_ids, adjacency=adjacency, normalized_adjacency=adjacency)

    for row_index, source in enumerate(records):
        for col_index, target in enumerate(records):
            if row_index == col_index:
                adjacency[row_index, col_index] = 1.0
                continue
            distance = _distance_km(source, target)
            distance_weight = np.exp(-distance / distance_scale_km)
            rainfall_similarity = 1.0 - min(
                abs(getattr(source, "rainfall_7d_mm", 0.0) - getattr(target, "rainfall_7d_mm", 0.0)) / 600.0,
                1.0,
            )
            slope_similarity = 1.0 - min(
                abs(getattr(source, "slope_deg", 0.0) - getattr(target, "slope_deg", 0.0)) / 50.0,
                1.0,
            )
            temperature_similarity = 1.0 - min(
                abs(getattr(source, "temperature_c", 0.0) - getattr(target, "temperature_c", 0.0)) / 20.0,
                1.0,
            )
            ndvi_similarity = 1.0 - min(
                abs(getattr(source, "ndvi", 0.5) - getattr(target, "ndvi", 0.5)),
                1.0,
            )
            state_bonus = 1.05 if getattr(source, "state", "") == getattr(target, "state", "") else 1.0
            terrain_zone_bonus = 1.05 if _same_terrain_zone(source, target) else 1.0
            weight = (
                distance_weight * 0.55
                + rainfall_similarity * 0.20
                + slope_similarity * 0.10
                + temperature_similarity * 0.075
                + ndvi_similarity * 0.075
            ) * state_bonus * terrain_zone_bonus
            adjacency[row_index, col_index] = float(weight)

    degree = adjacency.sum(axis=1)
    degree[degree == 0.0] = 1.0
    normalized_adjacency = adjacency / degree[:, None]
    return SpatialGraph(
        region_ids=region_ids,
        adjacency=adjacency,
        normalized_adjacency=normalized_adjacency,
    )


def propagate_signal(
    graph: SpatialGraph,
    base_signal: np.ndarray,
    steps: int = 2,
    alpha: float = 0.6,
) -> np.ndarray:
    signal = base_signal.astype(float)
    original = signal.copy()
    for _ in range(max(steps, 1)):
        signal = (alpha * graph.normalized_adjacency @ signal) + ((1.0 - alpha) * original)
    return np.clip(signal, 0.0, 1.0)


def _distance_km(source: RegionRecord, target: RegionRecord) -> float:
    from math import atan2, cos, radians, sin, sqrt

    earth_radius_km = 6371.0
    lat1 = radians(source.latitude)
    lon1 = radians(source.longitude)
    lat2 = radians(target.latitude)
    lon2 = radians(target.longitude)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a_value = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    c_value = 2 * atan2(sqrt(a_value), sqrt(1 - a_value))
    return earth_radius_km * c_value


def _same_terrain_zone(source: RegionRecord, target: RegionRecord) -> bool:
    source_elevation = getattr(source, "elevation_m", 0.0)
    target_elevation = getattr(target, "elevation_m", 0.0)
    source_slope = getattr(source, "slope_deg", 0.0)
    target_slope = getattr(target, "slope_deg", 0.0)
    return abs(source_elevation - target_elevation) <= 350.0 and abs(source_slope - target_slope) <= 8.0
