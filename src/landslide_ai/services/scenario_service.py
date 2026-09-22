from __future__ import annotations

from dataclasses import replace

from landslide_ai.data.schemas import RegionRecord


def apply_scenario(
    records: list[RegionRecord],
    rainfall_multiplier: float = 1.0,
    antecedent_rainfall_multiplier: float = 1.0,
    soil_wetness_delta: float = 0.0,
    ndvi_delta: float = 0.0,
    neighbor_risk_delta: float = 0.0,
    target_state: str | None = None,
) -> list[RegionRecord]:
    updated: list[RegionRecord] = []
    for record in records:
        if target_state and record.state != target_state:
            updated.append(record)
            continue

        updated.append(
            replace(
                record,
                rainfall_24h_mm=max(record.rainfall_24h_mm * rainfall_multiplier, 0.0),
                rainfall_7d_mm=max(record.rainfall_7d_mm * antecedent_rainfall_multiplier, 0.0),
                soil_wetness_index=min(max(record.soil_wetness_index + soil_wetness_delta, 0.0), 1.0),
                ndvi=min(max(record.ndvi + ndvi_delta, 0.0), 1.0),
                neighbor_risk_mean=min(max(record.neighbor_risk_mean + neighbor_risk_delta, 0.0), 1.0),
            )
        )
    return updated
