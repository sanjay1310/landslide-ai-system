from landslide_ai.data.schemas import RegionFeatures, RegionRecord


def normalize(value: float, upper: float) -> float:
    return min(max(value / upper, 0.0), 1.0)


def build_features(
    record: RegionRecord,
    terrain_change_signal: float,
    vegetation_signal: float,
    graph_signal: float,
) -> RegionFeatures:
    return RegionFeatures(
        rainfall_intensity=normalize(record.rainfall_24h_mm, 250.0),
        antecedent_rainfall=normalize(record.rainfall_7d_mm, 600.0),
        slope_factor=normalize(record.slope_deg, 50.0),
        soil_wetness=record.soil_wetness_index,
        vegetation_stress=max(vegetation_signal, terrain_change_signal),
        graph_influence=graph_signal,
    )
