from landslide_ai.data.schemas import RegionRecord


class VisionAgent:
    """Simulates image-derived terrain change and vegetation stress signals."""

    def evaluate(self, record: RegionRecord) -> tuple[float, float]:
        scar_signal = min((1.0 - record.ndvi) * 1.1, 1.0)
        moisture_proxy = min(record.rainfall_24h_mm / 250.0, 1.0)
        terrain_change_signal = min((scar_signal * 0.6) + (moisture_proxy * 0.4), 1.0)
        return terrain_change_signal, scar_signal
