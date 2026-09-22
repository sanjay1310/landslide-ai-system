from __future__ import annotations

import json
from pathlib import Path

from landslide_ai.config import load_config
from landslide_ai.services.gis_service import get_gis_status


def inspect_geojson(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    sample_properties = features[0].get("properties", {}) if features else {}
    return {
        "exists": True,
        "feature_count": len(features),
        "sample_property_keys": sorted(sample_properties.keys())[:20],
    }


def main() -> None:
    config = load_config(".")
    status = get_gis_status(config)
    district_info = inspect_geojson(status.district_boundary_path)
    state_info = inspect_geojson(status.state_boundary_path)

    print("GIS Verification")
    print("=" * 16)
    print(f"District file: {status.district_boundary_path}")
    print(district_info)
    print(f"State file: {status.state_boundary_path}")
    print(state_info)


if __name__ == "__main__":
    main()
