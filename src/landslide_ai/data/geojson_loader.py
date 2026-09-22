from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_geojson(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def geojson_feature_count(path: str | Path) -> int:
    payload = load_geojson(path)
    return len(payload.get("features", []))
