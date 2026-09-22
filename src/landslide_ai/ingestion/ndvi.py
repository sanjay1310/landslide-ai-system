from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"


@dataclass(slots=True)
class NdviFetchResult:
    output_path: Path
    row_count: int
    collection: str


def fetch_regional_ndvi_from_sentinel(
    region_csv_path: str | Path,
    output_csv_path: str | Path,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    collection: str = "sentinel-2-l2a",
    resume: bool = True,
) -> NdviFetchResult:
    try:
        from pystac_client import Client
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        raise ImportError(
            "Sentinel NDVI fetching requires `pystac-client` and `rasterio`."
        ) from exc

    frame = pd.read_csv(region_csv_path)
    catalog = Client.open(EARTH_SEARCH_URL)
    rows: list[dict[str, object]] = []
    output_path = Path(output_csv_path)
    completed_region_ids: set[str] = set()

    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        rows.extend(existing.to_dict(orient="records"))
        completed_region_ids = set(existing["region_id"].astype(str).tolist())

    for index, row in enumerate(frame.to_dict(orient="records"), start=1):
        region_id = str(row["region_id"])
        if region_id in completed_region_ids:
            continue
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        search = catalog.search(
            collections=[collection],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=4,
        )
        items = sorted(
            search.items(),
            key=lambda item: item.datetime,
            reverse=True,
        )
        if not items:
            payload = {
                "region_id": row["region_id"],
                "state": row["state"],
                "district": row["district"],
                "latitude": lat,
                "longitude": lon,
                "real_ndvi": None,
                "scene_id": None,
                "scene_datetime": None,
                "cloud_cover": None,
                "ndvi_source": collection,
            }
            rows.append(payload)
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"[{index}/{len(frame)}] {row['district']}: no Sentinel scene found", flush=True)
            continue

        item = items[0]
        red_href = item.assets["red"].href
        nir_href = item.assets["nir"].href
        with rasterio.open(red_href) as red_ds, rasterio.open(nir_href) as nir_ds:
            xs, ys = transform("EPSG:4326", red_ds.crs, [lon], [lat])
            red = float(list(red_ds.sample([(xs[0], ys[0])]))[0][0])
            nir = float(list(nir_ds.sample([(xs[0], ys[0])]))[0][0])

        ndvi = None if math.isclose(nir + red, 0.0) else (nir - red) / (nir + red)
        payload = {
            "region_id": row["region_id"],
            "state": row["state"],
            "district": row["district"],
            "latitude": lat,
            "longitude": lon,
            "real_ndvi": ndvi,
            "scene_id": item.id,
            "scene_datetime": item.datetime.isoformat() if item.datetime else None,
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "ndvi_source": collection,
        }
        rows.append(payload)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(
            f"[{index}/{len(frame)}] {row['district']}: "
            f"NDVI={ndvi:.3f} scene={item.id}"
            , flush=True
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return NdviFetchResult(
        output_path=output_path,
        row_count=len(rows),
        collection=collection,
    )
