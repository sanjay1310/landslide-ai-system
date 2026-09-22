from __future__ import annotations

import pandas as pd


def main() -> None:
    base = pd.read_csv("data/regional/kerala_uttarakhand_terrain.csv")
    dem = pd.read_csv("data/regional/kerala_uttarakhand_dem.csv")
    ndvi = pd.read_csv("data/regional/kerala_uttarakhand_ndvi.csv")

    merged = (
        base.merge(
            dem[["region_id", "dem_elevation_m", "dem_slope_proxy_deg", "dem_dataset"]],
            on="region_id",
            how="left",
        )
        .merge(
            ndvi[["region_id", "real_ndvi", "scene_id", "scene_datetime", "cloud_cover", "ndvi_source"]],
            on="region_id",
            how="left",
        )
    )
    merged["elevation_m"] = merged["dem_elevation_m"].fillna(merged["elevation_m"])
    merged["slope_deg"] = merged["dem_slope_proxy_deg"].fillna(merged["slope_deg"])
    merged["ndvi"] = merged["real_ndvi"].fillna(merged["ndvi"])
    merged["terrain_source"] = merged["dem_dataset"].fillna("seeded_terrain")
    merged["slope_source"] = merged["dem_slope_proxy_deg"].map(lambda value: "dem_proxy" if pd.notna(value) else "seeded_terrain")
    merged["ndvi_final_source"] = merged["real_ndvi"].map(lambda value: "sentinel_2" if pd.notna(value) else "seeded_terrain")
    output = merged[
        [
            "region_id",
            "state",
            "district",
            "elevation_m",
            "slope_deg",
            "ndvi",
            "dem_elevation_m",
            "dem_slope_proxy_deg",
            "real_ndvi",
            "terrain_source",
            "slope_source",
            "ndvi_final_source",
            "scene_id",
            "scene_datetime",
            "cloud_cover",
        ]
    ].copy()
    output.to_csv("data/regional/kerala_uttarakhand_terrain_real.csv", index=False)
    print("Regional real terrain build complete")
    print("Output: data/regional/kerala_uttarakhand_terrain_real.csv")
    print(f"Rows: {len(output)}")


if __name__ == "__main__":
    main()
