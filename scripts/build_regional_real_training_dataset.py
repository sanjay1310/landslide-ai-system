from __future__ import annotations

import pandas as pd


def main() -> None:
    timeseries = pd.read_csv("data/regional/kerala_uttarakhand_risk_timeseries.csv")
    terrain = pd.read_csv("data/regional/kerala_uttarakhand_terrain_real.csv")
    terrain_cols = terrain[
        [
            "region_id",
            "elevation_m",
            "slope_deg",
            "ndvi",
            "dem_elevation_m",
            "dem_slope_proxy_deg",
            "real_ndvi",
            "terrain_source",
            "slope_source",
            "ndvi_final_source",
        ]
    ].copy()
    merged = timeseries.drop(columns=["elevation_m", "slope_deg", "ndvi"], errors="ignore").merge(
        terrain_cols,
        on="region_id",
        how="left",
    )
    merged.to_csv("data/regional/kerala_uttarakhand_risk_timeseries_real.csv", index=False)
    print("Regional real training dataset build complete")
    print("Output: data/regional/kerala_uttarakhand_risk_timeseries_real.csv")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
