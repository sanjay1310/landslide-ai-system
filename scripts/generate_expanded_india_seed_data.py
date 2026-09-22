from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class RegionSeed:
    region_id: str
    state: str
    district: str
    latitude: float
    longitude: float
    elevation_m: int
    slope_deg: int
    ndvi: float
    temperature_c: int
    neighbor_risk_mean: float
    neighbor_count: int
    tier: str
    base_rainfall: int
    rainfall_step: int


REGIONS: list[RegionSeed] = [
    RegionSeed("IN-TVM-THIRUVANANTHAPURAM", "Kerala", "Thiruvananthapuram", 8.5241, 76.9366, 64, 12, 0.60, 29, 0.28, 2, "low", 22, 4),
    RegionSeed("IN-KOL-KOLLAM", "Kerala", "Kollam", 8.8932, 76.6141, 71, 14, 0.59, 30, 0.30, 2, "low", 24, 4),
    RegionSeed("IN-PAT-PATHANAMTHITTA", "Kerala", "Pathanamthitta", 9.2648, 76.7870, 110, 18, 0.55, 27, 0.36, 3, "medium", 43, 6),
    RegionSeed("IN-ALA-ALAPPUZHA", "Kerala", "Alappuzha", 9.4981, 76.3388, 8, 7, 0.63, 30, 0.24, 2, "low", 18, 3),
    RegionSeed("IN-KOT-KOTTAYAM", "Kerala", "Kottayam", 9.5916, 76.5222, 72, 16, 0.58, 28, 0.33, 3, "low", 28, 4),
    RegionSeed("IN-IDU-IDUKKI", "Kerala", "Idukki", 10.0889, 77.0595, 1240, 38, 0.31, 21, 0.72, 5, "high", 118, 11),
    RegionSeed("IN-ERN-ERNAKULAM", "Kerala", "Ernakulam", 9.9816, 76.2999, 12, 9, 0.66, 29, 0.26, 2, "low", 20, 3),
    RegionSeed("IN-THR-THRISSUR", "Kerala", "Thrissur", 10.5276, 76.2144, 20, 11, 0.62, 29, 0.29, 2, "low", 21, 3),
    RegionSeed("IN-PAL-PALAKKAD", "Kerala", "Palakkad", 10.7867, 76.6548, 140, 21, 0.53, 30, 0.38, 3, "medium", 38, 5),
    RegionSeed("IN-MAL-MALAPPURAM", "Kerala", "Malappuram", 11.0732, 76.0740, 54, 17, 0.57, 28, 0.35, 3, "medium", 35, 5),
    RegionSeed("IN-KOZ-KOZHIKODE", "Kerala", "Kozhikode", 11.2588, 75.7804, 11, 10, 0.65, 30, 0.27, 2, "low", 19, 3),
    RegionSeed("IN-WAY-WAYANAD", "Kerala", "Wayanad", 11.6854, 76.1320, 890, 33, 0.42, 23, 0.67, 5, "high", 94, 10),
    RegionSeed("IN-KAN-KANNUR", "Kerala", "Kannur", 11.8745, 75.3704, 10, 12, 0.61, 30, 0.25, 2, "low", 17, 3),
    RegionSeed("IN-KAS-KASARAGOD", "Kerala", "Kasaragod", 12.4996, 74.9869, 26, 14, 0.59, 31, 0.23, 2, "low", 16, 3),
    RegionSeed("IN-KOD-KODAGU", "Karnataka", "Kodagu", 12.3375, 75.8069, 1170, 35, 0.39, 22, 0.64, 5, "high", 88, 9),
    RegionSeed("IN-CHI-CHIKKAMAGALURU", "Karnataka", "Chikkamagaluru", 13.3161, 75.7720, 1090, 32, 0.44, 24, 0.60, 4, "high", 76, 8),
    RegionSeed("IN-NIL-NILGIRIS", "Tamil Nadu", "Nilgiris", 11.4916, 76.7337, 2240, 29, 0.46, 19, 0.59, 4, "high", 71, 8),
    RegionSeed("IN-DIN-DINDIGUL", "Tamil Nadu", "Dindigul", 10.3673, 77.9803, 1018, 27, 0.49, 24, 0.48, 4, "medium", 54, 7),
    RegionSeed("IN-RAI-RAIGAD", "Maharashtra", "Raigad", 18.5158, 73.1822, 570, 24, 0.52, 27, 0.47, 3, "medium", 49, 6),
    RegionSeed("IN-RAT-RATNAGIRI", "Maharashtra", "Ratnagiri", 16.9902, 73.3120, 350, 22, 0.57, 28, 0.43, 3, "medium", 44, 6),
    RegionSeed("IN-SAT-SATARA", "Maharashtra", "Satara", 17.6805, 74.0183, 742, 23, 0.55, 26, 0.45, 3, "medium", 42, 5),
    RegionSeed("IN-NGO-NORTHGOA", "Goa", "North Goa", 15.4909, 73.8278, 250, 18, 0.60, 29, 0.36, 3, "low", 28, 4),
    RegionSeed("IN-DAR-DARJEELING", "West Bengal", "Darjeeling", 27.0410, 88.2663, 980, 24, 0.58, 25, 0.36, 3, "low", 28, 6),
    RegionSeed("IN-KAL-KALIMPONG", "West Bengal", "Kalimpong", 27.0622, 88.4753, 1250, 28, 0.51, 24, 0.49, 4, "medium", 45, 7),
    RegionSeed("IN-GAN-GANGTOK", "Sikkim", "Gangtok", 27.3389, 88.6065, 1510, 42, 0.40, 19, 0.61, 4, "high", 70, 9),
    RegionSeed("IN-NAM-NAMCHI", "Sikkim", "Namchi", 27.1642, 88.3639, 1675, 34, 0.47, 20, 0.57, 4, "medium", 58, 8),
    RegionSeed("IN-MAN-MANGAN", "Sikkim", "Mangan", 27.5167, 88.5333, 1400, 36, 0.45, 18, 0.59, 4, "high", 63, 8),
    RegionSeed("IN-SHI-SHIMLA", "Himachal Pradesh", "Shimla", 31.1048, 77.1734, 2206, 36, 0.37, 16, 0.58, 4, "high", 83, 9),
    RegionSeed("IN-KUL-KULLU", "Himachal Pradesh", "Kullu", 31.9579, 77.1095, 1279, 34, 0.45, 15, 0.52, 4, "medium", 58, 9),
    RegionSeed("IN-CHM-CHAMBA", "Himachal Pradesh", "Chamba", 32.5534, 76.1258, 996, 31, 0.47, 17, 0.42, 3, "low", 37, 8),
    RegionSeed("IN-KIN-KINNAUR", "Himachal Pradesh", "Kinnaur", 31.5800, 78.2300, 2960, 37, 0.41, 12, 0.56, 4, "high", 69, 7),
    RegionSeed("IN-MAN-MANDI", "Himachal Pradesh", "Mandi", 31.7081, 76.9315, 1044, 26, 0.50, 18, 0.46, 3, "medium", 46, 7),
    RegionSeed("IN-ALM-ALMORA", "Uttarakhand", "Almora", 29.5892, 79.6467, 1638, 24, 0.50, 16, 0.40, 3, "low", 31, 5),
    RegionSeed("IN-BAG-BAGESHWAR", "Uttarakhand", "Bageshwar", 29.8374, 79.7714, 1004, 27, 0.46, 17, 0.46, 4, "medium", 46, 6),
    RegionSeed("IN-CHA-CHAMOLI", "Uttarakhand", "Chamoli", 30.4200, 79.3200, 1820, 41, 0.35, 17, 0.66, 5, "high", 91, 10),
    RegionSeed("IN-CHP-CHAMPAWAT", "Uttarakhand", "Champawat", 29.3360, 80.0910, 1615, 25, 0.49, 18, 0.44, 3, "low", 34, 5),
    RegionSeed("IN-DEH-DEHRADUN", "Uttarakhand", "Dehradun", 30.3165, 78.0322, 640, 18, 0.56, 24, 0.31, 2, "low", 22, 4),
    RegionSeed("IN-HAR-HARIDWAR", "Uttarakhand", "Haridwar", 29.9457, 78.1642, 249, 10, 0.61, 29, 0.18, 2, "low", 14, 3),
    RegionSeed("IN-PIT-PITHORAGARH", "Uttarakhand", "Pithoragarh", 29.5829, 80.2182, 1650, 37, 0.38, 18, 0.63, 5, "high", 82, 9),
    RegionSeed("IN-PAU-PAURIGARHWAL", "Uttarakhand", "Pauri Garhwal", 30.1460, 78.7812, 1814, 26, 0.47, 18, 0.43, 3, "low", 33, 5),
    RegionSeed("IN-RUD-RUDRAPRAYAG", "Uttarakhand", "Rudraprayag", 30.2844, 78.9811, 895, 32, 0.43, 19, 0.55, 4, "medium", 62, 8),
    RegionSeed("IN-TEH-TEHRIGARHWAL", "Uttarakhand", "Tehri Garhwal", 30.3783, 78.4800, 1550, 31, 0.44, 18, 0.51, 4, "medium", 57, 7),
    RegionSeed("IN-UDH-UDHAMSINGHNAGAR", "Uttarakhand", "Udham Singh Nagar", 28.9617, 79.5150, 230, 8, 0.63, 28, 0.16, 2, "low", 12, 3),
    RegionSeed("IN-UTK-UTTARKASHI", "Uttarakhand", "Uttarkashi", 30.7290, 78.4430, 1158, 35, 0.41, 15, 0.57, 4, "medium", 61, 8),
    RegionSeed("IN-NAI-NAINITAL", "Uttarakhand", "Nainital", 29.3919, 79.4542, 2084, 28, 0.48, 17, 0.47, 4, "low", 38, 6),
    RegionSeed("IN-AIZ-AIZAWL", "Mizoram", "Aizawl", 23.7271, 92.7176, 1132, 39, 0.34, 22, 0.69, 5, "high", 118, 9),
    RegionSeed("IN-LUN-LUNGLEI", "Mizoram", "Lunglei", 22.8671, 92.7650, 1222, 32, 0.43, 24, 0.58, 4, "high", 78, 9),
    RegionSeed("IN-KOL-KOLASIB", "Mizoram", "Kolasib", 24.2239, 92.6787, 636, 25, 0.50, 25, 0.46, 3, "medium", 48, 7),
    RegionSeed("IN-EKH-EASTKHASIHILLS", "Meghalaya", "East Khasi Hills", 25.5788, 91.8933, 1496, 30, 0.49, 20, 0.57, 4, "high", 72, 9),
    RegionSeed("IN-WJH-WESTJAINTIAHILLS", "Meghalaya", "West Jaintia Hills", 25.4440, 92.1970, 1380, 28, 0.51, 21, 0.49, 4, "medium", 56, 8),
    RegionSeed("IN-SGH-SOUTHGAROHILLS", "Meghalaya", "South Garo Hills", 25.2890, 90.5680, 920, 24, 0.56, 26, 0.42, 3, "low", 33, 6),
    RegionSeed("IN-DIM-DIMAHASAO", "Assam", "Dima Hasao", 25.2703, 93.0176, 686, 27, 0.55, 24, 0.44, 3, "low", 34, 7),
    RegionSeed("IN-KAR-KARBIALONG", "Assam", "Karbi Anglong", 25.8450, 93.4310, 503, 23, 0.58, 27, 0.39, 3, "low", 29, 6),
    RegionSeed("IN-PAP-PAPUMPARE", "Arunachal Pradesh", "Papum Pare", 27.0844, 93.6053, 1200, 34, 0.44, 21, 0.55, 4, "medium", 69, 8),
    RegionSeed("IN-TAW-TAWANG", "Arunachal Pradesh", "Tawang", 27.5861, 91.8630, 3048, 29, 0.46, 8, 0.43, 3, "low", 24, 5),
    RegionSeed("IN-WKA-WESTKAMENG", "Arunachal Pradesh", "West Kameng", 27.3630, 92.4180, 1840, 31, 0.48, 16, 0.50, 4, "medium", 51, 7),
    RegionSeed("IN-KOH-KOHIMA", "Nagaland", "Kohima", 25.6751, 94.1086, 1444, 26, 0.52, 20, 0.40, 3, "low", 31, 6),
    RegionSeed("IN-PHE-PHEK", "Nagaland", "Phek", 25.6800, 94.5000, 1524, 29, 0.49, 19, 0.46, 3, "medium", 46, 7),
    RegionSeed("IN-SEN-SENAPATI", "Manipur", "Senapati", 25.2670, 94.0220, 1061, 28, 0.50, 21, 0.44, 3, "low", 36, 6),
]


def main() -> None:
    rainfall_rows: list[dict[str, object]] = []
    terrain_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    for region in REGIONS:
        series = _build_series(region)
        rainfall_24h = series[-1]
        rainfall_7d = sum(series[-7:])
        soil_wetness = min(round(rainfall_7d / 560.0, 2), 1.0)
        hazard_label = 1 if region.tier == "high" else 0 if region.tier == "low" else int(rainfall_24h >= 95)

        rainfall_rows.append(
            {
                "region_id": region.region_id,
                "state": region.state,
                "district": region.district,
                "latitude": region.latitude,
                "longitude": region.longitude,
                "rainfall_24h_mm": rainfall_24h,
                "rainfall_7d_mm": rainfall_7d,
                "soil_wetness_index": soil_wetness,
                "temperature_c": region.temperature_c,
                "neighbor_risk_mean": region.neighbor_risk_mean,
                "neighbor_count": region.neighbor_count,
                "label": hazard_label,
            }
        )
        terrain_rows.append(
            {
                "region_id": region.region_id,
                "state": region.state,
                "district": region.district,
                "elevation_m": region.elevation_m,
                "slope_deg": region.slope_deg,
                "ndvi": region.ndvi,
            }
        )
        for offset, rainfall in enumerate(series, start=1):
            timeseries_rows.append(
                {
                    "region_id": region.region_id,
                    "date": f"2026-04-{offset:02d}",
                    "rainfall_mm": rainfall,
                }
            )

    root = Path(".")
    raw_dir = root / "data" / "raw"
    india_dir = root / "data" / "india"
    raw_dir.mkdir(parents=True, exist_ok=True)
    india_dir.mkdir(parents=True, exist_ok=True)

    rainfall_frame = pd.DataFrame(rainfall_rows)
    terrain_frame = pd.DataFrame(terrain_rows)
    timeseries_frame = pd.DataFrame(timeseries_rows)

    rainfall_frame.to_csv(raw_dir / "india_rainfall.csv", index=False)
    terrain_frame.to_csv(raw_dir / "india_terrain.csv", index=False)
    timeseries_frame.to_csv(raw_dir / "india_rainfall_timeseries.csv", index=False)

    india_regions = rainfall_frame.merge(terrain_frame, on=["region_id", "state", "district"], how="inner")
    india_regions.to_csv(india_dir / "india_regions.csv", index=False)

    print(f"Generated {len(rainfall_frame)} India regions")
    print(f"Generated {len(timeseries_frame)} rainfall time-series rows")


def _build_series(region: RegionSeed) -> list[int]:
    series: list[int] = []
    for day_index in range(10):
        day_factor = region.base_rainfall + (region.rainfall_step * day_index)
        if region.tier == "high":
            rainfall = day_factor + (3 if day_index >= 6 else 0)
        elif region.tier == "medium":
            rainfall = day_factor - (4 if day_index < 2 else 0)
        else:
            rainfall = max(day_factor - 6, 12)
        series.append(int(rainfall))
    return series


if __name__ == "__main__":
    main()
