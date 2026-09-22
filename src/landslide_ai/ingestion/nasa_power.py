from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from landslide_ai.config import AppConfig
from landslide_ai.data.loader import load_region_records


@dataclass(slots=True)
class NasaPowerFetchResult:
    daily_output_path: Path
    latest_output_path: Path
    row_count: int
    mode: str


def build_nasa_power_url(
    base_url: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:
    query = urlencode(
        {
            "parameters": "PRECTOTCORR,T2M",
            "community": "AG",
            "latitude": latitude,
            "longitude": longitude,
            "start": start_date,
            "end": end_date,
            "format": "JSON",
        }
    )
    return f"{base_url}?{query}"


def ingest_nasa_power_by_regions(config: AppConfig, region_csv_path: str | Path) -> NasaPowerFetchResult:
    records = load_region_records(region_csv_path)
    daily_output_path = config.raw_nasa_dir / "nasa_power_daily.csv"
    latest_output_path = config.raw_nasa_dir / "nasa_power_latest.csv"
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.nasa_power_url:
        _write_placeholder_outputs(daily_output_path, latest_output_path, records)
        return NasaPowerFetchResult(
            daily_output_path=daily_output_path,
            latest_output_path=latest_output_path,
            row_count=len(records),
            mode="placeholder",
        )

    rows: list[dict[str, object]] = []
    mode = "downloaded"
    for record in records:
        try:
            payload = _fetch_region_payload(
                base_url=config.nasa_power_url,
                latitude=record.latitude,
                longitude=record.longitude,
                start_date=config.nasa_power_start_date,
                end_date=config.nasa_power_end_date,
            )
            rows.extend(
                _payload_to_rows(
                    record.region_id,
                    record.state,
                    record.district,
                    record.latitude,
                    record.longitude,
                    payload,
                )
            )
        except Exception as exc:
            mode = "partial_fallback"
            rows.append(
                {
                    "region_id": record.region_id,
                    "state": record.state,
                    "district": record.district,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "date": config.nasa_power_end_date,
                    "rainfall_mm": record.rainfall_24h_mm,
                    "temperature_c": record.temperature_c,
                    "source_mode": f"fallback:{exc}",
                }
            )

    daily_frame = pd.DataFrame(rows)
    daily_frame.to_csv(daily_output_path, index=False)

    latest_frame = build_latest_nasa_features(daily_frame)
    latest_frame.to_csv(latest_output_path, index=False)
    return NasaPowerFetchResult(
        daily_output_path=daily_output_path,
        latest_output_path=latest_output_path,
        row_count=len(daily_frame),
        mode=mode,
    )


def build_latest_nasa_features(daily_frame: pd.DataFrame) -> pd.DataFrame:
    working = daily_frame.copy()
    working["date"] = pd.to_datetime(working["date"].astype(str), format="%Y%m%d", errors="coerce")
    working["rainfall_mm"] = pd.to_numeric(working["rainfall_mm"], errors="coerce")
    working["temperature_c"] = pd.to_numeric(working["temperature_c"], errors="coerce")
    working.loc[working["rainfall_mm"] < 0.0, "rainfall_mm"] = pd.NA
    working.loc[working["temperature_c"] < -100.0, "temperature_c"] = pd.NA
    working = working.sort_values(["region_id", "date"]).reset_index(drop=True)
    features: list[dict[str, object]] = []
    for region_id, group in working.groupby("region_id"):
        valid_group = group.dropna(subset=["rainfall_mm", "temperature_c"]).copy()
        if valid_group.empty:
            continue
        latest_row = valid_group.iloc[-1]
        features.append(
            {
                "region_id": region_id,
                "state": latest_row["state"],
                "district": latest_row["district"],
                "latitude": float(latest_row["latitude"]),
                "longitude": float(latest_row["longitude"]),
                "rainfall_24h_mm": float(latest_row["rainfall_mm"]),
                "rainfall_7d_mm": float(valid_group["rainfall_mm"].tail(7).sum()),
                "temperature_c": float(valid_group["temperature_c"].tail(3).mean()),
                "latest_valid_date": latest_row["date"].date().isoformat(),
                "valid_observation_count": int(len(valid_group)),
                "source_mode": latest_row.get("source_mode", "downloaded"),
            }
        )
    return pd.DataFrame(features)


def _fetch_region_payload(
    base_url: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    url = build_nasa_power_url(
        base_url=base_url,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _payload_to_rows(
    region_id: str,
    state: str,
    district: str,
    latitude: float,
    longitude: float,
    payload: dict[str, object],
) -> list[dict[str, object]]:
    parameter_block = payload["properties"]["parameter"]
    rainfall_map = parameter_block.get("PRECTOTCORR", {})
    temperature_map = parameter_block.get("T2M", {})
    rows: list[dict[str, object]] = []
    for date_key, rainfall_value in rainfall_map.items():
        rows.append(
            {
                "region_id": region_id,
                "state": state,
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "date": date_key,
                "rainfall_mm": float(rainfall_value),
                "temperature_c": float(temperature_map.get(date_key, 0.0)),
                "source_mode": "downloaded",
            }
        )
    return rows


def _write_placeholder_outputs(daily_output_path: Path, latest_output_path: Path, records: list) -> None:
    daily_rows = []
    latest_rows = []
    day_keys = ["20260128", "20260129", "20260130", "20260131"]
    rainfall_scales = [0.78, 0.86, 0.94, 1.0]
    for record in records:
        for day_key, rainfall_scale in zip(day_keys, rainfall_scales, strict=True):
            daily_rows.append(
                {
                    "region_id": record.region_id,
                    "state": record.state,
                    "district": record.district,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "date": day_key,
                    "rainfall_mm": round(record.rainfall_24h_mm * rainfall_scale, 2),
                    "temperature_c": record.temperature_c,
                    "source_mode": "placeholder",
                }
            )
        latest_rows.append(
            {
                "region_id": record.region_id,
                "state": record.state,
                "district": record.district,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "rainfall_24h_mm": record.rainfall_24h_mm,
                "rainfall_7d_mm": record.rainfall_7d_mm,
                "temperature_c": record.temperature_c,
                "source_mode": "placeholder",
            }
        )
    pd.DataFrame(daily_rows).to_csv(daily_output_path, index=False)
    pd.DataFrame(latest_rows).to_csv(latest_output_path, index=False)
