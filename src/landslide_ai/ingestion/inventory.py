from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd
from math import radians, sin, cos, sqrt, atan2


PROJECT_INVENTORY_COLUMNS = [
    "state",
    "district",
    "event_date",
    "source",
    "event_id",
    "trigger",
    "notes",
]


STATE_ALIASES = {
    "uttaranchal": "uttarakhand",
    "orissa": "odisha",
    "pondicherry": "puducherry",
}


DISTRICT_ALIASES = {
    "uttar kashi": "uttarkashi",
    "pauri garwal": "pauri garhwal",
    "pauri garhwal district": "pauri garhwal",
    "north goa district": "north goa",
    "east khasi hills district": "east khasi hills",
    "west jaintia hills district": "west jaintia hills",
    "udham singh nagar": "udham singh nagar",
}


NASA_GLC_COLUMNS = [
    "event_id",
    "event_date",
    "state",
    "district",
    "latitude",
    "longitude",
    "trigger",
    "notes",
]


def normalize_inventory_name(value: str, alias_map: dict[str, str] | None = None) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\bdistrict\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if alias_map and normalized in alias_map:
        return alias_map[normalized]
    return normalized


def create_inventory_template(output_csv_path: str | Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "state": "Kerala",
                "district": "Idukki",
                "event_date": "2018-08-16",
                "source": "NRSC Landslide Atlas",
                "event_id": "sample-001",
                "trigger": "Extreme rainfall",
                "notes": "Replace this sample row with verified inventory records.",
            }
        ],
        columns=PROJECT_INVENTORY_COLUMNS,
    )
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def standardize_inventory_frame(
    frame: pd.DataFrame,
    source_name: str = "external_inventory",
) -> pd.DataFrame:
    working = frame.copy()
    date_column = _find_date_column(working)
    state_column = _find_column(working, ["state", "state_name", "province", "admin1"])
    district_column = _find_column(working, ["district", "district_name", "admin2", "subregion"])
    event_id_column = _find_column(working, ["event_id", "id", "landslide_id"], required=False)
    trigger_column = _find_column(working, ["trigger", "cause", "landslide_trigger"], required=False)
    notes_column = _find_column(working, ["notes", "description", "comments"], required=False)

    standardized = pd.DataFrame()
    standardized["state"] = working[state_column].astype(str).map(lambda item: normalize_inventory_name(item, STATE_ALIASES))
    standardized["district"] = working[district_column].astype(str).map(
        lambda item: normalize_inventory_name(item, DISTRICT_ALIASES)
    )
    standardized["event_date"] = pd.to_datetime(working[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    standardized["source"] = source_name
    standardized["event_id"] = (
        working[event_id_column].astype(str) if event_id_column else pd.Series(range(1, len(working) + 1)).astype(str)
    )
    standardized["trigger"] = working[trigger_column].astype(str) if trigger_column else ""
    standardized["notes"] = working[notes_column].astype(str) if notes_column else ""
    standardized = standardized.dropna(subset=["event_date"]).copy()
    standardized = standardized[PROJECT_INVENTORY_COLUMNS]
    standardized = standardized.drop_duplicates(subset=["state", "district", "event_date", "event_id"]).reset_index(drop=True)
    return standardized


def write_standardized_inventory(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    source_name: str = "external_inventory",
) -> Path:
    frame = pd.read_csv(input_csv_path)
    standardized = standardize_inventory_frame(frame, source_name=source_name)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)
    return output_path


def prepare_nasa_global_landslide_catalog_frame(
    frame: pd.DataFrame,
    allowed_states: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    working = frame.copy()
    required_columns = {
        "event_id",
        "event_date",
        "admin_division_name",
        "latitude",
        "longitude",
    }
    missing = required_columns - set(working.columns)
    if missing:
        raise ValueError(f"NASA Global Landslide Catalog frame is missing required columns: {sorted(missing)}")

    working["state"] = working["admin_division_name"].astype(str)
    district_hint = working.get("gazeteer_closest_point", pd.Series(index=working.index, dtype=str)).fillna("")
    fallback_hint = working.get("location_description", pd.Series(index=working.index, dtype=str)).fillna("")
    working["district"] = district_hint.where(district_hint.astype(str).str.strip() != "", fallback_hint)
    working["trigger"] = working.get("landslide_trigger", pd.Series(index=working.index, dtype=str)).fillna("")

    note_parts = [
        "NASA Global Landslide Catalog import",
        "title=" + working.get("event_title", pd.Series(index=working.index, dtype=str)).fillna("").astype(str),
        "location=" + working.get("location_description", pd.Series(index=working.index, dtype=str)).fillna("").astype(str),
        "accuracy=" + working.get("location_accuracy", pd.Series(index=working.index, dtype=str)).fillna("").astype(str),
        "source_name=" + working.get("source_name", pd.Series(index=working.index, dtype=str)).fillna("").astype(str),
        "source_link=" + working.get("source_link", pd.Series(index=working.index, dtype=str)).fillna("").astype(str),
    ]
    working["notes"] = note_parts[0]
    for part in note_parts[1:]:
        working["notes"] = working["notes"] + " | " + part

    prepared = working[NASA_GLC_COLUMNS].copy()
    prepared["state_key"] = prepared["state"].map(lambda item: normalize_inventory_name(item, STATE_ALIASES))
    if allowed_states:
        allowed_state_keys = {normalize_inventory_name(item, STATE_ALIASES) for item in allowed_states}
        prepared = prepared[prepared["state_key"].isin(allowed_state_keys)].copy()

    prepared = prepared.drop(columns=["state_key"])
    parsed_dates = pd.to_datetime(
        prepared["event_date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    fallback_dates = pd.to_datetime(prepared["event_date"], format="mixed", errors="coerce")
    prepared["event_date"] = parsed_dates.fillna(fallback_dates).dt.strftime("%Y-%m-%d")
    prepared["latitude"] = pd.to_numeric(prepared["latitude"], errors="coerce")
    prepared["longitude"] = pd.to_numeric(prepared["longitude"], errors="coerce")
    prepared = prepared.dropna(subset=["event_date", "latitude", "longitude"]).copy()
    prepared = prepared.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    return prepared


def write_coordinate_mapped_inventory(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    region_metadata_csv_path: str | Path,
    source_name: str = "external_inventory",
    allowed_states: tuple[str, ...] | None = None,
) -> Path:
    raw = pd.read_csv(input_csv_path)
    regions = pd.read_csv(region_metadata_csv_path)

    standardized = standardize_inventory_frame(raw, source_name=source_name)
    if {"latitude", "longitude"}.issubset(raw.columns):
        standardized["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
        standardized["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    else:
        standardized["latitude"] = pd.NA
        standardized["longitude"] = pd.NA

    region_frame = regions.copy()
    region_frame["state_key"] = region_frame["state"].astype(str).map(lambda item: normalize_inventory_name(item, STATE_ALIASES))
    region_frame["district_key"] = region_frame["district"].astype(str).map(
        lambda item: normalize_inventory_name(item, DISTRICT_ALIASES)
    )
    if allowed_states:
        allowed_state_keys = {
            normalize_inventory_name(item, STATE_ALIASES)
            for item in allowed_states
        }
        region_frame = region_frame[region_frame["state_key"].isin(allowed_state_keys)].copy()

    mapped_rows: list[dict[str, object]] = []
    for row in standardized.to_dict(orient="records"):
        state_key = str(row["state"]).strip().lower()
        district_key = str(row["district"]).strip().lower()
        candidates = region_frame[region_frame["state_key"] == state_key].copy()
        if candidates.empty:
            candidates = region_frame.copy()

        exact_matches = candidates[candidates["district_key"] == district_key]
        if not exact_matches.empty:
            match = exact_matches.iloc[0]
        else:
            match = _match_inventory_row_to_region(row, candidates)
            if match is None:
                continue

        mapped_rows.append(
            {
                "state": match["state"],
                "district": match["district"],
                "event_date": row["event_date"],
                "source": row["source"],
                "event_id": row["event_id"],
                "trigger": row.get("trigger", ""),
                "notes": _build_mapping_note(row, match),
            }
        )

    output = pd.DataFrame(mapped_rows, columns=PROJECT_INVENTORY_COLUMNS)
    output = output.drop_duplicates(subset=["state", "district", "event_date", "event_id"]).reset_index(drop=True)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def merge_mapped_inventory_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=PROJECT_INVENTORY_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=PROJECT_INVENTORY_COLUMNS)

    combined = combined.copy()
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=["event_date"]).copy()
    combined["source_priority"] = combined["source"].map(
        {
            "provided_inventory": 0,
            "nasa_glc": 1,
        }
    ).fillna(5)
    combined = combined.sort_values(
        ["state", "district", "event_date", "source_priority", "event_id"],
        kind="stable",
    )

    merged_rows: list[dict[str, str]] = []
    for _, group in combined.groupby(["state", "district", "event_date"], sort=False):
        sources = _join_unique(group["source"])
        event_ids = _join_unique(group["event_id"])
        triggers = _join_unique(group["trigger"])
        notes = _join_unique(group["notes"])
        primary = group.iloc[0]
        source_value = primary["source"] if len(sources.split("; ")) == 1 else "multi_source"
        merged_rows.append(
            {
                "state": primary["state"],
                "district": primary["district"],
                "event_date": primary["event_date"],
                "source": source_value,
                "event_id": event_ids,
                "trigger": triggers,
                "notes": notes,
            }
        )

    return pd.DataFrame(merged_rows, columns=PROJECT_INVENTORY_COLUMNS)


def _find_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(f"Required column not found. Expected one of: {', '.join(candidates)}")
    return None


def _find_date_column(frame: pd.DataFrame) -> str:
    return _find_column(frame, ["event_date", "date", "incident_date", "landslide_date"])


def _match_inventory_row_to_region(row: dict[str, object], candidates: pd.DataFrame) -> pd.Series | None:
    latitude = row.get("latitude")
    longitude = row.get("longitude")
    district_key = str(row.get("district", "")).strip().lower()

    token_matches = candidates[
        candidates["district_key"].map(lambda candidate: candidate and candidate in district_key or district_key in candidate)
    ]
    if len(token_matches) == 1:
        return token_matches.iloc[0]

    if pd.notna(latitude) and pd.notna(longitude):
        scored = candidates.copy()
        scored["distance_km"] = scored.apply(
            lambda item: _haversine_km(
                float(latitude),
                float(longitude),
                float(item["latitude"]),
                float(item["longitude"]),
            ),
            axis=1,
        )
        scored = scored.sort_values("distance_km")
        if not scored.empty:
            return scored.iloc[0]

    if not token_matches.empty:
        return token_matches.iloc[0]
    return None


def _build_mapping_note(row: dict[str, object], match: pd.Series) -> str:
    original_state = row.get("state", "")
    original_district = row.get("district", "")
    return (
        f"Mapped from raw inventory state='{original_state}' district='{original_district}' "
        f"to region district='{match['district']}'."
    )


def _join_unique(values: pd.Series) -> str:
    unique_values: list[str] = []
    for value in values.fillna("").astype(str):
        cleaned = value.strip()
        if cleaned and cleaned not in unique_values:
            unique_values.append(cleaned)
    return "; ".join(unique_values)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius_km * c
