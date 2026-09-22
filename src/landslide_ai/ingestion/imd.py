from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from landslide_ai.config import AppConfig


@dataclass(slots=True)
class IMDFetchResult:
    raw_output_path: Path
    normalized_output_path: Path
    row_count: int
    mode: str


def ingest_imd_compatible_dataset(config: AppConfig) -> IMDFetchResult:
    raw_output_path = config.raw_imd_dir / "imd_latest_source.html"
    normalized_output_path = config.raw_imd_dir / "imd_latest_normalized.csv"
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.imd_source_url:
        _write_placeholder_imd(normalized_output_path)
        raw_output_path.write_text("", encoding="utf-8")
        return IMDFetchResult(
            raw_output_path=raw_output_path,
            normalized_output_path=normalized_output_path,
            row_count=0,
            mode="placeholder",
        )

    try:
        content = _read_source(config.imd_source_url)
        raw_output_path.write_text(content, encoding="utf-8")
        normalized_frame = normalize_imd_content(content)
        normalized_frame.to_csv(normalized_output_path, index=False)
        return IMDFetchResult(
            raw_output_path=raw_output_path,
            normalized_output_path=normalized_output_path,
            row_count=len(normalized_frame),
            mode=_imd_mode(config.imd_source_url),
        )
    except Exception:
        _write_placeholder_imd(normalized_output_path)
        raw_output_path.write_text("", encoding="utf-8")
        return IMDFetchResult(
            raw_output_path=raw_output_path,
            normalized_output_path=normalized_output_path,
            row_count=0,
            mode="fallback_placeholder",
        )


def _imd_mode(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return "downloaded_url"
    return "local_file"


def _read_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        with urlopen(source, timeout=30) as response:
            return response.read().decode("utf-8", errors="ignore")
    return Path(source).read_text(encoding="utf-8", errors="ignore")


def normalize_imd_content(content: str) -> pd.DataFrame:
    stripped = content.lstrip()
    first_line = stripped.splitlines()[0].lower() if stripped.splitlines() else ""
    if "," in first_line and any(token in first_line for token in ["state", "district", "rainfall"]):
        frame = pd.read_csv(StringIO(content))
    else:
        tables = pd.read_html(StringIO(content))
        if not tables:
            return _placeholder_imd_frame()
        frame = _find_best_imd_table(tables)

    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    column_map = {
        "state_name": "state",
        "district_name": "district",
        "rainfall_(mm)": "rainfall_mm",
        "rainfall_mm": "rainfall_mm",
        "actual_rainfall": "rainfall_mm",
        "date_of_observation": "date",
        "observation_date": "date",
    }
    frame = frame.rename(columns=column_map)

    required = ["state", "district", "rainfall_mm"]
    if not all(column in frame.columns for column in required):
        return _placeholder_imd_frame()

    if "date" not in frame.columns:
        frame["date"] = pd.Timestamp.now().date().isoformat()

    frame["state"] = frame["state"].astype(str)
    frame["district"] = frame["district"].astype(str)
    frame["rainfall_mm"] = pd.to_numeric(frame["rainfall_mm"], errors="coerce").fillna(0.0)
    return frame[["date", "state", "district", "rainfall_mm"]].reset_index(drop=True)


def _find_best_imd_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    best_table = tables[0]
    best_score = -1
    keywords = {"state", "district", "rainfall"}
    for table in tables:
        score = 0
        for column in table.columns:
            column_text = str(column).lower()
            if any(keyword in column_text for keyword in keywords):
                score += 1
        if score > best_score:
            best_score = score
            best_table = table
    return best_table


def _write_placeholder_imd(output_path: Path) -> None:
    _placeholder_imd_frame().to_csv(output_path, index=False)


def _placeholder_imd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp.now().date().isoformat(), "state": "Kerala", "district": "Idukki", "rainfall_mm": 0.0}
        ]
    )
