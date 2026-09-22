from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from landslide_ai.config import AppConfig
from landslide_ai.ingestion.http import download_to_file


@dataclass(slots=True)
class IngestionArtifact:
    source_name: str
    output_path: Path
    manifest_path: Path
    mode: str


def ingest_imd_dataset(config: AppConfig) -> IngestionArtifact:
    output_path = config.raw_imd_dir / "imd_latest.csv"
    mode, metadata = _download_or_placeholder(
        url=config.imd_source_url,
        output_path=output_path,
        placeholder_text="date,region_id,rainfall_mm\n",
        source_name="imd",
    )
    return _write_manifest("imd", output_path, mode, metadata)


def ingest_nasa_power_dataset(config: AppConfig) -> IngestionArtifact:
    output_path = config.raw_nasa_dir / "nasa_power_latest.json"
    mode, metadata = _download_or_placeholder(
        url=config.nasa_power_url,
        output_path=output_path,
        placeholder_text="{}",
        source_name="nasa_power",
    )
    return _write_manifest("nasa_power", output_path, mode, metadata)


def ingest_sentinel_catalog_snapshot(config: AppConfig) -> IngestionArtifact:
    output_path = config.raw_sentinel_dir / "sentinel_catalog_snapshot.json"
    mode, metadata = _download_or_placeholder(
        url=config.sentinel_catalog_url,
        output_path=output_path,
        placeholder_text="{}",
        source_name="sentinel",
    )
    return _write_manifest("sentinel", output_path, mode, metadata)


def _download_or_placeholder(
    url: str,
    output_path: Path,
    placeholder_text: str,
    source_name: str,
) -> tuple[str, dict[str, object]]:
    if not url:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(placeholder_text, encoding="utf-8")
        return "placeholder", {"source_url": url, "reason": "source_url_not_configured"}

    try:
        download_to_file(url, output_path)
        return "downloaded", {"source_url": url}
    except Exception as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(placeholder_text, encoding="utf-8")
        return "fallback_placeholder", {
            "source_url": url,
            "reason": "download_failed",
            "error": str(exc),
            "source_name": source_name,
        }


def _write_manifest(source_name: str, output_path: Path, mode: str, metadata: dict[str, object]) -> IngestionArtifact:
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_name": source_name,
                "output_path": str(output_path),
                "mode": mode,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return IngestionArtifact(
        source_name=source_name,
        output_path=output_path,
        manifest_path=manifest_path,
        mode=mode,
    )
