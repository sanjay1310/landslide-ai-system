from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import sklearn

from landslide_ai.utils.optional_dependencies import torch_available


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "torch": None,
    }
    if torch_available():
        import torch

        versions["torch"] = getattr(torch, "__version__", None)
    return versions


def build_artifact_lineage(
    artifact_path: str | Path,
    metadata_path: str | Path | None = None,
) -> dict[str, object]:
    artifact = Path(artifact_path)
    metadata = Path(metadata_path) if metadata_path is not None else artifact.with_name(f"{artifact.stem}_metadata.json")
    metadata_payload: dict[str, object] = {}
    if metadata.exists():
        try:
            metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata_payload = {}

    lineage = {
        "artifact_path": str(artifact),
        "artifact_exists": artifact.exists(),
        "metadata_path": str(metadata),
        "metadata_exists": metadata.exists(),
        "artifact_modified_at": datetime.fromtimestamp(artifact.stat().st_mtime, tz=timezone.utc).isoformat()
        if artifact.exists()
        else None,
        "runtime_versions": runtime_versions(),
        "metadata": metadata_payload,
    }
    lineage["version_match"] = _version_match(metadata_payload)
    return lineage


def write_experiment_log(
    experiment_name: str,
    parameters: dict[str, object],
    metrics: dict[str, object],
    artifacts: dict[str, object],
    output_dir: str | Path = "artifacts/experiments",
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = target_dir / f"{timestamp}_{experiment_name}.json"
    payload = {
        "experiment_name": experiment_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": runtime_versions(),
        "parameters": parameters,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _version_match(metadata_payload: dict[str, object]) -> bool | None:
    metadata_sklearn = metadata_payload.get("scikit_learn_version", metadata_payload.get("scikit_learn"))
    metadata_python = metadata_payload.get("python_version", metadata_payload.get("python"))
    if metadata_sklearn is None and metadata_python is None:
        return None
    versions = runtime_versions()
    return bool(
        (metadata_sklearn is None or metadata_sklearn == versions["scikit_learn"])
        and (metadata_python is None or metadata_python == versions["python"])
    )
