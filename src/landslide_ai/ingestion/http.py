from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


def download_to_file(url: str, destination: str | Path, timeout: int = 30) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        payload = response.read()
    target.write_bytes(payload)
    return target
