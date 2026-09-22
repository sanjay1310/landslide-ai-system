from __future__ import annotations

from dataclasses import asdict
import json

from landslide_ai.research.external_baselines import run_external_baseline_comparison


def main() -> None:
    rows = run_external_baseline_comparison()
    print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
