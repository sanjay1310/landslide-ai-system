from __future__ import annotations

import json

from landslide_ai.research.advanced_suite import run_advanced_research_suite


def main() -> None:
    payload = run_advanced_research_suite()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
