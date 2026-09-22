from __future__ import annotations

import json

from landslide_ai.research.reliability import generate_reliability_artifact


def main() -> None:
    payload = generate_reliability_artifact()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
