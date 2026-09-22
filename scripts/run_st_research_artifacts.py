from __future__ import annotations

from dataclasses import asdict
import json

from landslide_ai.research.case_studies import generate_case_study_artifact
from landslide_ai.research.error_analysis import generate_error_analysis_artifact
from landslide_ai.research.spatiotemporal_ablation_fast import run_fast_spatiotemporal_ablation
from landslide_ai.research.st_calibration import generate_st_calibration_artifact


def main() -> None:
    payload = {
        "st_ablation_fast": [asdict(row) for row in run_fast_spatiotemporal_ablation()],
        "st_calibration": [asdict(row) for row in generate_st_calibration_artifact()],
        "error_analysis": generate_error_analysis_artifact(),
        "case_studies": generate_case_study_artifact(),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
