from __future__ import annotations

from landslide_ai.research.evaluation import run_full_research_evaluation, write_research_outputs


def main() -> None:
    result = run_full_research_evaluation("data/india/india_regions.csv")
    write_research_outputs(
        result,
        json_path="artifacts/research_evaluation.json",
        markdown_path="docs/RESEARCH_EVALUATION.md",
    )
    print("Research evaluation complete")
    print("JSON: artifacts/research_evaluation.json")
    print("Markdown: docs/RESEARCH_EVALUATION.md")


if __name__ == "__main__":
    main()
