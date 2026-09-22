from landslide_ai.research.fair_graph_benchmark import generate_fair_graph_benchmark


def main() -> None:
    result = generate_fair_graph_benchmark()
    print("Fair Graph Benchmark")
    print("=" * 22)
    print(f"Output: {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Calibration rows: {result.calibration_rows}")
    print(f"Evaluation rows: {result.evaluation_rows}")


if __name__ == "__main__":
    main()
