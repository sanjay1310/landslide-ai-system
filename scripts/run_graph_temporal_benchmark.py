from __future__ import annotations

import argparse
import json
import os


def _configure_safe_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reduced temporal GNN benchmark on a chosen dataset CSV.")
    parser.add_argument(
        "--dataset-csv",
        default=os.getenv("LANDSLIDE_TEMPORAL_DATASET_CSV", "data/regional/kerala_uttarakhand_risk_timeseries.csv"),
        help="Path to the temporal district-date dataset CSV.",
    )
    parser.add_argument("--negative-stride", type=int, default=int(os.getenv("LANDSLIDE_GNN_NEGATIVE_STRIDE", "30")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("LANDSLIDE_GNN_HIDDEN_DIM", "16")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("LANDSLIDE_GNN_EPOCHS", "8")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LANDSLIDE_GNN_LR", "0.01")))
    args = parser.parse_args()

    _configure_safe_runtime()

    from landslide_ai.research.large_scale import benchmark_reduced_temporal_gnn

    result = benchmark_reduced_temporal_gnn(
        dataset_csv_path=args.dataset_csv,
        negative_stride=args.negative_stride,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print("Reduced temporal GNN benchmark complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
