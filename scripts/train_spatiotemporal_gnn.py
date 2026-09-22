from landslide_ai.training.train_spatiotemporal_gnn import train_spatiotemporal_graph_model


def main() -> None:
    result = train_spatiotemporal_graph_model()
    print("Spatio-Temporal GNN Training")
    print("=" * 30)
    print(f"Artifact: {result.artifact_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Experiment log: {result.experiment_log_path}")
    print(f"Sample dates: {result.sample_dates}")
    print(f"Node count: {result.node_count}")
    print(f"Validation F1: {result.validation_f1:.4f}")
    print(f"Validation ROC-AUC: {result.validation_roc_auc:.4f}")
    print(f"Validation PR-AUC: {result.validation_pr_auc:.4f}")
    print(f"Validation uncertainty mean: {result.validation_uncertainty_mean:.4f}")


if __name__ == "__main__":
    main()
