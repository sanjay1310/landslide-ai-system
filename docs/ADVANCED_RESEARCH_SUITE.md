# Advanced Research Suite

## Best Evaluation
- Best model: graph_gnn
- Decision threshold: 0.5879
- Metrics: `{"accuracy": 0.7824456114028507, "precision": 0.1891891891891892, "recall": 0.27631578947368424, "f1": 0.22459893048128343, "roc_auc": 0.7361123490351621, "pr_auc": 0.22525229417073783, "brier_score": 0.2159783701980523}`

## Ablation Study
`[
  {
    "ablation_name": "full_system",
    "feature_count": 24,
    "threshold": 0.07596967354127228,
    "accuracy": 0.9874700571217984,
    "precision": 0.1388888888888889,
    "recall": 0.06578947368421052,
    "f1": 0.08928571428571429,
    "roc_auc": 0.8563877259333466,
    "pr_auc": 0.060458769432339056,
    "brier_score": 0.009064631351609707
  },
  {
    "ablation_name": "without_graph",
    "feature_count": 21,
    "threshold": 0.06872939595395954,
    "accuracy": 0.9867330016583747,
    "precision": 0.13636363636363635,
    "recall": 0.07894736842105263,
    "f1": 0.1,
    "roc_auc": 0.8628447125315303,
    "pr_auc": 0.062191132743012995,
    "brier_score": 0.009047493047297248
  },
  {
    "ablation_name": "without_temporal",
    "feature_count": 15,
    "threshold": 0.10163336554703226,
    "accuracy": 0.989742644800688,
    "precision": 0.058823529411764705,
    "recall": 0.006578947368421052,
    "f1": 0.011834319526627219,
    "roc_auc": 0.8485728550404469,
    "pr_auc": 0.06896592232205923,
    "brier_score": 0.009078311536165605
  },
  {
    "ablation_name": "without_vegetation",
    "feature_count": 21,
    "threshold": 0.039095155087783225,
    "accuracy": 0.9806522940851299,
    "precision": 0.07772020725388601,
    "recall": 0.09868421052631579,
    "f1": 0.08695652173913043,
    "roc_auc": 0.8451493876671965,
    "pr_auc": 0.055086321104685206,
    "brier_score": 0.009127752415664893
  }
]`

## Explainability
- Global importance method: shap
- Global top features: `[{"feature": "rainfall_7d_mm", "importance": 0.13427}, {"feature": "vegetation_vulnerability", "importance": 0.076342}, {"feature": "soil_rainfall_interaction", "importance": 0.056398}, {"feature": "rainfall_3d_mm", "importance": 0.045851}, {"feature": "temperature_rainfall_interaction", "importance": 0.020411}, {"feature": "rainfall_ratio_24h_to_7d", "importance": 0.018526}, {"feature": "rainfall_14d_mm", "importance": 0.01791}, {"feature": "soil_wetness_index", "importance": 0.01773}, {"feature": "rainfall_7d_to_14d_ratio", "importance": 0.016973}, {"feature": "rainfall_3d_to_14d_ratio", "importance": 0.016531}]`
- District explanation: `{"method": "linear_contribution", "top_features": ["neighbor_risk_mean", "rainfall_7d_mm", "neighbor_pressure", "vegetation_vulnerability", "soil_rainfall_interaction"], "top_feature_scores": [1.509, -1.2287, -1.1428, -0.823, 0.6955], "row_region_id": "IN-ALA-ALAPPUZHA", "row_date": "2014-12-21 00:00:00"}`
- ST-GNN district explanation: `{"region_id": "IN-ALA-ALAPPUZHA", "state": "Kerala", "district": "Alappuzha", "date": "2008-05-23", "spatiotemporal_gnn_probability": 0.460669, "uncertainty_score": 0.006048, "confidence_level": "High", "recommended_interpretation": "Moderate signal. Watch rainfall persistence and neighboring districts.", "top_driver": "rainfall_7d_to_14d_ratio", "top_driver_score": 1.2899, "top_feature_drivers": ["rainfall_7d_to_14d_ratio", "day_of_year_cos", "neighbor_risk_mean"], "top_feature_driver_scores": [1.2899, 1.0698, 1.0118], "top_neighbor_influences": [{"region_id": "IN-KOT-KOTTAYAM", "weight": 0.03}, {"region_id": "IN-PAT-PATHANAMTHITTA", "weight": 0.0271}, {"region_id": "IN-ERN-ERNAKULAM", "weight": 0.0262}], "sequence_length": 10, "artifact_metadata_model_type": "spatiotemporal_graph_gnn"}`
