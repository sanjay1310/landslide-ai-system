# Large-Scale Risk Evaluation

## Dataset
- Total district-date rows: 81405
- Train rows: 65124
- Validation rows: 16281
- Label source: inventory
- Model: logistic_regression_balanced
- Decision threshold: 0.939
- Feature count: 24

## Temporal Evaluation Metrics
- Accuracy: 0.993
- Precision: 0.080
- Recall: 0.156
- F1: 0.106
- ROC-AUC: 0.846

- PR-AUC: 0.051
- Brier score: 0.086

## Confusion Matrix
- True negative: 16149
- False positive: 87
- False negative: 38
- True positive: 7

## Before vs After
| Metric | Previous baseline | Updated pipeline |
| --- | ---: | ---: |
| Accuracy | 0.9919 | 0.9928 |
| Precision | 0.0693 | 0.0805 |
| Recall | 0.1556 | 0.1556 |
| F1 | 0.0959 | 0.1061 |
| ROC-AUC | 0.8547 | 0.8465 |
| PR-AUC | 0.0710 | 0.0508 |

The updated pipeline improved thresholded classification quality, especially precision and F1, but did not improve ranking quality on PR-AUC.

## Model Comparison
[
  {
    "model_name": "logistic_regression_balanced",
    "threshold": 0.939420,
    "accuracy": 0.992752,
    "precision": 0.080460,
    "recall": 0.15555555555555556,
    "f1": 0.106061,
    "roc_auc": 0.846488,
    "pr_auc": 0.050814,
    "brier_score": 0.085712
  },
  {
    "model_name": "logistic_regression_balanced_calibrated",
    "threshold": 0.045303,
    "accuracy": 0.990971,
    "precision": 0.067797,
    "recall": 0.17777777777777778,
    "f1": 0.098160,
    "roc_auc": 0.846488,
    "pr_auc": 0.050814,
    "brier_score": 0.002696
  },
  {
    "model_name": "gradient_boosting",
    "threshold": 0.188539,
    "accuracy": 0.995025,
    "precision": 0.090909,
    "recall": 0.08888888888888889,
    "f1": 0.089888,
    "roc_auc": 0.831821,
    "pr_auc": 0.050403,
    "brier_score": 0.002973
  },
  {
    "model_name": "gradient_boosting_calibrated",
    "threshold": 0.033784,
    "accuracy": 0.954118,
    "precision": 0.028226,
    "recall": 0.4666666666666667,
    "f1": 0.053232,
    "roc_auc": 0.819627,
    "pr_auc": 0.023999,
    "brier_score": 0.002729
  }
]

## Sequence Model Benchmark
[
  {
    "model_name": "gru_sequence_classifier",
    "threshold": 0.614546,
    "accuracy": 0.515479,
    "precision": 0.005099,
    "recall": 0.888889,
    "f1": 0.010139,
    "roc_auc": 0.710853,
    "pr_auc": 0.004979,
    "brier_score": 0.284960
  },
  {
    "model_name": "lstm_sequence_classifier",
    "threshold": 0.551405,
    "accuracy": 0.522365,
    "precision": 0.005044,
    "recall": 0.866667,
    "f1": 0.010030,
    "roc_auc": 0.731662,
    "pr_auc": 0.030732,
    "brier_score": 0.208906
  }
]

The first GRU and LSTM sequence benchmarks did not beat the updated logistic baseline on F1, ROC-AUC, or PR-AUC. They were able to push recall very high, but at the cost of precision and overall usefulness.

## Reduced Temporal GNN Benchmark

This benchmark uses the same real Kerala-Uttarakhand dataset, but keeps all positive windows and a structured sample of negatives so the temporal graph remains computationally tractable.

[
  {
    "model_name": "graph_gnn_reduced_temporal",
    "negative_stride": 30,
    "hidden_dim": 16,
    "epochs": 8,
    "learning_rate": 0.01,
    "train_rows": 2222,
    "calibration_rows": 532,
    "validation_rows": 632,
    "threshold": 0.449618,
    "accuracy": 0.678797,
    "precision": 0.153509,
    "recall": 0.777778,
    "f1": 0.256410,
    "roc_auc": 0.790933,
    "pr_auc": 0.228088,
    "brier_score": 0.187971
  }
]

Compared with the full traditional benchmark, the reduced temporal GNN benchmark achieved much stronger recall and PR-AUC on its reduced real-data graph slice. This makes the GNN path a meaningful project differentiator, even though the current graph benchmark is not yet a one-to-one replacement for the full 16,281-row traditional validation setup.

## State Summary
[
  {
    "state": "Uttarakhand",
    "Positive Windows": 40,
    "Mean Predicted Probability": 0.248,
    "Mean Hazard Score": 0.164,
    "Window Count": 7839
  },
  {
    "state": "Kerala",
    "Positive Windows": 5,
    "Mean Predicted Probability": 0.191,
    "Mean Hazard Score": 0.114,
    "Window Count": 8442
  }
]

## Top Positive Windows
[
  {
    "sample_id": "IN-UTK-UTTARKASHI__2016-07-18",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2016-07-18",
    "predicted_probability": 0.983,
    "label": 1,
    "hazard_score": 0.326,
    "rainfall_24h_mm": 16.84,
    "rainfall_7d_mm": 155.93
  },
  {
    "sample_id": "IN-UTK-UTTARKASHI__2016-07-25",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2016-07-25",
    "predicted_probability": 0.981,
    "label": 0,
    "hazard_score": 0.31,
    "rainfall_24h_mm": 0.58,
    "rainfall_7d_mm": 132.37
  },
  {
    "sample_id": "IN-UTK-UTTARKASHI__2016-07-24",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2016-07-24",
    "predicted_probability": 0.98,
    "label": 0,
    "hazard_score": 0.349,
    "rainfall_24h_mm": 22.21,
    "rainfall_7d_mm": 148.63
  },
  {
    "sample_id": "IN-UTK-UTTARKASHI__2016-07-23",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2016-07-23",
    "predicted_probability": 0.978,
    "label": 0,
    "hazard_score": 0.419,
    "rainfall_24h_mm": 74.76,
    "rainfall_7d_mm": 189.95
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2016-07-23",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2016-07-23",
    "predicted_probability": 0.978,
    "label": 0,
    "hazard_score": 0.402,
    "rainfall_24h_mm": 56.43,
    "rainfall_7d_mm": 147.85
  },
  {
    "sample_id": "IN-UTK-UTTARKASHI__2016-07-17",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2016-07-17",
    "predicted_probability": 0.977,
    "label": 1,
    "hazard_score": 0.371,
    "rainfall_24h_mm": 63.53,
    "rainfall_7d_mm": 140.32
  },
  {
    "sample_id": "IN-RUD-RUDRAPRAYAG__2016-07-18",
    "state": "Uttarakhand",
    "district": "Rudraprayag",
    "date": "2016-07-18",
    "predicted_probability": 0.977,
    "label": 0,
    "hazard_score": 0.308,
    "rainfall_24h_mm": 16.84,
    "rainfall_7d_mm": 155.93
  },
  {
    "sample_id": "IN-UTK-UTTARKASHI__2015-07-12",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2015-07-12",
    "predicted_probability": 0.975,
    "label": 0,
    "hazard_score": 0.317,
    "rainfall_24h_mm": 13.62,
    "rainfall_7d_mm": 156.53
  },
  {
    "sample_id": "IN-RUD-RUDRAPRAYAG__2016-07-25",
    "state": "Uttarakhand",
    "district": "Rudraprayag",
    "date": "2016-07-25",
    "predicted_probability": 0.974,
    "label": 0,
    "hazard_score": 0.292,
    "rainfall_24h_mm": 0.58,
    "rainfall_7d_mm": 132.37
  },
  {
    "sample_id": "IN-PIT-PITHORAGARH__2015-06-27",
    "state": "Uttarakhand",
    "district": "Pithoragarh",
    "date": "2015-06-27",
    "predicted_probability": 0.973,
    "label": 0,
    "hazard_score": 0.332,
    "rainfall_24h_mm": 2.57,
    "rainfall_7d_mm": 141.01
  },
  {
    "sample_id": "IN-UTK-UTTARKASHI__2015-03-03",
    "state": "Uttarakhand",
    "district": "Uttarkashi",
    "date": "2015-03-03",
    "predicted_probability": 0.973,
    "label": 0,
    "hazard_score": 0.29,
    "rainfall_24h_mm": 10.08,
    "rainfall_7d_mm": 114.34
  },
  {
    "sample_id": "IN-RUD-RUDRAPRAYAG__2016-07-24",
    "state": "Uttarakhand",
    "district": "Rudraprayag",
    "date": "2016-07-24",
    "predicted_probability": 0.973,
    "label": 0,
    "hazard_score": 0.331,
    "rainfall_24h_mm": 22.21,
    "rainfall_7d_mm": 148.63
  }
]

## Limitations
- Labels are derived from a mapped external landslide inventory, but some raw locality-level event names may still introduce district-assignment noise.
- The temporal dataset is substantially larger than the original snapshot dataset and is better suited for time-aware evaluation.
- The real-data Kerala/Uttarakhand dataset is highly imbalanced, so accuracy alone can be misleading and recall-oriented metrics should be interpreted carefully.
- External validation can be improved further by refining event-to-district mapping, expanding terrain and vegetation features, and calibrating decision thresholds.
