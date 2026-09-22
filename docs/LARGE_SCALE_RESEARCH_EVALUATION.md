# Large-Scale Risk Evaluation

## Dataset
- Total district-date rows: 81405
- Train rows: 65124
- Validation rows: 16281
- Label source: inventory
- Model: graph_gnn
- Decision threshold: 0.605
- Feature count: 24

## Temporal Evaluation Metrics
- Accuracy: 0.782
- Precision: 0.188
- Recall: 0.276
- F1: 0.224
- ROC-AUC: 0.737

- PR-AUC: 0.226
- Brier score: 0.217

## Confusion Matrix
- True negative: 1000
- False positive: 181
- False negative: 110
- True positive: 42

## Model Comparison
[
  {
    "model_name": "logistic_regression_balanced",
    "threshold": 0.9155681979899434,
    "accuracy": 0.9829248817640194,
    "precision": 0.09615384615384616,
    "recall": 0.09868421052631579,
    "f1": 0.09740259740259741,
    "roc_auc": 0.8563877259333466,
    "pr_auc": 0.060458769432339056,
    "brier_score": 0.08826617897413293
  },
  {
    "model_name": "logistic_regression_balanced_calibrated",
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
    "model_name": "gradient_boosting",
    "threshold": 0.20221856799166915,
    "accuracy": 0.9880228487193662,
    "precision": 0.12280701754385964,
    "recall": 0.046052631578947366,
    "f1": 0.06698564593301436,
    "roc_auc": 0.8617327892550523,
    "pr_auc": 0.0732631575272734,
    "brier_score": 0.009391169896699754
  },
  {
    "model_name": "gradient_boosting_calibrated",
    "threshold": 0.04778156996587031,
    "accuracy": 0.9590934217799889,
    "precision": 0.08945686900958466,
    "recall": 0.3684210526315789,
    "f1": 0.14395886889460155,
    "roc_auc": 0.8490696718235542,
    "pr_auc": 0.06748302122738328,
    "brier_score": 0.008973906410597463
  },
  {
    "model_name": "graph_gnn",
    "threshold": 0.6054872274398804,
    "accuracy": 0.781695423855964,
    "precision": 0.18834080717488788,
    "recall": 0.27631578947368424,
    "f1": 0.224,
    "roc_auc": 0.7371819154151256,
    "pr_auc": 0.22573322499515738,
    "brier_score": 0.2173569933138401
  },
  {
    "model_name": "gru_sequence_classifier",
    "threshold": 0.890505313873291,
    "accuracy": 0.9849731886768924,
    "precision": 0.06796116504854369,
    "recall": 0.046052631578947366,
    "f1": 0.054901960784313725,
    "roc_auc": 0.751477633401141,
    "pr_auc": 0.03131887698464458,
    "brier_score": 0.1703709129498566
  },
  {
    "model_name": "lstm_sequence_classifier",
    "threshold": 0.8977692723274231,
    "accuracy": 0.9850978925052999,
    "precision": 0.10810810810810811,
    "recall": 0.07894736842105263,
    "f1": 0.09125475285171103,
    "roc_auc": 0.8538733210970268,
    "pr_auc": 0.059128399880782896,
    "brier_score": 0.18355860907045
  },
  {
    "model_name": "temporal_cnn_classifier",
    "threshold": 0.9393261671066284,
    "accuracy": 0.9871555056740242,
    "precision": 0.10294117647058823,
    "recall": 0.046052631578947366,
    "f1": 0.06363636363636363,
    "roc_auc": 0.8385358342665173,
    "pr_auc": 0.05848753634900873,
    "brier_score": 0.09748347758629393
  },
  {
    "model_name": "temporal_transformer_classifier",
    "threshold": 0.8490917682647705,
    "accuracy": 0.9755580496321237,
    "precision": 0.10526315789473684,
    "recall": 0.21052631578947367,
    "f1": 0.14035087719298245,
    "roc_auc": 0.8916548500168967,
    "pr_auc": 0.08354619530416044,
    "brier_score": 0.10990438530980323
  },
  {
    "model_name": "hist_gradient_boosting_cost_sensitive",
    "threshold": 0.9043752891670765,
    "accuracy": 0.9867330016583747,
    "precision": 0.15217391304347827,
    "recall": 0.09210526315789473,
    "f1": 0.11475409836065574,
    "roc_auc": 0.8504328587604544,
    "pr_auc": 0.08450714570038234,
    "brier_score": 0.052530648522713566
  },
  {
    "model_name": "logistic_regression_smote",
    "threshold": 0.06818961417991322,
    "accuracy": 0.986855844235612,
    "precision": 0.12195121951219512,
    "recall": 0.06578947368421052,
    "f1": 0.08547008547008547,
    "roc_auc": 0.8566071737406632,
    "pr_auc": 0.057936780050893276,
    "brier_score": 0.009071971368904856
  },
  {
    "model_name": "logistic_regression_random_oversample",
    "threshold": 0.07613954669858157,
    "accuracy": 0.9876543209876543,
    "precision": 0.14492753623188406,
    "recall": 0.06578947368421052,
    "f1": 0.09049773755656108,
    "roc_auc": 0.8562139624279249,
    "pr_auc": 0.06032822655126997,
    "brier_score": 0.009065601316700138
  },
  {
    "model_name": "xgboost_cost_sensitive",
    "threshold": 0.8619564175605774,
    "accuracy": 0.9863030526380443,
    "precision": 0.1414141414141414,
    "recall": 0.09210526315789473,
    "f1": 0.11155378486055777,
    "roc_auc": 0.8446652156462208,
    "pr_auc": 0.0834202922798732,
    "brier_score": 0.03506789708139023
  },
  {
    "model_name": "lightgbm_cost_sensitive",
    "threshold": 0.8978798943591961,
    "accuracy": 0.989067010625883,
    "precision": 0.19047619047619047,
    "recall": 0.05263157894736842,
    "f1": 0.08247422680412371,
    "roc_auc": 0.8038666866807418,
    "pr_auc": 0.0801670463594172,
    "brier_score": 0.0234053285988442
  }
]

## State Summary
[
  {
    "state": "Uttarakhand",
    "Positive Windows": 142,
    "Mean Predicted Probability": 0.597,
    "Mean Hazard Score": 0.178,
    "Window Count": 707
  },
  {
    "state": "Kerala",
    "Positive Windows": 10,
    "Mean Predicted Probability": 0.314,
    "Mean Hazard Score": 0.115,
    "Window Count": 626
  }
]

## Top Positive Windows
[
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-12-28",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-12-28",
    "predicted_probability": 0.65,
    "label": 0,
    "hazard_score": 0.217,
    "rainfall_24h_mm": 0.0,
    "rainfall_7d_mm": 0.0
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-11-16",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-11-16",
    "predicted_probability": 0.65,
    "label": 0,
    "hazard_score": 0.217,
    "rainfall_24h_mm": 0.0,
    "rainfall_7d_mm": 0.0
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-11-30",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-11-30",
    "predicted_probability": 0.65,
    "label": 0,
    "hazard_score": 0.217,
    "rainfall_24h_mm": 0.0,
    "rainfall_7d_mm": 1.1
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-11-02",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-11-02",
    "predicted_probability": 0.65,
    "label": 0,
    "hazard_score": 0.217,
    "rainfall_24h_mm": 0.0,
    "rainfall_7d_mm": 0.98
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2016-03-07",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2016-03-07",
    "predicted_probability": 0.65,
    "label": 0,
    "hazard_score": 0.218,
    "rainfall_24h_mm": 0.07,
    "rainfall_7d_mm": 1.78
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-02-15",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-02-15",
    "predicted_probability": 0.65,
    "label": 0,
    "hazard_score": 0.218,
    "rainfall_24h_mm": 0.03,
    "rainfall_7d_mm": 0.03
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-03-29",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-03-29",
    "predicted_probability": 0.65,
    "label": 1,
    "hazard_score": 0.227,
    "rainfall_24h_mm": 4.16,
    "rainfall_7d_mm": 4.67
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2016-03-21",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2016-03-21",
    "predicted_probability": 0.649,
    "label": 0,
    "hazard_score": 0.226,
    "rainfall_24h_mm": 0.01,
    "rainfall_7d_mm": 5.33
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2016-02-22",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2016-02-22",
    "predicted_probability": 0.649,
    "label": 0,
    "hazard_score": 0.226,
    "rainfall_24h_mm": 0.0,
    "rainfall_7d_mm": 8.57
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2016-01-11",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2016-01-11",
    "predicted_probability": 0.649,
    "label": 0,
    "hazard_score": 0.217,
    "rainfall_24h_mm": 0.0,
    "rainfall_7d_mm": 0.44
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-04-02",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-04-02",
    "predicted_probability": 0.649,
    "label": 1,
    "hazard_score": 0.231,
    "rainfall_24h_mm": 1.06,
    "rainfall_7d_mm": 26.52
  },
  {
    "sample_id": "IN-CHA-CHAMOLI__2015-04-01",
    "state": "Uttarakhand",
    "district": "Chamoli",
    "date": "2015-04-01",
    "predicted_probability": 0.649,
    "label": 1,
    "hazard_score": 0.241,
    "rainfall_24h_mm": 1.67,
    "rainfall_7d_mm": 25.96
  }
]

## Limitations
- Labels are derived from a mapped external landslide inventory, but some raw locality-level event names may still introduce district-assignment noise.
- The temporal dataset is substantially larger than the original snapshot dataset and is better suited for time-aware evaluation.
- The real-data Kerala/Uttarakhand dataset is highly imbalanced, so accuracy alone can be misleading and recall-oriented metrics should be interpreted carefully.
- External validation can be improved further by refining event-to-district mapping, expanding terrain and vegetation features, and calibrating decision thresholds.
