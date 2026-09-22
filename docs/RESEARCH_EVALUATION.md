# Research Evaluation Summary

## Dataset Summary
- Rows: 59
- States: 15
- Districts: 59
- Positive labels: 24
- Negative labels: 35

## Experimental Setup
- Dataset: `data/india/india_regions.csv`
- Validation split: stratified holdout (0.25)
- Trained model: gradient_boosting
- Metrics: accuracy, precision, recall, f1, roc_auc

## Model Comparisons
| Model | Validation Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule_based_baseline | 59 | 0.831 | 1.000 | 0.583 | 0.737 | 0.941 |
| trained_risk_with_graph | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| trained_risk_without_graph | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Scenario Examples
### Kerala heavy rainfall stress test
- Target scope: Kerala
- Mean risk delta: 0.001
- Change in high-risk districts: 0
- Top movers: [
  {
    "district": "Alappuzha",
    "state": "Kerala",
    "risk_delta": 0.003,
    "forecast_risk": 0.233,
    "primary_driver": "Propagated Graph"
  },
  {
    "district": "Kottayam",
    "state": "Kerala",
    "risk_delta": 0.003,
    "forecast_risk": 0.239,
    "primary_driver": "Propagated Graph"
  },
  {
    "district": "Ernakulam",
    "state": "Kerala",
    "risk_delta": 0.003,
    "forecast_risk": 0.23,
    "primary_driver": "Propagated Graph"
  },
  {
    "district": "Kasaragod",
    "state": "Kerala",
    "risk_delta": 0.003,
    "forecast_risk": 0.238,
    "primary_driver": "Propagated Graph"
  },
  {
    "district": "Wayanad",
    "state": "Kerala",
    "risk_delta": 0.003,
    "forecast_risk": 0.383,
    "primary_driver": "Terrain"
  }
]

### Himalayan slope saturation test
- Target scope: Uttarakhand
- Mean risk delta: 0.003
- Change in high-risk districts: 0
- Top movers: [
  {
    "district": "Haridwar",
    "state": "Uttarakhand",
    "risk_delta": 0.028,
    "forecast_risk": 0.324,
    "primary_driver": "Soil Wetness"
  },
  {
    "district": "Udham Singh Nagar",
    "state": "Uttarakhand",
    "risk_delta": 0.01,
    "forecast_risk": 0.328,
    "primary_driver": "Soil Wetness"
  },
  {
    "district": "Bageshwar",
    "state": "Uttarakhand",
    "risk_delta": 0.01,
    "forecast_risk": 0.437,
    "primary_driver": "Soil Wetness"
  },
  {
    "district": "Chamoli",
    "state": "Uttarakhand",
    "risk_delta": 0.01,
    "forecast_risk": 0.477,
    "primary_driver": "Terrain"
  },
  {
    "district": "Rudraprayag",
    "state": "Uttarakhand",
    "risk_delta": 0.01,
    "forecast_risk": 0.468,
    "primary_driver": "Terrain"
  }
]

## State Analysis
[
  {
    "State": "Himachal Pradesh",
    "Risk Score": 0.588,
    "Forecast Risk": 0.456,
    "Uncertainty": 0.085,
    "Region Count": 5
  },
  {
    "State": "Uttarakhand",
    "Risk Score": 0.519,
    "Forecast Risk": 0.41,
    "Uncertainty": 0.103,
    "Region Count": 13
  },
  {
    "State": "Sikkim",
    "Risk Score": 0.447,
    "Forecast Risk": 0.384,
    "Uncertainty": 0.182,
    "Region Count": 3
  },
  {
    "State": "Karnataka",
    "Risk Score": 0.443,
    "Forecast Risk": 0.38,
    "Uncertainty": 0.179,
    "Region Count": 2
  },
  {
    "State": "Arunachal Pradesh",
    "Risk Score": 0.427,
    "Forecast Risk": 0.37,
    "Uncertainty": 0.185,
    "Region Count": 3
  },
  {
    "State": "Mizoram",
    "Risk Score": 0.409,
    "Forecast Risk": 0.357,
    "Uncertainty": 0.187,
    "Region Count": 3
  },
  {
    "State": "Tamil Nadu",
    "Risk Score": 0.397,
    "Forecast Risk": 0.349,
    "Uncertainty": 0.177,
    "Region Count": 2
  },
  {
    "State": "Manipur",
    "Risk Score": 0.393,
    "Forecast Risk": 0.347,
    "Uncertainty": 0.188,
    "Region Count": 1
  }
]

## Top District Hotspots
[
  {
    "Region ID": "IN-PIT-PITHORAGARH",
    "State": "Uttarakhand",
    "District": "Pithoragarh",
    "Risk Score": 0.614,
    "Forecast Risk": 0.473,
    "Uncertainty": 0.067,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-KUL-KULLU",
    "State": "Himachal Pradesh",
    "District": "Kullu",
    "Risk Score": 0.612,
    "Forecast Risk": 0.472,
    "Uncertainty": 0.074,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-SHI-SHIMLA",
    "State": "Himachal Pradesh",
    "District": "Shimla",
    "Risk Score": 0.609,
    "Forecast Risk": 0.471,
    "Uncertainty": 0.086,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-KIN-KINNAUR",
    "State": "Himachal Pradesh",
    "District": "Kinnaur",
    "Risk Score": 0.608,
    "Forecast Risk": 0.47,
    "Uncertainty": 0.088,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-CHA-CHAMOLI",
    "State": "Uttarakhand",
    "District": "Chamoli",
    "Risk Score": 0.603,
    "Forecast Risk": 0.468,
    "Uncertainty": 0.107,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-UTK-UTTARKASHI",
    "State": "Uttarakhand",
    "District": "Uttarkashi",
    "Risk Score": 0.602,
    "Forecast Risk": 0.467,
    "Uncertainty": 0.111,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-RUD-RUDRAPRAYAG",
    "State": "Uttarakhand",
    "District": "Rudraprayag",
    "Risk Score": 0.59,
    "Forecast Risk": 0.459,
    "Uncertainty": 0.113,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-CHM-CHAMBA",
    "State": "Himachal Pradesh",
    "District": "Chamba",
    "Risk Score": 0.584,
    "Forecast Risk": 0.454,
    "Uncertainty": 0.101,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-TEH-TEHRIGARHWAL",
    "State": "Uttarakhand",
    "District": "Tehri Garhwal",
    "Risk Score": 0.583,
    "Forecast Risk": 0.453,
    "Uncertainty": 0.113,
    "Primary Driver": "Terrain"
  },
  {
    "Region ID": "IN-NAI-NAINITAL",
    "State": "Uttarakhand",
    "District": "Nainital",
    "Risk Score": 0.559,
    "Forecast Risk": 0.436,
    "Uncertainty": 0.08,
    "Primary Driver": "Terrain"
  }
]

## Limitations
- The district dataset is modest in size and partially seed-generated, so results should be treated as research-oriented rather than production-validated.
- The current labels are proxies for landslide susceptibility and do not represent a complete historical landslide inventory.
- Forecasting and graph artifacts depend on local environment compatibility and should be retrained in the target runtime before deployment claims are made.
- The dashboard is suitable for comparative analysis and scenario exploration, but not yet for official disaster warning operations.
