# NASA GLC Merged Real-Data Update

## Inventory merge summary

- Existing mapped project inventory: `151` district-date events
- NASA Global Landslide Catalog events mapped into Kerala/Uttarakhand: `239`
- Overlapping NASA district-date events already covered: `112`
- New unique district-date events added from NASA GLC: `85`
- Final merged inventory size: `211` district-date events

## Label impact on the temporal dataset

| Dataset | Rows | Positive windows | Positive rate |
| --- | ---: | ---: | ---: |
| Original Kerala-Uttarakhand real dataset | 81,405 | 562 | 0.0069 |
| Merged-inventory Kerala-Uttarakhand real dataset | 81,405 | 893 | 0.0110 |

Validation split:
- Validation rows: `16,281`
- Validation positives after merge: `152`

## Merged-label traditional benchmark

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (balanced) | 0.9808 | 0.0914 | 0.1184 | 0.1032 | 0.8674 | 0.0662 | 0.0882 |
| Logistic regression (balanced, calibrated) | 0.9851 | 0.1186 | 0.0921 | 0.1037 | 0.8674 | 0.0662 | 0.0091 |
| Gradient boosting | 0.9893 | 0.0769 | 0.0132 | 0.0225 | 0.8603 | 0.0752 | 0.0091 |

## Merged-label GNN benchmark

Best completed reduced temporal GNN run:
- Dataset: `data/regional/kerala_uttarakhand_risk_timeseries_merged_inventory.csv`
- Negative stride: `30`
- Hidden dimension: `16`
- Epochs: `8`
- Learning rate: `0.01`
- Validation rows: `733`

Metrics:
- Accuracy: `0.2074`
- Precision: `0.2074`
- Recall: `1.0000`
- F1: `0.3435`
- ROC-AUC: `0.2416`
- PR-AUC: `0.1456`
- Brier score: `0.2101`

## Honest interpretation

The merged NASA-backed inventory improves the realism of the benchmark and helps the best traditional logistic model on ranking quality, especially ROC-AUC and PR-AUC. The reduced temporal GNN remains the project's main differentiator because it recovers far more positives and delivers higher F1 and PR-AUC on the graph slice, but it still needs better ranking quality and calibration before it can be presented as uniformly better than the traditional baseline.
