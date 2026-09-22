# Temporal Model Before vs After

## Real Kerala-Uttarakhand Dataset

### Winning model before improvements
`logistic_regression_balanced`

| Metric | Previous |
| --- | ---: |
| Accuracy | 0.9919 |
| Precision | 0.0693 |
| Recall | 0.1556 |
| F1 | 0.0959 |
| ROC-AUC | 0.8547 |
| PR-AUC | 0.0710 |

### Winning model after improvements
`logistic_regression_balanced`

| Metric | Updated |
| --- | ---: |
| Accuracy | 0.9928 |
| Precision | 0.0805 |
| Recall | 0.1556 |
| F1 | 0.1061 |
| ROC-AUC | 0.8465 |
| PR-AUC | 0.0508 |
| Brier score | 0.0857 |

### Delta
| Metric | Change |
| --- | ---: |
| Accuracy | +0.0009 |
| Precision | +0.0112 |
| Recall | +0.0000 |
| F1 | +0.0102 |
| ROC-AUC | -0.0083 |
| PR-AUC | -0.0202 |

## Updated Candidate Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (balanced) | 0.9928 | 0.0805 | 0.1556 | 0.1061 | 0.8465 | 0.0508 | 0.0857 |
| Logistic regression (balanced, calibrated) | 0.9910 | 0.0678 | 0.1778 | 0.0982 | 0.8465 | 0.0508 | 0.0027 |
| Gradient boosting | 0.9950 | 0.0909 | 0.0889 | 0.0899 | 0.8318 | 0.0504 | 0.0030 |
| Gradient boosting (calibrated) | 0.9541 | 0.0282 | 0.4667 | 0.0532 | 0.8196 | 0.0240 | 0.0027 |
| GRU sequence classifier | 0.5155 | 0.0051 | 0.8889 | 0.0101 | 0.7109 | 0.0050 | 0.2850 |
| LSTM sequence classifier | 0.5224 | 0.0050 | 0.8667 | 0.0100 | 0.7317 | 0.0307 | 0.2089 |

## Traditional vs GNN

| Approach | Benchmark scope | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Traditional best model: Logistic regression (balanced) | Full real temporal validation (`16,281` rows) | 0.9928 | 0.0805 | 0.1556 | 0.1061 | 0.8465 | 0.0508 |
| GNN: Reduced temporal graph benchmark | Real-data reduced graph validation (`632` rows) | 0.6788 | 0.1535 | 0.7778 | 0.2564 | 0.7909 | 0.2281 |

Note: the current GNN benchmark uses a reduced temporal graph slice that keeps all positive windows and a structured sample of negatives so the graph remains computationally tractable on real data. It is therefore a genuine real-data benchmark, but not yet a one-to-one replacement for the full traditional validation setup.

## NASA GLC Merge Update

The merged inventory combines the project inventory with mapped NASA Global Landslide Catalog events for Kerala and Uttarakhand.

| Inventory view | Unique district-date events | Positive windows | Positive rate |
| --- | ---: | ---: | ---: |
| Original mapped inventory | 151 | 562 | 0.0069 |
| Merged project + NASA GLC inventory | 211 | 893 | 0.0110 |

### Merged-label traditional benchmark

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (balanced) | 0.9808 | 0.0914 | 0.1184 | 0.1032 | 0.8674 | 0.0662 | 0.0882 |
| Logistic regression (balanced, calibrated) | 0.9851 | 0.1186 | 0.0921 | 0.1037 | 0.8674 | 0.0662 | 0.0091 |
| Gradient boosting | 0.9893 | 0.0769 | 0.0132 | 0.0225 | 0.8603 | 0.0752 | 0.0091 |

### Merged-label traditional vs GNN

| Approach | Benchmark scope | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Traditional best model: Logistic regression (balanced) | Full merged-label temporal validation (`16,281` rows) | 0.9808 | 0.0914 | 0.1184 | 0.1032 | 0.8674 | 0.0662 |
| GNN: Reduced temporal graph benchmark | Merged-label reduced graph validation (`733` rows) | 0.2074 | 0.2074 | 1.0000 | 0.3435 | 0.2416 | 0.1456 |

This merged-label comparison is the most honest publication-ready story in the repo right now. The traditional logistic model remains the strongest full-scale ranking model, while the GNN becomes much more aggressive and recovers all positives on the reduced graph slice, which lifts recall, F1, and PR-AUC but still leaves ROC-AUC and overall calibration in need of improvement.

## Ready-to-say Interpretation

The updated temporal pipeline improved thresholded classification performance for the main logistic model, especially precision and F1, but it did not improve ranking quality on PR-AUC. The first GRU and LSTM sequence-model benchmarks did not beat the logistic baseline on the real regional dataset. In contrast, the reduced temporal GNN benchmark produced much stronger recall and a much higher PR-AUC than the traditional full-dataset baseline, which makes the graph path a credible differentiator for the project even though the benchmark scope is currently reduced for computational reasons.
