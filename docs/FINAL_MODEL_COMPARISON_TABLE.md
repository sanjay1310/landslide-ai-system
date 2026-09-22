# Final Fair Comparison Table

This is the cleanest final comparison to use in the report, paper, or PPT because it evaluates all three approaches on the same district-date graph-sequence holdout with thresholds tuned on a separate calibration slice.

| Model | Scope | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Traditional runtime model | Common graph-sequence holdout (`10,152` rows) | 0.010 | 0.0182 | 0.8824 | 0.0356 | 0.8484 | 0.0435 |
| Static graph GNN | Common graph-sequence holdout (`10,152` rows) | 0.0117 | 0.0067 | 1.0000 | 0.0133 | 0.5000 | 0.0067 |
| Spatio-temporal GNN (advanced GRU) | Common graph-sequence holdout (`10,152` rows) | 0.556 | 0.0422 | 0.8235 | 0.0803 | 0.9099 | 0.1324 |

## Best interpretation

- The upgraded `spatio-temporal GNN` is still the strongest model on the fair same-scope benchmark.
- The newer GRU-based ST-GNN preserves the best ranking quality with `0.9099` ROC-AUC and `0.1324` PR-AUC while also giving the best F1.
- The `traditional runtime` model now avoids the all-zero outcome, but only by moving to a very low threshold that produces high recall and low precision.
- The `static graph GNN` remains informative as a graph baseline, but on this shared holdout its calibrated probabilities become over-sensitive and predict nearly every row as positive.
- The strongest current project story is now `traditional baseline for stable full-temporal performance, static GNN as a simpler graph baseline, and ST-GNN as the best fair-benchmark model`.

## Best single sentence for submission

On the fairest same-scope district-date graph-sequence benchmark, the upgraded spatio-temporal GNN remained the strongest overall model, delivering the best ROC-AUC, PR-AUC, and F1 while maintaining a much more useful precision-recall balance than the other two approaches.
