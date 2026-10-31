# Sampling and OOB Predictions

PFGAP supports three whole-tree sampling strategies and optional fit-time out-of-bag aggregation. Sampling determines which observations fit each tree. OOB routing determines whether non-sampled observations also traverse the fitted tree for requested downstream work. These are related, but they are not the same switch.

## Sampling strategies

| Python value | Java value | Behavior |
|---|---|---|
| `full` | `FULL` | Use every training observation exactly once. |
| `bootstrap` | `BOOTSTRAP` | Draw with replacement. |
| `subsample` | `SUBSAMPLE` | Draw distinct observations without replacement. |

The default is `bootstrap`.

## Sample size

`sample_size=None` uses the training-set size. A positive integer is an exact count. A finite float in `(0, 1]` is a fraction, rounded up with a minimum of one observation. Integer and fractional forms remain distinct: `1` means one observation, while `1.0` means the full dataset.

`sampling="full"` requires `sample_size=None`. A subsample cannot exceed the population. A bootstrap count may exceed it because sampling is with replacement.

```python
sampling="full"; sample_size=None
sampling="bootstrap"; sample_size=None
sampling="subsample"; sample_size=0.8
sampling="subsample"; sample_size=256
```

Direct Java uses `-sampling=...` and `-sample_size=...`.

## Fit-time OOB aggregation

```python
oob_scores=True
return_oob_predictions=True
```

`oob_scores=True` requests aggregation from trees for which an observation was out of bag. It requires bootstrap or subsample sampling. `return_oob_predictions=True` writes `training_oob_predictions.csv` and requires OOB scoring.

The artifact supports classification and regression, including multi-output classification and multi-target regression. It is not supported for isolation forests. An instance with zero contributing OOB trees remains in the file with its original `instance_index` and actual value, but blank prediction-derived fields.

## OOB routing used by other work

Depending on the selected proximity definition, routing can also be required for requested training proximities, training imputation, test imputation, or supervised classification training outlier scores. This can occur with `oob_scores=False` and does not create `training_oob_predictions.csv`.

Sampling alone does not require routing. If no requested task needs non-sampled observations, PFGAP can fit the sample without recursively routing its complement.

## Isolation configuration

```python
forest_mode="isolation"
sampling="subsample"
sample_size=256
oob_scores=False
return_oob_predictions=False
```

This avoids OOB routing only when no other requested workflow needs it. The value 256 is a conventional starting point, not a required setting.

## Output naming

The base filename is `training_oob_predictions.csv`. Multiple repetitions use the ordinary suffix policy, for example `training_oob_predictions_repeat_2.csv`. Classification and regression use their ordinary prediction schemas; multi-output columns use positional suffixes.

## Related documentation

- [Classification](Classification.md)
- [Regression](Regression.md)
- [Imputation](Imputation.md)
- [Outlier Scoring](Outlier_Scoring.md)
- [Configuration Reference](../reference/Configuration_Reference.md)
- [CLI Reference](../reference/CLI_Reference.md)
- [Outputs](../reference/Outputs.md)
