# Outputs

This reference describes the files PFGAP can write during training, validation, saved-model evaluation, imputation, scoring, and experiment reporting. It defines current filenames, schemas, indexing, repetition-aware naming, and output-request relationships.

For configuration names, see [Configuration Reference](Configuration_Reference.md). For complete imputed-only semantics, see [Imputed-Only Output](Imputed_Only_Output.md).

## Output directory

Set the output directory in Python with:

```python
output_directory="../output/run_01"
```

Direct Java form:

```text
-out=output/run_01/
```

Artifact paths are normalized beneath the configured output directory. Writers create parent directories where supported. Existing CSV, JSON, and Matrix Market targets are replaced unless a specific writer explicitly uses append mode.

## Output requests are independent

PFGAP treats these as independent requests:

- ordinary aggregate predictions;
- enhanced per-instance prediction details;
- OOD scores;
- proximities;
- supervised training outlier scores;
- isolation scores;
- complete imputed datasets;
- imputed-only values;
- saved models; and
- standardization statistics.

For example, OOD output does not require ordinary predictions or enhanced prediction details. The structured evaluation path is used whenever enhanced prediction output or OOD output is requested.

## Repetition-aware naming

For a single repetition, PFGAP uses the base artifact filename.

For multiple repetitions, it inserts a one-based repetition suffix before the extension:

```text
test_predictions.csv
test_predictions_repeat_2.csv
```

An extensionless artifact uses the same suffix rule:

```text
model
model_repeat_2
```

The internal repetition index is zero-based, but artifact suffixes and result-record repetition numbers are one-based.

## Ordinary predictions

Request ordinary predictions with:

```python
return_predictions=True
```

Direct Java form:

```text
-get_predictions=true
```

## Prediction filenames

Training-time validation or test predictions:

```text
validation_predictions.csv
```

Saved-model evaluation predictions:

```text
test_predictions.csv
```

A multi-repetition run adds `_repeat_N` before `.csv`.

## Classification predictions

Scalar classification without actual labels:instance\_index,prediction  
Scalar classification with actual labels:instance\_index,prediction,actual,correct
- instance\_index is zero-based.
- prediction and actual preserve the integer or string class labels used by the dataset.
- correct records whether the prediction equals the actual label.

Multi-output classification expands the prediction and actual columns by output index. Without actual labels:instance\_index,prediction\_0,prediction\_1  
With actual labels:instance\_index,prediction\_0,actual\_0,prediction\_1,actual\_1,correct  
Additional outputs continue the same numbered pattern. Output indices are zero-based and follow the configured label-column or sidecar-column order. The final correct value uses exact-match semantics: it is true only when every predicted component equals the corresponding actual component.

The actual-label columns are omitted when no actual-label collection is supplied, the collection is empty, or every value is null. If a particular actual label is null while the columns are present, its actual fields and correct are blank for that row.

## Regression predictions

Scalar regression without actual targets:instance\_index,prediction  
Scalar regression with actual targets:instance\_index,prediction,actual,residual,absolute\_error  
Regression residuals use:residual = prediction - actual  
Absolute error is:absolute\_error = abs(residual)

Multi-target regression expands these fields by target index. Without actual targets:instance\_index,prediction\_0,prediction\_1  
With actual targets:instance\_index,prediction\_0,actual\_0,residual\_0,absolute\_error\_0,prediction\_1,actual\_1,residual\_1,absolute\_error\_1  
Additional targets continue the same numbered pattern. Target indices are zero-based and follow the configured target-column or sidecar-column order. Residual and absolute-error calculations are performed independently for each target.

If a particular actual target is null while actual-target columns are present, the corresponding actual, residual, and absolute-error fields are blank for that row.

## Fit-time OOB predictions

With both `oob_scores` and `return_oob_predictions` enabled during classification or regression training, PFGAP writes `training_oob_predictions.csv`. It uses the ordinary prediction schema, including positional columns for multi-output tasks. Instances with no contributing OOB tree retain their zero-based index and actual value, with blank prediction-derived fields.

The file is not written merely because bootstrap or subsample sampling is used, or when routing is needed only for proximity, imputation, or supervised outlier work. Repetition suffixes follow the standard artifact policy.

## Structured evaluation output

Structured output is written when enhanced prediction details, OOD output, or both are requested. Ordinary prediction CSVs support scalar and multi-output tasks. Enhanced prediction details remain scalar-only in v1; OOD-only structured output remains valid independently.

Request enhanced prediction details with:

```python
return_enhanced_outputs=True
```

Request OOD output with:

```python
return_ood_scores=True
```

Direct Java forms:

```text
-return_enhanced_outputs=true
-return_ood_scores=true
```

## Structured filenames

Training-time validation or test data:

```text
validation_enhanced.csv
```

Saved-model evaluation data:

```text
test_enhanced.csv
```

## Structured schema

The CSV columns are:

```text
instance_index
prediction_kind
prediction
prediction_tree_count
prediction_mean
prediction_standard_deviation
class_vote_probabilities
ood_score_type
ood_mean
ood_standard_deviation
ood_available_tree_count
ood_total_tree_count
```

Inapplicable fields are written as empty CSV fields.

### Common field

- `instance_index`: zero-based position in evaluation-instance order.

### Classification prediction fields

- `prediction_kind`: identifies the structured prediction type.
- `prediction`: aggregate integer class prediction.
- `prediction_tree_count`: number of tree predictions contributing to the aggregate result.
- `class_vote_probabilities`: class vote proportions encoded inside one CSV field as a semicolon-delimited map.

A vote-proportion field can contain delimiters of its own, so normal CSV quoting applies when required.

### Regression prediction fields

- `prediction_kind`: identifies the structured prediction type.
- `prediction`: aggregate numeric prediction.
- `prediction_tree_count`: number of contributing tree predictions.
- `prediction_mean`: mean of the contributing tree predictions.
- `prediction_standard_deviation`: population dispersion of the tree predictions reported by the structured result.

### OOD fields

- `ood_score_type`: current scorer identifier, such as `relative_support_exceedance`.
- `ood_mean`: mean bounded OOD score across contributing trees. For `relative_support_exceedance`, available tree scores lie in `[0, 1]`; higher values indicate stronger exceedance and are not calibrated probabilities.
- `ood_standard_deviation`: standard deviation across available tree-level OOD scores.
- `ood_available_tree_count`: number of trees that produced an available OOD score.
- `ood_total_tree_count`: total number of trees considered.

Relative support exceedance requires finite, nonnegative winning distances. Disable `early_abandon_distances` when diagnosing non-finite behavior.

OOD-only output is valid. Prediction fields may be empty when no enhanced prediction details were requested.

## Read structured output in Python

The Python helper provides:

```python
import PF_wrapper as PF

rows = PF.read_enhanced_output(
    "../output/run_01/test_enhanced.csv"
)
```

Each row is returned as a dictionary using the CSV column names.

## Outlier and isolation scores

PFGAP writes supervised classification outlier scores and unsupervised isolation scores through the score-writer layer, but they remain different computations.

## Score filenames

Training observations:

```text
training_outlier_scores.csv
```

Training-time validation or test observations:

```text
validation_outlier_scores.csv
```

Saved-model evaluation observations:

```text
test_outlier_scores.csv
```

## Basic score schema

Without labels or diagnostics:

```text
instance_index,outlier_score
```

## Score schema with labels

```text
instance_index,outlier_score,label
```

The label column is included only when the supplied label collection contains at least one non-null value. This score-file schema is a single label field; it is separate from the numbered multi-output prediction schema described above.

## Score schema with diagnostics

A caller can add one diagnostic column, such as `mean_path_length`:

```text
instance_index,outlier_score,mean_path_length
```

With labels and diagnostics:

```text
instance_index,outlier_score,label,mean_path_length
```

Null boxed score values are written as blank fields.

## Supervised classification outlier output

Request with:

```python
return_training_outlier_scores=True
```

Direct Java form:

```text
-get_training_outlier_scores=true
```

This output describes within-class unusualness among classification training observations and can include the scalar class label.

## Isolation output

Isolation-mode scores are produced by `forest_mode="isolation"`. Isolation output can include a diagnostic such as mean path length.

See [Outlier Scoring](../guides/Outlier_Scoring.md).

## Proximity matrices

Request proximity output with:

```python
return_proximities=True
```

Direct Java form:

```text
-getprox=true
```

The selected `proximity_type` controls the proximity definition. Output storage can be dense CSV or sparse Matrix Market according to the computed representation.

## Proximity filenames

Training-to-training proximities:

```text
training_proximities.csv
training_proximities.mtx
```

Test-to-training or validation-to-training proximities:

```text
test_train_proximities.csv
test_train_proximities.mtx
```

Only the file matching the active dense or sparse representation is written.

## Dense proximity CSV

The first header field is:

```text
instance_index
```

Remaining header fields are zero-based matrix column indices. Each data row contains:

1. the zero-based matrix row index; and
2. every numeric value in that matrix row.

Conceptually:

```csv
instance_index,0,1,2
0,1.0,0.4,0.0
1,0.4,1.0,0.2
2,0.0,0.2,1.0
```

## Sparse proximity Matrix Market

Sparse proximities use Matrix Market coordinate format:

```text
%%MatrixMarket matrix coordinate real general
```

The shape line is:

```text
row_count column_count entry_count
```

Entries use one-based Matrix Market coordinates:

```text
row column value
```

PFGAP internal row and column indices are zero-based and are converted during writing.

General storage is the default because a proximity matrix may be rectangular or asymmetric. Symmetric storage is used only when a square matrix and its reflected values have been validated exactly.

Sparse proximity storage rejects retained exact-zero entries. Unstored positions represent zero.

## Complete imputed datasets

Request complete imputed output with:

```python
return_imputed_training=True
return_imputed_testing=True
```

Direct Java forms:

```text
-impute_train=true
-impute_test=true
```

Complete imputed datasets contain both originally observed and final imputed values. PFGAP selects an output writer compatible with the source reader type and physical data family.

Exact filenames and layouts therefore depend on the selected reader/writer pair. See [Writers](../data/Writers.md).

## Imputed-only coordinate output

Request only final values at originally missing coordinates with:

```python
return_imputed_training_csr=True
return_imputed_testing_csr=True
```

The option names retain `csr` for compatibility, but the format depends on logical data rank.

Default base filenames are:

```text
training_imputed_values
testing_imputed_values
```

PFGAP replaces the final extension when writing:

- tabular or univariate data uses Matrix Market coordinate `.mtx`, with `(instance, feature-or-time, value)` entries;
- multivariate data uses sparse tensor coordinate `.tns`, with `(instance, dimension, time, value)` entries.

Multivariate output bypasses CSR and does not flatten dimension and time into an offset column. The `.tns` shape comment records the maximum envelope and supports unequal-length or ragged source data when applied to the corresponding original dataset.

Both formats retain exact-zero entries. They are imputed-only sparse patches, not complete datasets. Complete imputed datasets continue to use reader-matched classes in `datasets.writers`.

See [Imputed-Only Output](Imputed_Only_Output.md).

## Matrix Market formats

## Coordinate general

```text
%%MatrixMarket matrix coordinate real general
```

## Coordinate symmetric

```text
%%MatrixMarket matrix coordinate real symmetric
```

## Dense array

```text
%%MatrixMarket matrix array real general
```

Dense array values are emitted in Matrix Market column-major order.

Optional descriptions are emitted as `%` comment lines. All coordinates in Matrix Market files are one-based.

Matrix Market writers reject non-finite numeric values. Dense writers also reject ragged matrices and null rows.

## Saved models

Request model persistence with:

```python
save_model=True
model_name="model"
```

Direct Java forms:

```text
-savemodel=true
-modelname=model
```

The artifact path is recorded in the repetition result's `artifacts` map when available. Multi-repetition runs apply the standard `_repeat_N` suffix rule.

See [Model Persistence](Model_Persistence.md) for saved-state contents and compatibility.

## Standardization statistics

Request saved reusable statistics with:

```python
save_standardization_stats=True
standardization_stats_output=(
    "../output/standardization_stats.json"
)
```

Direct Java forms:

```text
-save_standardization_stats=true
-standardization_stats_output=output/standardization_stats.json
```

Statistics output applies to reusable `global` and `per_dimension` scopes. Per-series scopes do not create reusable dataset-level statistics.

The artifact path is included in experiment-result metadata when available. See [Standardization](../guides/Standardization.md).

## Experiment results JSON

When `export_level >= 1`, PFGAP writes:

```text
experiment_results.json
```

The document collects all completed repetition records for one dataset and forest mode.

## Top-level schema

```text
formatVersion
generatedAt
dataset
forestMode
numRepeats
results
aggregateMetrics
aggregateCounts
aggregateTimingMilliseconds
aggregateForestStatistics
```

- `formatVersion` is currently `2`.
- `generatedAt` is an ISO offset date-time in UTC.
- `dataset` identifies the dataset shared by all records.
- `forestMode` identifies the shared task mode.
- `numRepeats` is the number of collected repetition records.
- `results` contains one record per repetition.

Records with different datasets, forest modes, format versions, or duplicate repetition numbers cannot be collected in one document.

## Repetition record schema

Each result record contains:

```text
formatVersion
dataset
repetition
forestId
forestMode
metrics
counts
timingMilliseconds
forestStatistics
configuration
artifacts
```

The `repetition` value is one-based.

The record intentionally does not embed forests, trees, datasets, prediction arrays, OOD arrays, outlier-score arrays, or proximity matrices. Large artifacts are written separately and referenced by path in `artifacts`.

## Representative metrics

Depending on the task and requested outputs, `metrics` can contain:

```text
accuracy
errorRate
rmse
mae
r2
meanOODScore
standardDeviationOODScore
meanPredictionStandardDeviation
```

For multi-output classification, accuracy uses exact full-vector matching. For multi-target regression, target-wise metrics are combined into the existing scalar summary where supported, including the averaged target-wise R2 summary.  
Non-finite metric values may remain present in individual repetition records.

## Representative counts

```text
correct
errors
predictionCount
trainingInstanceCount
testingInstanceCount
structuredResultCount
oodAvailableResultCount
oodAvailableTreeCount
```

Counts are nonnegative integers.

## Representative timings

```text
trainingMilliseconds
testingMilliseconds
proximityMilliseconds
```

Timing values must be finite and nonnegative.

## Representative forest statistics

```text
numTrees
meanNodesPerTree
standardDeviationNodesPerTree
meanDepthPerTree
standardDeviationDepthPerTree
meanWeightedDepthPerTree
standardDeviationWeightedDepthPerTree
```

## Configuration map

The configuration map can include values such as:

- forest mode and forest settings;
- reader types;
- proximity type;
- standardization method, scope, and variance convention;
- structured-output selection; and
- OOD selection.

Configuration keys and values are nonblank strings.

## Artifacts map

The artifacts map can reference:

- ordinary predictions;
- enhanced or OOD structured output;
- outlier or isolation scores;
- training proximities;
- test-to-training proximities;
- saved models;
- complete imputed datasets;
- imputed-only `.mtx` or `.tns` coordinate files; and
- standardization statistics.

Artifact keys and paths are nonblank strings.

## Aggregate summaries

The document aggregates recurring numeric values across repetitions. Each aggregate contains:

```text
count
mean
populationStandardDeviation
minimum
maximum
```

Only finite values contribute to aggregate metric, timing, and forest-statistic summaries. A non-finite value can remain in its individual repetition record while being excluded from aggregation.

Count maps are aggregated by converting the available count values to numeric summaries.

## CSV format

PFGAP row-oriented CSV artifacts use:

- UTF-8 encoding;
- commas as field delimiters; and
- CRLF record separators.

CSV fields are escaped using conventional rules:

1. Null values become empty fields.
2. Fields containing a comma, double quote, carriage return, or line feed are enclosed in double quotes.
3. A double quote inside a quoted field is written twice.

Examples:

```text
plain
```

remains:

```text
plain
```

while:

```text
value,with,commas
```

becomes:

```text
"value,with,commas"
```

and:

```text
value "with quotes"
```

becomes:

```text
"value ""with quotes"""
```

Finite numeric values are written numerically. CSV numeric rendering uses explicit text for non-finite values:

```text
NaN
Infinity
-Infinity
```

A boxed null numeric value is written as an empty field.

## File creation and replacement

Current CSV, JSON, Matrix Market, and sparse tensor writers:

- normalize output paths;
- create missing parent directories;
- write UTF-8; and
- replace an existing target file.

Experiment results are pretty-printed JSON and include null fields and special floating-point values where present.

## Legacy forest-result export

`ProximityForestResult` retains a legacy `exportJSON(...)` method that writes a timestamped JSON serialization using a filename composed from the forest ID and a timestamp.

The current coordinated experiment output is `experiment_results.json`, which stores compact repetition records and references large artifacts separately. New automation should use the coordinated output contract rather than rely on the legacy timestamped serialization.

## Output availability by workflow

## Training

A training repetition can produce:

- a saved model;
- scalar or multi-output training or validation predictions;
- enhanced scalar validation output;
- validation OOD output;
- supervised training outlier scores;
- isolation scores;
- training proximities;
- validation-to-training proximities;
- complete imputed training or validation data;
- imputed-only training or validation values;
- standardization statistics; and
- an experiment result record.

Availability depends on forest mode, supplied data, selected reader, and requested outputs.

## Saved-model evaluation

An evaluation repetition can produce:

- scalar or multi-output ordinary test predictions;
- enhanced scalar test prediction details;
- test OOD output when the model retains required summaries;
- isolation scores from an isolation model;
- test-to-training proximities;
- complete imputed test data;
- imputed-only test values; and
- an experiment result record.

## Export level

The coordinated experiment-results document is written when:

```text
export_level >= 1
```

Individual artifact requests are controlled by their corresponding output options.

## Common problems

## An expected file is absent

Verify that the corresponding output option is enabled and that the task, data, and model support that artifact.

## A structured CSV has empty columns

Structured output uses one stable combined schema. Fields that do not apply to the requested prediction or OOD result are intentionally empty.

## Enhanced output is requested for a multi-output task

Use ordinary prediction output for multi-output classification or multi-target regression. Enhanced prediction details are scalar-only in v1. OOD-only structured output remains independent of ordinary or enhanced prediction output.

## Actual-value columns are absent

Prediction writers omit actual-value columns when no actual values are available or all supplied actual values are null.

## A repetition overwrites another artifact

Use the coordinated repetition runner and artifact paths. Multi-repetition filenames include one-based `_repeat_N` suffixes.

## Matrix Market indices appear shifted

Matrix Market uses one-based coordinates. PFGAP instance indices in CSV files are zero-based.

## A sparse proximity file rejects zero

Sparse proximity storage does not retain exact-zero entries. Imputed-only output is different and can retain exact-zero coordinates.

## Matrix Market writing rejects a value

Matrix Market output requires finite numeric values. Correct or remove `NaN` and infinite values before writing that artifact.

## CSV values contain extra quotation marks

Quoted fields follow standard CSV escaping. Parse the file with a CSV reader instead of splitting rows manually on commas.

## Experiment aggregates omit a metric value

Only finite values contribute to aggregate summaries. Inspect the corresponding repetition record for `NaN` or infinity.

## An output path points outside the expected directory

Use paths resolved beneath the configured output directory and inspect the artifact paths recorded in `experiment_results.json`.

## Related documentation

- [Configuration Reference](Configuration_Reference.md)
- [CLI Reference](CLI_Reference.md)
- [Classification](../guides/Classification.md)
- [Regression](../guides/Regression.md)
- [Outlier Scoring](../guides/Outlier_Scoring.md)
- [OOD Scoring](../guides/OOD_Scoring.md)
- [Imputation](../guides/Imputation.md)
- [Imputed-Only Output](Imputed_Only_Output.md)
- [Model Persistence](Model_Persistence.md)
- [Writers](../data/Writers.md)
