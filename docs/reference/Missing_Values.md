# Missing Values

This reference defines how PFGAP recognizes, stores, and handles missing feature values. It covers numeric and generic missingness, original missing-coordinate tracking, imputation, missing-aware distances, output behavior, and unsupported cases.

For task-oriented imputation instructions, see [Imputation](../guides/Imputation.md). For distance compatibility, see [Distances](Distances.md).

## Missing features and missing targets

PFGAP distinguishes between:

- **missing feature values**, which can be handled through supported imputation or missing-aware distance workflows; and
- **missing labels or targets**, which are not supported for supervised training.

A classification training observation requires one known integer class label. A regression training observation requires one known numeric target. Feature imputation does not impute labels or targets.

## Numeric missing values

Numeric missing feature values are represented internally as `NaN`.

Supported numeric readers convert configured missing-value tokens to the appropriate primitive missing value:

- `Double.NaN` for `double`-backed observations; and
- `Float.NaN` for `float`-backed observations.

Numeric storage is configured independently through `numeric_storage`. Missing numeric values remain `NaN` regardless of whether the observation uses `float32` or `float64` feature storage.

## Generic missing values

Generic or object-backed datasets use the missing representation supported by the selected reader and dataset implementation. Configure:

```python
numeric_data=False
```

Direct Java form:

```text
-isNumeric=false
```

Mode-based initial imputation and proximity-weighted mode updates are available for supported categorical or generic observations.

## Missing-value tokens

For compatible delimited readers, configure the text tokens interpreted as missing:

```python
missing_indicators=["", "NA", "NaN", "null", "nan", "NAN"]
```

Direct Java form:

```text
-MissingStrings=[,NA,NaN,null,nan,NAN]
```

The leading empty entry represents an empty field.

Token matching applies according to the selected reader. A configured token is treated as missing wherever that reader applies missing-value recognition, so do not include a token that is also a valid observed value.

## Declare missing-value handling

The direct Java setting is:

```text
-hasMissingValues=true
```

The Python helper accepts:

```python
has_missing_values=True
```

When `has_missing_values` is omitted, the helper derives it from the selected imputation operations, imputed-only output requests, proximity-first initialization, and missing-aware proximity distances.

Explicitly setting the value is useful when the input contains recognized missing features even if no imputation output is requested.

## Original missing coordinates

PFGAP records the coordinates that were missing in the original input. These coordinates remain distinct from the current feature values after initialization and iterative updates.

This distinction allows PFGAP to:

- update only originally missing positions during imputation;
- preserve originally observed values;
- return a complete imputed dataset; and
- return only imputed values at the originally missing coordinates.

An imputed value equal to zero remains an imputed value. It is not discarded or confused with an absent sparse entry.

## Missing-data strategies

PFGAP supports two principal ways to handle missing features:

1. **Imputation**, which fills missing positions and can iteratively update them from forest proximities.
2. **Missing-aware distances**, which compare incomplete observations directly in supported workflows.

The selected reader, observation representation, and distance set must support the chosen strategy.

## Imputation

## Training-data imputation

Enable training-data imputation with:

```python
impute_training_data=True
```

Direct Java form:

```text
-perform_train_imputation=true
```

## Test-data imputation

Enable test, validation, or evaluation-data imputation with:

```python
impute_testing_data=True
```

Direct Java form:

```text
-perform_test_imputation=true
```

## Initialization strategies

PFGAP supports:

```text
impute_first
proximity_first
```

### Impute first

```python
imputation_initialization="impute_first"
```

An initial imputer first fills missing positions. PFGAP then constructs forest proximities from the initialized data and performs the configured iterative updates.

### Proximity first

```python
imputation_initialization="proximity_first"
missing_proximity_distances=["nan_euclidean"]
```

PFGAP first computes proximities from incomplete observations using configured missing-aware distances. Those proximities initialize the missing positions before ordinary iterative updates continue.

## Initial imputers

The supported initial-imputer names are:

```text
mean
global_mean
median
global_median
mode
global_mode
linear
knn
```

Numeric data can use compatible mean, median, linear, or KNN initialization. Supported categorical or generic data can use mode-based initialization.

KNN initialization requires a separate distance set:

```python
initial_imputer="knn"
knn_distances=["dtw", "erp"]
```

The current KNN initializer uses five neighbors.

## Iterative update strategies

PFGAP supports:

```text
standard
dtw_alignment
```

### Standard update

```python
gap_update="standard"
```

The standard update transfers proximity-weighted values from corresponding positions or features.

### DTW-aligned update

```python
gap_update="dtw_alignment"
```

The DTW-aligned update uses dynamic time-warping alignment while transferring values between supported sequences.

## Imputation iterations

Set the number of iterative updates with:

```python
impute_iterations=5
```

Direct Java form:

```text
-numImputes=5
```

See [Imputation](../guides/Imputation.md) for complete workflows and examples.

## Missing-aware distances

The distances accepted for missing-aware proximity-first initialization are:

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

## One-dimensional distances

```text
nan_euclidean
dtwarow
```

## Independent multivariate distances

```text
nan_euclidean_i
dtwarow_i
```

## Dependent multivariate distance

```text
dtwarow_d
```

Configure them separately from ordinary forest distances:

```python
missing_proximity_distances=["nan_euclidean"]
```

Direct Java form:

```text
-missing_proximity_distances=[nan_euclidean]
```

Do not assume that an ordinary distance accepts `NaN` merely because it belongs to the same family as a missing-aware distance. Use the registered missing-aware identifier when incomplete observations are compared directly.

See [Distances](Distances.md) for dimensionality and representation compatibility.

## Missingness and observation representations

## One-dimensional numeric observations

One-dimensional numeric observations may represent:

- fixed-width tabular feature vectors; or
- univariate time series.

The meaning of a missing coordinate follows that interpretation. For tabular data, an index represents a feature. For a univariate series, an index represents a time position.

## Two-dimensional numeric observations

Two-dimensional numeric observations represent multivariate series or another supported matrix-like observation. Coordinates identify a dimension and a position within that dimension.

Independent missing-aware distances compare dimensions independently. Dependent missing-aware distances treat the multivariate observation jointly.

## Variable-length observations

Missingness and variable length are different concepts. A shorter sequence does not automatically contain missing values beyond its realized length. Missing coordinates exist only within the represented observation where the selected reader records a missing feature value.

Do not pad variable-length observations with missing tokens unless the intended data contract explicitly treats those padded positions as missing observations.

## Complete imputed output

Request a complete imputed training dataset with:

```python
return_imputed_training=True
```

Request a complete imputed test dataset with:

```python
return_imputed_testing=True
```

Direct Java forms:

```text
-impute_train=true
-impute_test=true
```

A complete imputed dataset contains both:

- originally observed values; and
- final imputed values at the originally missing coordinates.

The output layout depends on the supported writer selected by the application workflow. See [Outputs](Outputs.md) and [Writers](../data/Writers.md).

## Imputed-only output

PFGAP can write only the final values at originally missing coordinates in Matrix Market `.mtx` format.

For training data:

```python
return_imputed_training_csr=True
training_imputed_csr_file="../output/training_imputed_values.mtx"
```

For test data:

```python
return_imputed_testing_csr=True
testing_imputed_csr_file="../output/testing_imputed_values.mtx"
```

Direct Java forms:

```text
-output_train_imputed_csr=true
-train_imputed_csr_file=output/training_imputed_values.mtx
```

```text
-output_test_imputed_csr=true
-test_imputed_csr_file=output/testing_imputed_values.mtx
```

The option names retain `csr`, while the supported file representation is Matrix Market `.mtx`.

The output includes entries for originally missing coordinates even when the final imputed value is exactly zero. See [Imputed-Only Output](Imputed_Only_Output.md) for matrix shape, coordinate conventions, and companion metadata.

## Standardization and missing values

Standardization applies to supported observed numeric values according to the selected method and scope. Missing-value recognition and imputation are configured separately.

Reusable `global` and `per_dimension` statistics are fitted from training data or loaded from a saved statistics file. Per-series scopes calculate local statistics when each series is transformed.

When training data are standardized, use the same fitted reusable statistics during later evaluation. See [Standardization](../guides/Standardization.md).

## Reader compatibility

Missing-value support depends on the selected reader and representation.

A compatible reader must:

- recognize the format's missing representation;
- produce the appropriate numeric or generic missing value;
- preserve original missing coordinates when the workflow requires imputation; and
- support the materialization or mutation required by the requested output.

Some readers can return observations containing missing values but do not support iterative mutation or complete imputed-data output. Consult [Readers](../data/Readers.md) before selecting a reader for an imputation workflow.

## Classification, regression, and scoring

## Classification

Training requires one known integer class label per observation. Missing feature values can be imputed or handled through compatible missing-aware distances.

## Regression

Training requires one known numeric target per observation. Missing feature values use the same supported handling strategies as classification.

## Isolation scoring

Isolation workflows can use supported missing-data handling without class labels. The chosen distance and reader must remain compatible.

## OOD scoring

Evaluation observations must be transformed and handled consistently with the saved model. If training used imputation or fitted standardization statistics, later OOD evaluation must use the corresponding data contract and preprocessing artifacts.

## Unsupported cases

The following are not supported by the missing-feature workflow:

- missing classification labels;
- missing regression targets;
- using an ordinary distance that does not accept the represented missing values;
- KNN initialization without KNN distances;
- proximity-first initialization without a compatible missing-aware distance;
- iterative imputation through a reader that does not support the required materialization or mutation; and
- treating absent variable-length positions as automatically missing.

## Common problems

## Missing tokens are parsed as ordinary values

Add the exact token to `missing_indicators` or `-MissingStrings`. Verify that the selected reader applies token-based missing recognition.

## Numeric parsing fails on an empty field

Include the empty string in the configured missing indicators:

```python
missing_indicators=["", "NA", "NaN"]
```

## Proximity-first initialization rejects the distance list

Use only:

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

## KNN initialization fails

Supply at least one compatible KNN distance:

```python
initial_imputer="knn"
knn_distances=["dtw"]
```

## Originally observed values change unexpectedly

Use the PFGAP imputation workflow and a supported reader. Iterative imputation updates coordinates recorded as originally missing.

## An exactly zero imputation is absent from an external conversion

Use the PFGAP Matrix Market `.mtx` output and preserve its original-missing-coordinate semantics. A zero imputation is meaningful and must not be dropped solely because its value is zero.

## Complete imputed output is unavailable

Choose a reader and writer combination that supports complete materialization and output, or request imputed-only `.mtx` output where supported.

## Training and evaluation use different missing tokens

Configure the union of tokens required by the datasets, and use each token consistently throughout the workflow.

## Standardization differs between training and evaluation

Reuse the fitted training statistics for `global` or `per_dimension` standardization. Do not fit reusable statistics independently from evaluation data.

## Related documentation

- [Imputation](../guides/Imputation.md)
- [Distances](Distances.md)
- [Dataset Representations](../data/Dataset_Representations.md)
- [Readers](../data/Readers.md)
- [Standardization](../guides/Standardization.md)
- [Outputs](Outputs.md)
- [Imputed-Only Output](Imputed_Only_Output.md)
- [Writers](../data/Writers.md)
