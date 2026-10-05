# Standardization

This guide explains how to standardize numeric PFGAP data. It covers the supported transformation methods, statistic scopes, variance conventions, fitting and reusing training statistics, per-series standardization, and common configuration errors.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
3. Confirm that the selected numeric representation and reader are supported by [Readers](../data/Readers.md).

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Standardization configuration

Standardization is controlled by three primary settings:

```python
standardization="z_score"
standardization_scope="per_dimension"
standardization_variance="population"
```

The corresponding direct Java settings are:

```text
-standardization=z_score
-standardization_scope=per_dimension
-standardization_variance=population
```

The method defines the mathematical transformation. The scope defines which values share transformation parameters. The variance convention affects z-score standardization only.

## Disable standardization

Standardization is disabled by default:

```python
standardization="none"
```

Direct Java form:

```text
-standardization=none
```

When standardization is disabled, do not provide a statistics input path, request fitted-statistics output, or provide a statistics output path.

## Supported methods

PFGAP currently supports `none`, `z_score`, `mean_center`, and `min_max`.

Method names are case-insensitive. Hyphens and spaces are converted to underscores.

### Z-score

```python
standardization="z_score"
```

For a value `x`, center `mean`, and scale `standardDeviation`:

```text
z = (x - mean) / standardDeviation
```

Z-score standardization uses the configured population or sample variance convention.

Recognized aliases include:

```text
zscore
standard
standardize
standardization
z_normalize
znormalize
z_normalization
znormalization
```

### Mean centering

```python
standardization="mean_center"
```

Mean centering subtracts the applicable mean without rescaling:

```text
centered = x - mean
```

The stored scale is `1.0`.

Recognized aliases include:

```text
mean_centering
meancenter
meancentering
center
centering
centre
centring
```

### Min-max scaling

```python
standardization="min_max"
```

Min-max scaling uses the applicable minimum and maximum:

```text
scaled = (x - minimum) / (maximum - minimum)
```

A constant statistic group uses scale `1.0`, so its values transform to zero.

Recognized aliases include:

```text
minmax
rescale
rescaling
range
range_scale
range_scaling
```

## Standardization scopes

PFGAP supports four scopes:

- `global`
- `per_dimension`
- `per_series`
- `per_series_per_dimension`

Scope names are case-insensitive. Hyphens and spaces are converted to underscores. A blank scope defaults to `per_dimension`.

## Global scope

```python
standardization_scope="global"
```

PFGAP fits one center and one scale from all accepted numeric values in the training dataset, across every instance, dimension, and time point.

The fitted parameters are reused for test, validation, and later evaluation data.

Recognized aliases include:

```text
dataset
whole_dataset
```

## Per-dimension scope

```python
standardization_scope="per_dimension"
```

PFGAP fits one reusable center and scale for each realized dimension.

For one-dimensional tabular rows, each array position is treated as a feature and receives its own training-set statistic group.

For dimension-major multivariate series, each outer-array channel receives its own statistic group fitted across training instances and time points.

The fitted parameters are reused for test, validation, and later evaluation data.

Recognized aliases include:

```text
dimension
dimensions
per_feature
feature
features
per_channel
channel
channels
```

## Per-series scope

```python
standardization_scope="per_series"
```

PFGAP calculates one center and scale independently for each realized series during transformation.

For a univariate series, all observed time points contribute to that series's local parameters.

For a multivariate series, all observed dimensions and time points in that instance contribute to the same local parameter group.

Per-series parameters are not reusable dataset-level training statistics.

Recognized aliases include:

```text
series
per_instance
instance
```

## Per-series, per-dimension scope

```python
standardization_scope="per_series_per_dimension"
```

PFGAP calculates one center and scale independently for each dimension within each realized series. This is per-instance, per-channel standardization for multivariate time series.

For univariate input, `per_series_per_dimension` is equivalent to `per_series`.

Recognized aliases include:

```text
per_series_dimension
series_per_dimension
per_instance_per_dimension
per_instance_dimension
per_series_per_feature
per_instance_per_feature
per_series_per_channel
per_instance_per_channel
```

## Choose a scope

Use `global` when every accepted numeric value should share one transformation.

Use `per_dimension` when features or channels should retain separate training-fitted parameters. This is the default scope.

Use `per_series` when each complete series should be centered and scaled from its own values.

Use `per_series_per_dimension` when every channel of every series should be standardized independently.

Only `global` and `per_dimension` use reusable statistics fitted from training data or loaded from JSON. The two per-series scopes calculate local parameters when each series is transformed.

## Variance convention

The variance convention applies only to z-score standardization.

### Population variance

```python
standardization_variance="population"
```

For `n` observed values and accumulated squared deviation `M2`:

```text
variance = M2 / n
```

Population variance is the default. Recognized aliases include:

```text
population_variance
n
ddof_0
ddof0
```

### Sample variance

```python
standardization_variance="sample"
```

Sample variance applies Bessel's correction:

```text
variance = M2 / (n - 1)
```

Recognized aliases include:

```text
sample_variance
n_minus_1
n_1
bessel
bessels_correction
ddof_1
ddof1
```

Sample variance requires more than one observed value in each statistic group.

## Fit reusable training statistics

The `global` and `per_dimension` scopes fit reusable statistics from training data when no statistics input path is supplied.

Example:

```python
standardization="z_score"
standardization_scope="per_dimension"
standardization_variance="population"
```

The fitted training parameters are then applied to the test or validation data in the same run.

Do not fit a separate set of reusable statistics from the test data.

## Save fitted statistics

To save newly fitted reusable statistics as JSON:

```python
save_standardization_stats=True
standardization_stats_output="../output/standardization_stats.json"
```

Direct Java form:

```text
-save_standardization_stats=true
-standardization_stats_output=output/standardization_stats.json
```

Statistics can be saved only with `global` or `per_dimension` scope. They cannot be saved when standardization is disabled or when a per-series scope is selected.

If `save_standardization_stats=True`, an optional output path may be supplied. Do not supply `standardization_stats_output` while saving is disabled.

## Load reusable statistics

To apply previously fitted training statistics:

```python
standardization="z_score"
standardization_scope="per_dimension"
standardization_variance="population"
standardization_stats="../output/standardization_stats.json"
```

Direct Java form:

```text
-standardization=z_score
-standardization_scope=per_dimension
-standardization_variance=population
-standardization_stats=output/standardization_stats.json
```

The loaded statistics must use the same method and scope as the active configuration. For z-score standardization, the variance convention must also match.

A configuration cannot both load precomputed statistics and request saving of newly fitted statistics.

## Per-series statistics

The `per_series` and `per_series_per_dimension` scopes calculate local parameters from each realized series. They do not use dataset-level statistics JSON.

Do not use these options with a per-series scope:

```python
standardization_stats="..."
save_standardization_stats=True
standardization_stats_output="..."
```

## Example: per-feature z-score standardization

This example fits one population z-score transformation per tabular feature and saves the reusable training statistics.

### Python helper

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    exists_testlabels=True,
    forest_mode="classification",
    num_trees=101,
    r=5,
    seed=42,
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    standardization="z_score",
    standardization_scope="per_dimension",
    standardization_variance="population",
    save_standardization_stats=True,
    standardization_stats_output=(
        "../output/standardized_model/standardization_stats.json"
    ),
    save_model=True,
    model_name="standardized_model",
    output_directory="../output/standardized_model",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

Create the output directory, then run:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -test=data/test.csv \
  -exists_testlabels=true \
  -forest_mode=classification \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -standardization=z_score \
  -standardization_scope=per_dimension \
  -standardization_variance=population \
  -save_standardization_stats=true \
  -standardization_stats_output=output/standardized_model/standardization_stats.json \
  -savemodel=true \
  -modelname=standardized_model \
  -out=output/standardized_model/
```

## Apply saved statistics during prediction

Use the training statistics when applying the saved model to later data.

### Python helper

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/standardized_model/standardized_model",
    testfile="../data/new_data.csv",
    exists_testlabels=False,
    forest_mode="classification",
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    standardization="z_score",
    standardization_scope="per_dimension",
    standardization_variance="population",
    standardization_stats=(
        "../output/standardized_model/standardization_stats.json"
    ),
    return_predictions=True,
    output_directory="../output/standardized_predictions",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=true \
  -train=data/new_data.csv \
  -test=data/new_data.csv \
  -exists_testlabels=false \
  -modelname=output/standardized_model/standardized_model \
  -forest_mode=classification \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -standardization=z_score \
  -standardization_scope=per_dimension \
  -standardization_variance=population \
  -standardization_stats=output/standardized_model/standardization_stats.json \
  -get_predictions=true \
  -out=output/standardized_predictions/
```

The current evaluation interface supplies evaluation data through both `-train` and `-test`.

## Example: per-series z-normalization

For univariate time series, calculate a separate z-score transformation for each series:

```python
standardization="z_score"
standardization_scope="per_series"
standardization_variance="population"
```

No statistics path or statistics-output options are used because parameters are calculated independently for each series.

## Example: per-channel multivariate normalization

For multivariate time series, calculate a separate transformation for every channel in every instance:

```python
standardization="z_score"
standardization_scope="per_series_per_dimension"
standardization_variance="population"
data_dimension=2
```

No dataset-level statistics file is used.

## Constant groups

For min-max scaling, a constant statistic group uses scale `1.0` and transforms to zero.

Standardization statistics and transformation behavior for constant z-score groups are defined by the standardization implementation and represented in the saved statistics. Use a compatible PFGAP revision when reusing statistics files.

## Numeric data requirement

Standardization applies to numeric dataset values. Configure:

```python
numeric_data=True
```

Direct Java form:

```text
-isNumeric=true
```

Do not enable numeric standardization for generic or categorical feature values.

## Missing values and imputation

Standardization accepts the observed numeric values available to the selected workflow. Missing-value recognition and imputation are configured separately.

When combining standardization and imputation, use a reader and representation supported by both features and retain the fitted training statistics for later evaluation.

See [Imputation](Imputation.md) and [Missing Values](../reference/Missing_Values.md).

## Eager and lazy data

Reusable `global` and `per_dimension` statistics are fitted from eager training data when no statistics file is supplied. A supplied statistics file can be loaded before readers are constructed.

Per-series scopes calculate local parameters from each realized series during transformation and do not require dataset-level fitting.

See [Eager and Lazy Data](Eager_and_Lazy_Data.md) and [Readers](../data/Readers.md) for reader-specific support.

## Common problems

### The method is rejected

Use one of the implemented methods:

```text
none
z_score
mean_center
min_max
```

### The scope is rejected

Use one of:

```text
global
per_dimension
per_series
per_series_per_dimension
```

### Statistics do not match the configuration

Use the same method and scope that were used when the JSON statistics were fitted. For z-score statistics, also use the same variance convention.

### Statistics are supplied for a per-series scope

Remove `standardization_stats`, `save_standardization_stats`, and `standardization_stats_output`. Per-series scopes calculate local parameters for each realized series.

### Loading and saving are requested together

Choose one operation:

- load existing statistics with `standardization_stats`; or
- fit and save new statistics with `save_standardization_stats=True`.

### A statistics output path is supplied without saving

Set `save_standardization_stats=True`, or remove `standardization_stats_output`.

### Sample variance has too few observed values

Each statistic group requires at least two observed values when `standardization_variance="sample"`.

### Prediction data use a different feature or channel layout

Use the same feature ordering, dimension count, reader configuration, method, scope, and variance convention as the training workflow.

## Next steps

- Read [Configuration Reference](../reference/Configuration_Reference.md) for the complete standardization option definitions.
- Read [Readers](../data/Readers.md) for reader compatibility.
- Read [Imputation](Imputation.md) before combining standardization with missing-data workflows.
- Read [Model Persistence](../reference/Model_Persistence.md) for the relationship between models and preprocessing artifacts.
- Read [Outputs](../reference/Outputs.md) for generated statistics and model artifacts.
