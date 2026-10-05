# Imputation

This guide explains how to impute missing feature values with PFGAP. It covers missing-value recognition, training and test imputation, the `impute_first` and `proximity_first` initialization strategies, initial imputers, standard and DTW-aligned proximity updates, iteration count, and full or imputed-only output.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
3. Confirm that the input format, representation, and reader are supported by [Readers](../data/Readers.md).
4. Review [Missing Values](../reference/Missing_Values.md) for the complete missing-data contract.

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Imputation workflow

PFGAP imputation has three main choices:

1. **Initialization strategy**
   - `impute_first`: fill missing values with an initial imputer before constructing the first forest proximities.
   - `proximity_first`: construct the first proximities directly from incomplete observations using missing-aware distances.
2. **Proximity update strategy**
   - `standard`: update missing positions from ordinary forest proximity weights.
   - `dtw_alignment`: use DTW alignment when transferring proximity-weighted values between sequences.
3. **Output form**
   - a complete imputed dataset; or
   - an imputed-only Matrix Market `.mtx` file containing values at the originally missing coordinates.

The initialization and update choices are independent. For example, a run can use `impute_first` initialization followed by `dtw_alignment` updates.

## Missing-value recognition

For delimited input, configure the strings that represent missing feature values:

```python
missing_indicators=["", "NA", "NaN", "null", "nan", "NAN"]
```

Direct Java form:

```text
-MissingStrings=[,NA,NaN,null,nan,NAN]
```

Numeric missing values are represented internally as `NaN`. Generic data use the missing representation supported by the selected reader and dataset representation.

Set missing indicators carefully. Any configured string is treated as missing wherever that reader applies the missing-value rules.

## Impute training data

Enable training-data imputation with:

```python
impute_training_data=True
```

Direct Java form:

```text
-perform_train_imputation=true
```

Training-data imputation is performed as part of forest training. The originally observed values remain observed; iterative updates apply to the positions recorded as originally missing.

## Impute test data

During a training run that also includes a test or validation dataset, enable test-data imputation with:

```python
impute_testing_data=True
```

Direct Java form:

```text
-perform_test_imputation=true
```

When applying a saved model through `PF.predict(...)`, use the same Python option:

```python
impute_testing_data=True
```

The corresponding direct Java option remains:

```text
-perform_test_imputation=true
```

## Initialization strategies

Select the initialization strategy with:

```python
imputation_initialization="impute_first"
```

or:

```python
imputation_initialization="proximity_first"
```

Direct Java forms:

```text
-imputation_initialization=impute_first
```

```text
-imputation_initialization=proximity_first
```

### Impute first

`impute_first` fills missing positions with the selected initial imputer before the first forest proximity calculation. The resulting complete data provide the starting values for proximity-based iterative updates.

Example:

```python
imputation_initialization="impute_first"
initial_imputer="mean"
```

This is the default initialization strategy in the Python helper.

### Proximity first

`proximity_first` computes the initial proximities while missing values are still present. It therefore requires one or more missing-aware proximity distances:

```python
imputation_initialization="proximity_first"
missing_proximity_distances=["nan_euclidean"]
```

Direct Java form:

```text
-imputation_initialization=proximity_first
-missing_proximity_distances=[nan_euclidean]
```

After the initial proximities are available, PFGAP uses them to initialize the missing positions and continues with the configured proximity update strategy.

## Missing-aware proximity distances

The following distances are supported by `missing_proximity_distances`:

- `nan_euclidean`
- `nan_euclidean_i`
- `dtwarow`
- `dtwarow_i`
- `dtwarow_d`

Specify one or more compatible measures:

```python
missing_proximity_distances=[
    "nan_euclidean",
    "dtwarow",
]
```

Direct Java form:

```text
-missing_proximity_distances=[nan_euclidean,dtwarow]
```

Use a distance whose dimensionality and sequence assumptions match the dataset. The `_i` and `_d` variants are multivariate distance variants and must be used with compatible two-dimensional observations.

`missing_proximity_distances` are used for the proximity-first handling of incomplete observations. The ordinary `distances` option selects the forest's regular candidate distances.

## Initial imputers

Select an initial imputer with:

```python
initial_imputer="mean"
```

Direct Java form:

```text
-initial_imputer=mean
```

The following initial imputers are available.

### Mean

```text
mean
```

Uses mean-based initial values within the applicable observation or feature structure.

### Global mean

```text
global_mean
```

Uses a mean calculated across the applicable dataset values.

### Median

```text
median
```

Uses median-based initial values within the applicable observation or feature structure.

### Global median

```text
global_median
```

Uses a median calculated across the applicable dataset values.

### Mode

```text
mode
```

Uses mode-based initial values within the applicable observation or feature structure. This is the initial imputer intended for supported categorical or generic values.

### Global mode

```text
global_mode
```

Uses a mode calculated across the applicable dataset values.

### Linear

```text
linear
```

Uses linear interpolation for missing positions in supported ordered sequences.

### KNN

```text
knn
```

Uses nearest neighbors under one or more configured KNN distances. KNN initialization requires `knn_distances`:

```python
initial_imputer="knn"
knn_distances=["dtw", "erp"]
```

Direct Java form:

```text
-initial_imputer=knn
-knn_distances=[dtw,erp]
```

The current KNN initializer uses five neighbors. The selected KNN distances must support the dataset representation and values supplied to the initializer.

## Update strategies

After initialization, PFGAP updates originally missing values from forest proximities. Select the update strategy with `gap_update`.

### Standard update

Python:

```python
gap_update="standard"
```

Direct Java:

```text
-gap_update=standard
```

The standard update uses proximity-weighted values from the corresponding positions of neighboring observations. It is appropriate when positions or features are directly comparable across observations.

### DTW-aligned update

Python:

```python
gap_update="dtw_alignment"
```

Direct Java:

```text
-gap_update=dtw_alignment
```

The DTW-aligned update uses dynamic time warping alignment when transferring values from neighboring sequences. It is intended for sequential data where corresponding patterns may occur at different time positions.

The older Python option:

```python
DTWImpute=True
```

also selects `dtw_alignment` when `gap_update` is not supplied. New configurations should use `gap_update` directly because it states the update strategy explicitly.

## Number of imputation iterations

Set the number of proximity-based imputation iterations with:

```python
impute_iterations=5
```

Direct Java form:

```text
-numImputes=5
```

Each iteration rebuilds or reapplies the configured proximity-based imputation process using the current imputed values. The Python helper default is five iterations.

## Complete imputed output

Request a complete imputed training dataset with:

```python
return_imputed_training=True
```

Direct Java form:

```text
-impute_train=true
```

Request a complete imputed test dataset with:

```python
return_imputed_testing=True
```

Direct Java form:

```text
-impute_test=true
```

The Python helper automatically enables the corresponding training or test imputation operation when a complete imputed output is requested.

Complete imputed output contains both originally observed values and imputed values. See [Writers](../data/Writers.md) and [Outputs](../reference/Outputs.md) for the applicable output layout and file names.

## Imputed-only Matrix Market output

PFGAP can return only the values at originally missing coordinates in a Matrix Market `.mtx` file.

For training data:

```python
return_imputed_training_csr=True
training_imputed_csr_file="../output/train_imputed_only.mtx"
```

Direct Java form:

```text
-output_train_imputed_csr=true
-train_imputed_csr_file=output/train_imputed_only.mtx
```

For test data:

```python
return_imputed_testing_csr=True
testing_imputed_csr_file="../output/test_imputed_only.mtx"
```

Direct Java form:

```text
-output_test_imputed_csr=true
-test_imputed_csr_file=output/test_imputed_only.mtx
```

The option names retain `csr`, while the supported file output is Matrix Market `.mtx`. Requesting imputed-only output automatically enables the corresponding imputation operation.

The `.mtx` output contains entries only for positions that were missing in the original input, including imputed values that are exactly zero. See [Imputed-Only Output](../reference/Imputed_Only_Output.md) for matrix dimensions, coordinate conventions, and metadata.

## Example: impute first with standard updates

This example performs mean initialization followed by five standard proximity updates. It returns a complete imputed dataset and imputed-only `.mtx` output for both training and test data.

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
    missing_indicators=["", "NA", "NaN", "null"],
    impute_training_data=True,
    impute_testing_data=True,
    imputation_initialization="impute_first",
    initial_imputer="mean",
    gap_update="standard",
    impute_iterations=5,
    return_imputed_training=True,
    return_imputed_testing=True,
    return_imputed_training_csr=True,
    return_imputed_testing_csr=True,
    training_imputed_csr_file="../output/imputation/train_imputed_only.mtx",
    testing_imputed_csr_file="../output/imputation/test_imputed_only.mtx",
    save_model=True,
    model_name="imputed_model",
    output_directory="../output/imputation",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

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
  '-MissingStrings=[,NA,NaN,null]' \
  -hasMissingValues=true \
  -perform_train_imputation=true \
  -perform_test_imputation=true \
  -imputation_initialization=impute_first \
  -initial_imputer=mean \
  -gap_update=standard \
  -numImputes=5 \
  -impute_train=true \
  -impute_test=true \
  -output_train_imputed_csr=true \
  -output_test_imputed_csr=true \
  -train_imputed_csr_file=output/imputation/train_imputed_only.mtx \
  -test_imputed_csr_file=output/imputation/test_imputed_only.mtx \
  -savemodel=true \
  -modelname=imputed_model \
  -out=output/imputation/
```

Create `output/imputation/` before the direct Java run.

## Example: proximity first with standard updates

This example constructs the initial proximities from incomplete data using missing-aware distances, then performs standard proximity updates.

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
    missing_indicators=["", "NA", "NaN", "null"],
    impute_training_data=True,
    impute_testing_data=True,
    imputation_initialization="proximity_first",
    missing_proximity_distances=["nan_euclidean"],
    gap_update="standard",
    impute_iterations=5,
    return_imputed_training=True,
    return_imputed_testing=True,
    output_directory="../output/proximity_first",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

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
  '-MissingStrings=[,NA,NaN,null]' \
  -hasMissingValues=true \
  -perform_train_imputation=true \
  -perform_test_imputation=true \
  -imputation_initialization=proximity_first \
  -missing_proximity_distances=[nan_euclidean] \
  -gap_update=standard \
  -numImputes=5 \
  -impute_train=true \
  -impute_test=true \
  -out=output/proximity_first/
```

## Example: DTW-aligned updates

For sequential data, select DTW-aligned proximity updates:

```python
status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    forest_mode="classification",
    data_dimension=1,
    numeric_data=True,
    impute_training_data=True,
    impute_testing_data=True,
    imputation_initialization="impute_first",
    initial_imputer="linear",
    gap_update="dtw_alignment",
    impute_iterations=5,
    return_imputed_training=True,
    return_imputed_testing=True,
    output_directory="../output/dtw_imputation",
)
```

The corresponding direct update option is:

```text
-gap_update=dtw_alignment
```

## Impute data with a saved model

Use `PF.predict(...)` to impute evaluation data with a saved forest:

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/imputation/imputed_model",
    testfile="../data/new_data.csv",
    exists_testlabels=False,
    data_dimension=1,
    numeric_data=True,
    missing_indicators=["", "NA", "NaN", "null"],
    impute_testing_data=True,
    imputation_initialization="impute_first",
    initial_imputer="mean",
    gap_update="standard",
    impute_iterations=5,
    return_imputed_testing=True,
    return_imputed_testing_csr=True,
    testing_imputed_csr_file="../output/predict/test_imputed_only.mtx",
    output_directory="../output/predict",
)

if status != 0:
    raise SystemExit(status)
```

For proximity-first prediction, also supply `missing_proximity_distances`:

```python
imputation_initialization="proximity_first"
missing_proximity_distances=["nan_euclidean"]
```

Use the same reader, representation, standardization, and missing-value conventions used by the saved model.

## Numeric and generic data

Numeric imputation supports the numeric representations accepted by the selected reader and workflow. Numeric missing values use `NaN` internally.

Mode-based initial imputation and proximity-weighted mode updates support categorical or generic values where the selected representation, reader, and distance configuration support generic data.

Configure the data type explicitly:

```python
numeric_data=True
```

or:

```python
numeric_data=False
```

Direct Java forms:

```text
-isNumeric=true
```

```text
-isNumeric=false
```

## One-dimensional and multivariate data

For tabular vectors or univariate sequences:

```python
data_dimension=1
```

For multivariate observations:

```python
data_dimension=2
```

The initial imputer, update strategy, missing-aware distance, and ordinary forest distances must support the selected representation.

Use independent (`_i`) or dependent (`_d`) multivariate missing-aware distances as appropriate for the configured multivariate distance behavior.

## Standardization and imputation

Standardization and imputation can be used in the same workflow when supported by the selected representation and reader. Training statistics are fitted from the training data and reused for test or later prediction data.

Example configuration:

```python
standardization="zscore"
standardization_scope="per_dimension"
standardization_variance="population"
save_standardization_stats=True
```

See [Standardization](Standardization.md) for the order of operations, saved statistics, and supported combinations.

## Reader compatibility

Imputation requires a dataset representation that supports access to missing coordinates and updating or materializing imputed values.

The current Python helper rejects imputation and imputed-data output for lazy `PER_FILE_PARQUET` datasets. Choose a supported eager reader or another compatible representation when imputation is required.

See [Readers](../data/Readers.md) and [Eager and Lazy Data](Eager_and_Lazy_Data.md) for the complete compatibility rules.

## Common problems

### Missing values are parsed as ordinary strings

Add the exact input representation to `missing_indicators` or `-MissingStrings`. Confirm that the selected reader applies those indicators to the relevant feature fields.

### KNN initialization reports missing distances

Supply at least one KNN distance:

```python
initial_imputer="knn"
knn_distances=["dtw"]
```

### Proximity-first initialization rejects a distance

Use only supported missing-aware proximity distances:

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

### Imputed output is not produced

Request the appropriate output explicitly:

```python
return_imputed_training=True
return_imputed_testing=True
```

or request imputed-only `.mtx` output:

```python
return_imputed_training_csr=True
return_imputed_testing_csr=True
```

### The `.mtx` path is incorrect

Set `training_imputed_csr_file` or `testing_imputed_csr_file` to the desired `.mtx` path. Ensure that its parent output directory exists for direct Java execution.

### DTW-aligned imputation is not used

Set:

```python
gap_update="dtw_alignment"
```

or:

```text
-gap_update=dtw_alignment
```

### The selected reader does not support imputation

Choose a reader and execution mode listed as imputation-compatible in [Readers](../data/Readers.md).

### Training and test data use different missing indicators

Configure all missing-value strings needed by both datasets. The same token must have the same meaning throughout the workflow.

## Next steps

- Read [Missing Values](../reference/Missing_Values.md) for the full missing-data contract.
- Read [Imputed-Only Output](../reference/Imputed_Only_Output.md) for `.mtx` coordinate and metadata conventions.
- Read [Distances](../reference/Distances.md) for ordinary and missing-aware distance compatibility.
- Read [Standardization](Standardization.md) before combining preprocessing and imputation.
- Read [Outputs](../reference/Outputs.md) for complete imputed dataset filenames and layouts.
- Read [Writers](../data/Writers.md) for complete dataset output formats.
