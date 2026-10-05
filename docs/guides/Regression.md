# Regression

This guide explains how to train, evaluate, save, and reuse a PFGAP regression forest. It covers numeric targets, regression-specific settings, prediction, optional outputs, preprocessing, and common errors.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Quick Start](../getting-started/Quick_Start.md) for the standard PFGAP workflow.
3. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
4. Confirm that the selected input layout and observation representation are supported by [Readers](../data/Readers.md).

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Regression data

A labeled regression dataset contains:

- one observation per instance; and
- one numeric target per instance.

Targets may be integers or floating-point values, but PFGAP treats them as continuous regression outcomes. Classification labels and multi-target regression are not part of the regression workflow described here.

Example targets include:

```text
2.4
-0.75
18.0
```

Missing training targets are not supported. Missing feature values are handled separately through imputation or compatible missing-aware distances.

## Target placement

Targets may be embedded in a supported input file or supplied separately when the selected reader supports separate target data.

### Target in the first column

A headerless delimited file may place the target first:

```csv
2.4,0.0,0.1,0.2,0.1,0.0
2.8,0.1,0.2,0.3,0.2,0.1
8.9,1.0,0.9,0.8,0.9,1.0
8.3,0.9,0.8,0.7,0.8,0.9
```

Python configuration:

```python
entry_separator=","
file_has_header=False
target_column="first"
```

Direct Java configuration:

```text
-entry_separator=,
-csv_has_header=false
-target_column=first
```

To use the final column instead, set `target_column="last"` or `-target_column=last`.

### Separate target files

When targets are stored separately, use `train_labels` and, when available, `test_labels`:

```python
train_file="../data/train.csv"
train_labels="../data/train_targets.csv"
test_file="../data/test.csv"
test_labels="../data/test_targets.csv"
```

The target order must match the observation order exactly.

## Train a regression forest

The following examples train a regression forest, evaluate it on a labeled test set, return predictions, and save the model.

### Python helper

Run the script from `Application/` when using the standard repository layout:

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    exists_testlabels=True,
    forest_mode="regression",
    repeats=1,
    num_trees=101,
    r=5,
    regression_num_branches=2,
    regressor_aggregation="mean",
    seed=42,
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    numeric_storage="auto",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    return_predictions=True,
    return_enhanced_outputs=True,
    save_model=True,
    model_name="regression_model",
    output_directory="../output/regression_model",
    export=1,
    verbosity=1,
)

if status != 0:
    raise SystemExit(status)
```

When `forest_mode="regression"`, the Python helper selects regression behavior and changes the default classification purity setting to regression variance.

### Direct Java

Create the output directory before direct execution:

```bash
mkdir -p output/regression_model
```

Run from the repository root:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -test=data/test.csv \
  -exists_testlabels=true \
  -forest_mode=regression \
  -isRegression=true \
  -repeats=1 \
  -trees=101 \
  -r=5 \
  -regression_num_branches=2 \
  -voting=mean \
  -purity_measure=variance \
  -purity_threshold=1e-6 \
  -seed=42 \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=auto \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -return_enhanced_outputs=true \
  -savemodel=true \
  -modelname=regression_model \
  -out=output/regression_model/ \
  -export=1 \
  -verbosity=1
```

## Regression mode

Set regression mode explicitly:

```python
forest_mode="regression"
```

Direct Java form:

```text
-forest_mode=regression
```

The direct interface also accepts the regression flag:

```text
-isRegression=true
```

Use `forest_mode` as the primary task setting. The Python helper supplies the corresponding regression flag automatically.

## Regression-specific settings

### Regression branches

Python:

```python
regression_num_branches=2
```

Direct Java:

```text
-regression_num_branches=2
```

This controls the configured number of branches used by regression splits.

### Prediction aggregation

Python:

```python
regressor_aggregation="mean"
```

Direct Java:

```text
-voting=mean
```

The aggregation setting controls how tree-level regression predictions are combined into the forest prediction. The default helper setting is `mean`.

### Regression purity

Regression uses variance as its default purity measure:

```python
purity="variance"
```

Direct Java form:

```text
-purity_measure=variance
```

The purity threshold can also affect stopping behavior:

```python
purity_threshold=1e-6
```

Direct Java form:

```text
-purity_threshold=1e-6
```

The [Configuration Reference](../reference/Configuration_Reference.md) contains the complete regression option definitions and accepted values.

## Core forest settings

### Number of trees

Python:

```python
num_trees=101
```

Direct Java:

```text
-trees=101
```

Increasing the number of trees increases training and prediction work and may increase saved-model and optional proximity-output size.

### Candidate split configurations

Python:

```python
r=5
```

Direct Java:

```text
-r=5
```

This setting controls the number of candidate distance configurations considered at each split.

### Bootstrap sampling

Python:

```python
bootstrap_trees=True
```

Direct Java:

```text
-bootstrap_trees=true
```

When enabled, trees are trained using bootstrap samples of the training observations.

### Maximum depth

Python:

```python
max_depth=0
```

Direct Java:

```text
-max_depth=0
```

See the [Configuration Reference](../reference/Configuration_Reference.md) for the current default and accepted values.

### Shuffling

Python:

```python
shuffle=True
```

Direct Java:

```text
-shuffle=true
```

Record the shuffle setting with the random seed and worker count when reproducing a run.

## Select distances

If `distances` is omitted or empty, PFGAP uses its configured default distance selection. To provide an explicit set:

```python
distances=["dtw", "erp", "lcss"]
```

Direct Java form:

```text
-distances=[dtw,erp,lcss]
```

The selected distances must support the observation representation, dimensionality, sequence-length behavior, data type, and missing-value configuration used by the dataset.

See [Distances](../reference/Distances.md) for supported measures and compatibility requirements.

## One-dimensional regression

For tabular feature vectors or univariate sequences, use:

```python
data_dimension=1
```

Direct Java form:

```text
-is2D=false
```

A one-dimensional numeric observation may use `double[]` or `float[]`. Its interpretation as tabular data or a univariate sequence depends on the selected reader and distance.

See [Dataset Representations](../data/Dataset_Representations.md).

## Multivariate regression

For multivariate observations, use:

```python
data_dimension=2
numeric_data=True
```

Direct Java form:

```text
-is2D=true
-isNumeric=true
```

The physical data layout must be supported by a compatible two-dimensional reader.

Dimension subsampling can be enabled for multivariate data:

```python
subsample_dimensions=True
dimension_selection_strategy="SQRT"
```

Supported strategies and companion settings are documented in the [Configuration Reference](../reference/Configuration_Reference.md).

## Labeled evaluation data

When the test or validation data include known numeric targets, set:

```python
exists_testlabels=True
```

Direct Java form:

```text
-exists_testlabels=true
```

PFGAP can then produce predictions and supported regression evaluation summaries.

The `test_file` option can be used for validation data during model development or for a final test set. Its role depends on how the data are used.

## Save a model

Request model persistence during training:

```python
save_model=True
model_name="regression_model"
```

Direct Java form:

```text
-savemodel=true
-modelname=regression_model
```

Use a separate output directory and model name for each run. See [Model Persistence](../reference/Model_Persistence.md) for saved-model structure and compatibility.

## Apply a saved model

Use the model path created by the training run.

### Python helper

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/regression_model/regression_model",
    testfile="../data/new_data.csv",
    exists_testlabels=False,
    forest_mode="regression",
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    return_predictions=True,
    return_enhanced_outputs=True,
    output_directory="../output/regression_predictions",
    verbosity=1,
)

if status != 0:
    raise SystemExit(status)
```

Adjust `model_name` to the model path produced by the training run.

### Direct Java

Create the prediction output directory, then run:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=true \
  -train=data/new_data.csv \
  -test=data/new_data.csv \
  -exists_testlabels=false \
  -modelname=output/regression_model/regression_model \
  -forest_mode=regression \
  -isRegression=true \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -return_enhanced_outputs=true \
  -out=output/regression_predictions/ \
  -verbosity=1
```

The current evaluation interface supplies the evaluation data through both `-train` and `-test`. The Python helper performs the same mapping.

## Unlabeled prediction data

For data without known targets, set:

```python
exists_testlabels=False
```

The selected reader must be configured so that no feature column is interpreted as a target. Target-dependent evaluation summaries are not produced for unlabeled data, but regression predictions can still be requested.

## Regression outputs

### Predictions

Python:

```python
return_predictions=True
```

Direct Java:

```text
-get_predictions=true
```

Regression predictions are numeric values.

### Enhanced outputs

Python:

```python
return_enhanced_outputs=True
```

Direct Java:

```text
-return_enhanced_outputs=true
```

Enhanced regression output includes the supported per-instance regression summary fields, including the mean and standard deviation of tree predictions and the number of contributing trees. See [Outputs](../reference/Outputs.md) for the exact file and column definitions.

The Python helper can read an enhanced CSV file:

```python
import PF_wrapper as PF

rows = PF.read_enhanced_output(
    "../output/regression_predictions/test_enhanced.csv"
)

for row in rows:
    print(
        row["instance_index"],
        row["prediction_mean"],
        row["prediction_standard_deviation"],
    )
```

Use the enhanced filename created by the run.

### Proximities

Python:

```python
return_proximities=True
```

Direct Java:

```text
-getprox=true
```

Proximity output can be used by supported downstream workflows and can be much larger than prediction output.

The authoritative artifact names, schemas, and availability rules are documented in [Outputs](../reference/Outputs.md).

## Missing feature values

Regression data may contain supported missing feature values. PFGAP can handle them through:

- imputation; or
- compatible missing-aware distances.

Missing regression targets are not supported for supervised training.

See [Imputation](Imputation.md) for the imputation workflow and [Missing Values](../reference/Missing_Values.md) for the complete missing-data contract.

## Standardization

Standardization can be enabled for supported numeric representations:

```python
standardization="zscore"
standardization_scope="per_dimension"
standardization_variance="population"
save_standardization_stats=True
```

Training statistics are fitted from the training data and reused for validation, test, and later prediction data. Standardization applies to input features, not to the regression targets.

See [Standardization](Standardization.md) for supported methods, scopes, saved statistics, and reader compatibility.

## Reproducible runs

Set and record the random seed and worker count:

```python
seed=42
num_workers=4
```

Also preserve:

- the PFGAP revision;
- the input data and targets;
- the reader and representation settings;
- the selected distances;
- preprocessing and imputation settings;
- the tree, branch, and split-candidate settings;
- the regression aggregation setting; and
- the saved model and requested outputs.

See [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md).

## Common problems

### Regression is not selected

Set:

```python
forest_mode="regression"
```

or:

```text
-forest_mode=regression
```

### Targets are not numeric

Convert the targets to numeric values before running PFGAP. Use classification instead when the target values represent discrete classes.

### Targets and observations are misaligned

When targets are stored separately, ensure that the target count and ordering match the observations exactly.

### A selected distance is incompatible

Check its support for the observation dimensionality, representation, variable length, and missing values. See [Distances](../reference/Distances.md).

### Prediction data are parsed differently from training data

Use compatible reader, dimensionality, numeric storage, separators, column selections, feature ordering, standardization, and missing-value settings.

### The model cannot be found

Use the saved model path produced by the training run. Relative paths are resolved from the current working directory.

### Python and direct Java runs differ

Compare the effective option values, working directory, resolved paths, input order, seed, worker count, and PFGAP revision.

## Next steps

- Use [Classification](Classification.md) for discrete class labels.
- Use [Imputation](Imputation.md) for missing feature values.
- Consult [Distances](../reference/Distances.md) for distance compatibility.
- Consult [Outputs](../reference/Outputs.md) for regression prediction and enhanced-output schemas.
- Consult [Model Persistence](../reference/Model_Persistence.md) before moving saved models between environments or PFGAP revisions.
