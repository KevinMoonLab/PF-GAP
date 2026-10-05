# Classification

This guide explains how to train, evaluate, save, and reuse a PFGAP classification forest. It covers the classification data contract, core configuration, prediction, optional outputs, preprocessing, and common user-facing errors.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Run the [Quick Start](../getting-started/Quick_Start.md), or prepare an equivalent dataset.
3. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
4. Confirm that the selected file layout and observation representation are supported by [Readers](../data/Readers.md).

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Classification data

A labeled classification dataset contains:

- one observation per instance; and
- one integer class label per instance.

PFGAP currently supports single-label classification with integer class labels. Examples of valid labels include:

```text
0
1
2
```

or:

```text
1
2
3
```

String labels and multi-label targets are not currently supported. If the original dataset uses text labels, encode them as integers before running PFGAP and retain the mapping so predictions can be converted back to the original class names.

For example:

```text
setosa     -> 1
versicolor -> 2
virginica  -> 3
```

Apply the same mapping to the training, validation, test, and later prediction data.

## Label placement

Labels may be embedded in a supported input file or supplied separately when the selected reader supports separate label data.

### Labels in the first column

A headerless delimited file may store the label first:

```csv
1,0.0,0.1,0.2,0.1,0.0
1,0.1,0.2,0.3,0.2,0.1
2,1.0,0.9,0.8,0.9,1.0
2,0.9,0.8,0.7,0.8,0.9
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

### Separate label files

When labels are stored separately, use `train_labels` and, when available, `test_labels`:

```python
train_file="../data/train.csv"
train_labels="../data/train_labels.csv"
test_file="../data/test.csv"
test_labels="../data/test_labels.csv"
```

The labels must appear in the same order as the corresponding observations.

## Train a classification forest

The following examples train a classification forest, evaluate it on a labeled test set, return predictions, and save the model.

### Python helper

Run the script from `Application/` when using the standard repository layout:

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    exists_testlabels=True,
    forest_mode="classification",
    repeats=1,
    num_trees=101,
    r=5,
    seed=42,
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    numeric_storage="auto",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    return_predictions=True,
    save_model=True,
    model_name="classification_model",
    output_directory="../output/classification_model",
    export=1,
    verbosity=1,
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

Create the output directory before direct execution:

```bash
mkdir -p output/classification_model
```

Run from the repository root:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -test=data/test.csv \
  -exists_testlabels=true \
  -forest_mode=classification \
  -repeats=1 \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=auto \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -savemodel=true \
  -modelname=classification_model \
  -out=output/classification_model/ \
  -export=1 \
  -verbosity=1
```

## Classification mode

Set classification mode explicitly:

```python
forest_mode="classification"
```

Direct Java form:

```text
-forest_mode=classification
```

This selects classification behavior for splitting, leaf predictions, voting, and evaluation.

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

Increasing the number of trees increases training and prediction work and may increase the size of saved models and optional proximity outputs.

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

The [Configuration Reference](../reference/Configuration_Reference.md) defines the current default and accepted values.

### Shuffling

Python:

```python
shuffle=True
```

Direct Java:

```text
-shuffle=true
```

Record the shuffle setting together with the random seed and worker count when reproducing a run.

## Purity settings

Classification uses a classification-compatible purity measure. The default Python configuration uses Gini impurity:

```python
purity_measure="gini"
```

Direct Java form:

```text
-purity_measure=gini
```

The purity threshold can also affect stopping behavior:

```python
purity_threshold=1e-6
```

Direct Java form:

```text
-purity_threshold=1e-6
```

See the [Configuration Reference](../reference/Configuration_Reference.md) for supported purity settings and defaults.

## Select distances

If `distances` is omitted or empty, PFGAP uses its configured default distance selection. To provide an explicit set:

```python
distances=["dtw", "erp", "lcss"]
```

Direct Java form:

```text
-distances=[dtw,erp,lcss]
```

The selected distances must support the observation representation, numeric or generic data type, sequence-length behavior, dimensionality, and missing-value configuration used by the dataset.

See [Distances](../reference/Distances.md) for the supported measures and their compatibility requirements.

## One-dimensional classification

For tabular feature vectors or univariate sequences, use:

```python
data_dimension=1
```

Direct Java form:

```text
-is2D=false
```

A one-dimensional observation may be represented by `double[]`, `float[]`, or a supported generic one-dimensional representation. Its interpretation as tabular data or a univariate sequence depends on the selected reader, distance, and task configuration.

See [Dataset Representations](../data/Dataset_Representations.md).

## Multivariate classification

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

The physical input layout must be supported by a compatible two-dimensional reader.

Dimension subsampling can be enabled for multivariate data:

```python
subsample_dimensions=True
dimension_selection_strategy="SQRT"
```

Supported strategies and their companion settings are documented in the [Configuration Reference](../reference/Configuration_Reference.md).

## Labeled evaluation data

When the test or validation data include known labels, set:

```python
exists_testlabels=True
```

Direct Java form:

```text
-exists_testlabels=true
```

PFGAP can then produce predictions and supported evaluation summaries for those observations.

The `test_file` option can be used for a validation set during model development or for a final test set. Its role depends on how the data are used in the experiment.

## Save a model

Request model persistence during training:

```python
save_model=True
model_name="classification_model"
```

Direct Java form:

```text
-savemodel=true
-modelname=classification_model
```

Use a separate output directory and model name for each run. See [Model Persistence](../reference/Model_Persistence.md) for saved-model structure and compatibility.

## Apply a saved model

Use the model path created by the training run.

### Python helper

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/classification_model/classification_model",
    testfile="../data/new_data.csv",
    exists_testlabels=False,
    forest_mode="classification",
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    numeric_storage="auto",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    return_predictions=True,
    output_directory="../output/classification_predictions",
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
  -modelname=output/classification_model/classification_model \
  -forest_mode=classification \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=auto \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -out=output/classification_predictions/ \
  -verbosity=1
```

The current evaluation interface supplies the evaluation data through both `-train` and `-test`. The Python helper performs the same mapping.

## Unlabeled prediction data

For data without known labels, set:

```python
exists_testlabels=False
```

The selected reader must be configured so that no feature column is interpreted as a target. Label-dependent evaluation summaries are not produced for unlabeled data, but predictions can still be requested.

## Classification outputs

### Predictions

Python:

```python
return_predictions=True
```

Direct Java:

```text
-get_predictions=true
```

Predictions use the integer class labels supplied during training. If the original classes were encoded from strings, convert predictions using the same saved mapping.

### Enhanced outputs

Python:

```python
return_enhanced_outputs=True
```

Direct Java:

```text
-return_enhanced_outputs=true
```

Enhanced output availability and schemas are documented in [Outputs](../reference/Outputs.md).

### Proximities

Python:

```python
return_proximities=True
```

Direct Java:

```text
-getprox=true
```

Proximity output can be used for supported downstream analysis and can be much larger than prediction output.

### Training outlier scores

Python:

```python
return_training_outlier_scores=True
```

Direct Java:

```text
-get_training_outlier_scores=true
```

See [Outlier Scoring](Outlier_Scoring.md) for the complete workflow.

### OOD scores

Python:

```python
return_ood_scores=True
```

OOD scoring has additional training-time and evaluation-time requirements. See [OOD Scoring](OOD_Scoring.md).

The authoritative artifact names, schemas, and availability rules are documented in [Outputs](../reference/Outputs.md).

## Missing feature values

Classification data may contain supported missing feature values. PFGAP can handle them through:

- imputation; or
- compatible missing-aware distances.

Missing class labels are not supported for supervised training.

See [Imputation](Imputation.md) for the imputation workflow and [Missing Values](../reference/Missing_Values.md) for the complete missing-data contract.

## Standardization

Standardization can be enabled for supported numeric representations:

```python
standardization="zscore"
standardization_scope="per_dimension"
standardization_variance="population"
save_standardization_stats=True
```

Training statistics are fitted from the training data and reused for validation, test, and later prediction data. See [Standardization](Standardization.md) for supported methods, scopes, saved statistics, and reader compatibility.

## Reproducible runs

Set and record the random seed and worker count:

```python
seed=42
num_workers=4
```

Also preserve:

- the PFGAP revision;
- the input data and label mapping;
- the reader and representation settings;
- the selected distances;
- preprocessing and imputation settings;
- the tree and split-candidate counts; and
- the saved model and requested outputs.

See [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md).

## Common problems

### String labels are present

Encode every class as an integer before running PFGAP. Apply the same mapping to all datasets and retain it with the model outputs.

### Labels and observations are misaligned

When labels are stored separately, ensure that the label count and ordering match the observations exactly.

### Classification is not selected

Set:

```python
forest_mode="classification"
```

or:

```text
-forest_mode=classification
```

### A selected distance is incompatible

Check its support for the observation dimensionality, representation, numeric or generic type, variable length, and missing values. See [Distances](../reference/Distances.md).

### Prediction data are parsed differently from training data

Use compatible reader, dimensionality, numeric storage, separators, column selections, feature ordering, standardization, and missing-value settings.

### The model cannot be found

Use the saved model path produced by the training run. Relative paths are resolved from the current working directory.

### Python and direct Java runs differ

Compare the effective option values, working directory, resolved paths, input order, seed, worker count, and PFGAP revision.

## Next steps

- Use [Regression](Regression.md) for continuous targets.
- Use [Imputation](Imputation.md) for missing feature values.
- Use [Outlier Scoring](Outlier_Scoring.md) for training-set outlier analysis.
- Use [OOD Scoring](OOD_Scoring.md) for evaluation-time distribution-shift scoring.
- Consult [Distances](../reference/Distances.md) for distance compatibility.
- Consult [Outputs](../reference/Outputs.md) for classification artifact schemas.
- Consult [Model Persistence](../reference/Model_Persistence.md) before moving saved models between environments or PFGAP revisions.
