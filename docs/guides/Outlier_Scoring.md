# Outlier Scoring

PFGAP provides two distinct outlier-scoring workflows:

1. **Unsupervised isolation scoring**, which trains a forest in `isolation` mode and scores observations from their isolation behavior.
2. **Supervised classification outlier scoring**, which produces Breiman-style training outlier scores from within-class forest proximities.

These workflows use different forest modes, different information, and different interpretations. Isolation scoring is the primary workflow in this guide.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
3. Confirm that the input format and representation are supported by [Readers](../data/Readers.md).
4. Read [Classification](Classification.md) before using supervised classification outlier scores.

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Choose the scoring workflow

Use **isolation mode** when:

- outlier scoring should not depend on class labels;
- the data are unlabeled; or
- the desired score is based on how quickly observations are isolated by the forest.

Use **supervised classification outlier scoring** when:

- a classification forest is being trained;
- integer class labels are available; and
- the desired score describes how unusual each training observation is relative to observations in its own class.

Do not compare isolation scores and supervised classification outlier scores as though they were the same quantity.

# Unsupervised isolation scoring

## Isolation data

Isolation mode operates on observations without using class labels to define splits or outlier scores. The selected observation representation, reader, distances, missing-value handling, and standardization settings must still be compatible.

If the physical input contains a target column because it shares a layout with another task, configure the reader consistently. The target is not used as a class label by the isolation forest.

## Train an isolation forest

Set:

```python
forest_mode="isolation"
```

Direct Java form:

```text
-forest_mode=isolation
```

The Python helper automatically selects the isolation purity setting when `forest_mode="isolation"` is used with its default purity argument.

### Python helper

Run the script from `Application/` when using the standard repository layout:

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    forest_mode="isolation",
    repeats=1,
    num_trees=101,
    r=5,
    isolation_num_branches=2,
    isolation_min_leaf_size=1,
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
    model_name="isolation_model",
    output_directory="../output/isolation_model",
    export=1,
    verbosity=1,
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

Create the output directory before direct execution:

```bash
mkdir -p output/isolation_model
```

Run from the repository root:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -forest_mode=isolation \
  -isRegression=false \
  -repeats=1 \
  -trees=101 \
  -r=5 \
  -isolation_num_branches=2 \
  -isolation_min_leaf_size=1 \
  -purity_measure=isolation_path_length \
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
  -modelname=isolation_model \
  -out=output/isolation_model/ \
  -export=1 \
  -verbosity=1
```

## Isolation settings

### Number of trees

Python:

```python
num_trees=101
```

Direct Java:

```text
-trees=101
```

Increasing the number of trees increases training and scoring work and may increase the size of saved models and optional outputs.

### Candidate split configurations

Python:

```python
r=5
```

Direct Java:

```text
-r=5
```

This controls the number of candidate distance configurations considered at each split.

### Number of branches

Python:

```python
isolation_num_branches=2
```

Direct Java:

```text
-isolation_num_branches=2
```

This controls the number of branches created by an isolation split.

### Minimum leaf size

Python:

```python
isolation_min_leaf_size=1
```

Direct Java:

```text
-isolation_min_leaf_size=1
```

This controls the minimum leaf size used by isolation trees.

### Isolation purity

Isolation mode uses:

```python
purity="isolation_path_length"
```

Direct Java form:

```text
-purity_measure=isolation_path_length
```

When the Python helper receives `forest_mode="isolation"` with its default `purity="gini"`, it replaces that default with `isolation_path_length`.

### Bootstrap sampling

Python:

```python
bootstrap_trees=True
```

Direct Java:

```text
-bootstrap_trees=true
```

Bootstrap sampling controls whether individual trees are trained from bootstrap samples of the training observations.

## Select distances

Isolation trees use the configured candidate distances when constructing splits. To supply an explicit set:

```python
distances=["dtw", "erp", "lcss"]
```

Direct Java form:

```text
-distances=[dtw,erp,lcss]
```

The selected distances must support the observation representation, dimensionality, sequence-length behavior, data type, and missing-value configuration.

See [Distances](../reference/Distances.md) for supported measures and compatibility requirements.

## One-dimensional and multivariate data

For tabular vectors or univariate sequences:

```python
data_dimension=1
```

Direct Java form:

```text
-is2D=false
```

For multivariate observations:

```python
data_dimension=2
```

Direct Java form:

```text
-is2D=true
```

For multivariate isolation, dimension subsampling can be enabled:

```python
subsample_dimensions=True
dimension_selection_strategy="SQRT"
```

See [Dataset Representations](../data/Dataset_Representations.md) and the [Configuration Reference](../reference/Configuration_Reference.md).

## Isolation outputs

Request isolation predictions with:

```python
return_predictions=True
```

Direct Java form:

```text
-get_predictions=true
```

Request the supported per-instance enhanced fields with:

```python
return_enhanced_outputs=True
```

Direct Java form:

```text
-return_enhanced_outputs=true
```

The exact output files, score fields, ordering, and numeric interpretation are documented in [Outputs](../reference/Outputs.md). Use a consistent PFGAP revision when comparing scores across runs.

## Score additional data with a saved isolation model

Save the model during training:

```python
save_model=True
model_name="isolation_model"
```

Then apply it to new observations.

### Python helper

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/isolation_model/isolation_model",
    testfile="../data/new_data.csv",
    exists_testlabels=False,
    forest_mode="isolation",
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    numeric_storage="auto",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    return_predictions=True,
    return_enhanced_outputs=True,
    output_directory="../output/isolation_scores",
    verbosity=1,
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

Create the output directory, then run:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=true \
  -train=data/new_data.csv \
  -test=data/new_data.csv \
  -exists_testlabels=false \
  -modelname=output/isolation_model/isolation_model \
  -forest_mode=isolation \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=auto \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -return_enhanced_outputs=true \
  -out=output/isolation_scores/ \
  -verbosity=1
```

The current evaluation interface supplies the evaluation data through both `-train` and `-test`. The Python helper performs the same mapping.

## Missing feature values

Isolation data may contain supported missing feature values. Handle them through:

- imputation; or
- compatible missing-aware distances where the selected workflow supports them.

See [Imputation](Imputation.md) and [Missing Values](../reference/Missing_Values.md).

## Standardization

Standardization can be enabled for supported numeric representations:

```python
standardization="zscore"
standardization_scope="per_dimension"
standardization_variance="population"
save_standardization_stats=True
```

Training statistics are fitted from the training data and reused when the saved isolation model scores later data. See [Standardization](Standardization.md).

# Supervised classification outlier scores

PFGAP also supports Breiman-style supervised outlier scores for classification training data. This is separate from isolation mode.

Supervised classification outlier scoring:

- requires a classification forest;
- requires integer class labels;
- uses forest proximities within each class; and
- returns scores for the training observations.

An observation receives its score relative to other training observations in the same class. The result therefore measures within-class unusualness rather than label-free isolation.

## Request supervised training outlier scores

### Python helper

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
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
    return_training_outlier_scores=True,
    output_directory="../output/classification_outliers",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
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
  -get_training_outlier_scores=true \
  -out=output/classification_outliers/
```

Create `output/classification_outliers/` before the direct Java run.

## Supervised score output

The supervised score output is associated with the training observations and their integer class labels. The score file name, columns, ordering, and normalization are documented in [Outputs](../reference/Outputs.md).

`return_training_outlier_scores` does not switch the forest into isolation mode. It adds supervised outlier output to a classification training run.

# Isolation scores versus supervised scores

## Isolation mode

- Forest mode: `isolation`
- Labels required: no
- Comparison basis: isolation behavior across the data
- Applicable data: training data and later data scored by a saved isolation model
- Main settings: `isolation_num_branches`, `isolation_min_leaf_size`, and `isolation_path_length`

## Supervised classification scoring

- Forest mode: `classification`
- Labels required: yes
- Comparison basis: proximity to other training observations in the same class
- Applicable data: classification training observations
- Main request: `return_training_outlier_scores=True`

Choose one according to the meaning required by the task. Enabling supervised training outlier output is not an alternative spelling for isolation mode.

## Reproducible scoring

Record:

```python
seed=42
num_workers=4
```

Also preserve:

- the PFGAP revision;
- the input data and ordering;
- labels for supervised scoring;
- the reader and representation settings;
- selected distances;
- missing-value and standardization settings;
- tree and split-candidate counts;
- isolation branch and leaf settings when applicable; and
- the saved model and output artifacts.

See [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md).

## Common problems

### Isolation mode is not selected

Set:

```python
forest_mode="isolation"
```

or:

```text
-forest_mode=isolation
```

### Supervised outlier scores are requested in isolation mode

Use `return_training_outlier_scores=True` with `forest_mode="classification"`. Isolation mode produces its own unsupervised scores.

### Supervised scoring has no class labels

Provide one integer class label for every training observation. See [Classification](Classification.md).

### A selected distance is incompatible

Check support for the observation dimensionality, representation, numeric or generic type, variable length, and missing values. See [Distances](../reference/Distances.md).

### New data are parsed differently from isolation training data

Use compatible reader, dimensionality, numeric storage, separators, feature ordering, standardization, and missing-value settings.

### The saved isolation model cannot be found

Use the model path produced by the training run. Relative paths are resolved from the current working directory.

### Python and direct Java runs differ

Compare the effective option values, working directory, resolved paths, input order, seed, worker count, and PFGAP revision.

## Next steps

- Read [OOD Scoring](OOD_Scoring.md) for evaluation-time distribution-shift scoring from split-distance support.
- Read [Classification](Classification.md) for supervised classification workflows.
- Read [Distances](../reference/Distances.md) for distance compatibility.
- Read [Outputs](../reference/Outputs.md) for score files and schemas.
- Read [Model Persistence](../reference/Model_Persistence.md) before moving saved isolation models between environments or PFGAP revisions.
