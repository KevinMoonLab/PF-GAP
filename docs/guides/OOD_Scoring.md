# OOD Scoring

This guide explains how to configure PFGAP's evaluation-time out-of-distribution (OOD) scoring. OOD scores compare evaluation observations with the split-distance support recorded by a trained forest. They are separate from unsupervised isolation scores and supervised classification outlier scores.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
3. Read the task guide for the forest being trained, such as [Classification](Classification.md) or [Regression](Regression.md).
4. Confirm that the selected data representation and reader are supported by [Readers](../data/Readers.md).

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## How OOD scoring works

During training, PFGAP can collect the distribution of winning split distances for each branch of each fitted split. During evaluation, each observation is routed through the saved forest. At every visited internal node, PFGAP compares the observation's winning distance with the maximum winning distance recorded for its selected branch during training.

The current OOD score type is:

```text
relative_support_exceedance
```

For a finite, nonnegative evaluation winning distance `d` and the selected branch's finite, nonnegative training maximum `m`, the node contribution is:

```text
0,            if d <= m
1 - (m / d),  if d > m
```

Each node contribution lies in `[0, 1]`:

- `0.0` means the evaluation distance does not exceed the observed branch boundary;
- `0.5` means the evaluation distance is twice the observed training maximum;
- `0.75` means the evaluation distance is four times the observed training maximum;
- values approach `1.0` as the exceedance grows; and
- a positive evaluation distance against a zero training maximum contributes `1.0`.

The tree score is the arithmetic mean of node contributions over every visited internal node. A visited node without usable summary metadata remains in the denominator and contributes zero to the numerator. Consequently, every available tree score also lies in `[0, 1]`.

The enhanced output aggregates available tree scores into `ood_mean` and `ood_standard_deviation`. Higher values indicate stronger relative support exceedance. These scores are comparative support measures, not calibrated probabilities that an observation is out of distribution.

The formula is invariant under positive rescaling of a distance measure and does not use an arbitrary near-zero denominator. OOD scoring requires finite, nonnegative winning distances and finite, nonnegative recorded training maxima.

OOD scoring is evaluation-time scoring. It does not change the forest's classification or regression prediction rule.

## OOD scoring is distinct from outlier scoring

PFGAP provides separate workflows for:

- **OOD scoring:** compares evaluation observations with training split-distance support;
- **isolation scoring:** uses an unsupervised isolation forest; and
- **supervised classification outlier scoring:** measures within-class unusualness among training observations using forest proximities.

See [Outlier Scoring](Outlier_Scoring.md) for the two outlier-scoring workflows.

## Training requirements

A forest must retain split-distance summaries before it can produce OOD scores for later data.

Enable summary collection with:

```python
collect_split_distance_summaries=True
```

Direct Java form:

```text
-collect_split_distance_summaries=true
```

Save the trained model so those summaries can be used during later evaluation:

```python
save_model=True
model_name="ood_model"
```

Direct Java form:

```text
-savemodel=true
-modelname=ood_model
```

## Train a model for later OOD scoring

### Python helper

The following example trains a classification forest, collects split-distance summaries, and saves the model:

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
    numeric_storage="auto",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    collect_split_distance_summaries=True,
    save_model=True,
    model_name="ood_model",
    output_directory="../output/ood_model",
    export=1,
    verbosity=1,
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

Create the output directory before direct execution:

```bash
mkdir -p output/ood_model
```

Run from the repository root:

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
  -numeric_storage=auto \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -collect_split_distance_summaries=true \
  -savemodel=true \
  -modelname=ood_model \
  -out=output/ood_model/ \
  -export=1 \
  -verbosity=1
```

## Score new data

Use the saved model and request OOD output during evaluation.

### Python helper

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/ood_model/ood_model",
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
    return_enhanced_outputs=True,
    return_ood_scores=True,
    ood_score_type="relative_support_exceedance",
    output_directory="../output/ood_scores",
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
  -modelname=output/ood_model/ood_model \
  -forest_mode=classification \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=auto \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -return_enhanced_outputs=true \
  -return_ood_scores=true \
  -ood_score_type=relative_support_exceedance \
  -out=output/ood_scores/ \
  -verbosity=1
```

The current evaluation interface supplies the evaluation data through both `-train` and `-test`. The Python helper performs the same mapping.

## Score a training-time test or validation set

OOD scores can also be requested for a test or validation set supplied during training:

```python
status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/validation.csv",
    exists_testlabels=True,
    forest_mode="classification",
    collect_split_distance_summaries=True,
    return_ood_scores=True,
    ood_score_type="relative_support_exceedance",
    return_enhanced_outputs=True,
    output_directory="../output/ood_validation",
)
```

When `return_ood_scores=True` is used with `PF.train(...)`, the Python helper automatically enables split-distance summary collection for the same run.

For direct Java use, set both options explicitly:

```text
-collect_split_distance_summaries=true
-return_ood_scores=true
```

## OOD score type

The current supported value is:

```python
ood_score_type="relative_support_exceedance"
```

Direct Java form:

```text
-ood_score_type=relative_support_exceedance
```

The Python helper also normalizes hyphens and spaces to underscores before checking the value.

## OOD output

Request OOD scores with:

```python
return_ood_scores=True
```

Direct Java form:

```text
-return_ood_scores=true
```

OOD information is included in the enhanced per-instance output. Current numeric fields include:

- `ood_mean`;
- `ood_standard_deviation`;
- `ood_available_tree_count`; and
- `ood_total_tree_count`.

`ood_available_tree_count` reports the number of trees that contributed an OOD score for the observation. `ood_total_tree_count` reports the total number of trees considered.

The exact filename, complete schema, empty-value behavior, and ordering are documented in [Outputs](../reference/Outputs.md).

## Read enhanced OOD output in Python

The helper can parse an enhanced CSV file into dictionaries:

```python
import PF_wrapper as PF

rows = PF.read_enhanced_output(
    "../output/ood_scores/test_enhanced.csv"
)

for row in rows:
    print(
        row["instance_index"],
        row["ood_mean"],
        row["ood_standard_deviation"],
        row["ood_available_tree_count"],
        row["ood_total_tree_count"],
    )
```

Use the enhanced filename created by the run.

## Classification and regression

OOD scoring is attached to the trained forest's split-distance support. It can be requested for supported saved models while retaining the model's task mode.

For classification:

```python
forest_mode="classification"
```

For regression:

```python
forest_mode="regression"
```

Use the same forest mode at evaluation that was used to train the saved model.

## Distance requirements

Relative support exceedance requires the selected distance measures to produce finite, nonnegative winning distances for both summary collection and later scoring.

PFGAP rejects a non-finite or negative winning distance when collecting OOD summaries. It also rejects a non-finite or negative evaluation winning distance rather than assigning an artificial finite contribution.

This requirement applies to built-in distances and custom Java distances. A distance that is valid for general routing is not necessarily compatible with distance-based OOD scoring if it can return `NaN`, positive infinity, negative infinity, or a negative value for the supplied data.

### Early abandoning

PFGAP can optionally disable best-so-far early abandoning:

```python
early_abandon_distances = False
```

Direct Java form:

```text
-early_abandon_distances=false
```

When early abandoning is enabled, cutoff-aware distances receive the current best competing distance. When it is disabled, they receive positive infinity and should perform a complete distance calculation.

Disabling early abandoning is primarily a diagnostic and reproducibility option. A correct cutoff-aware distance should select the same branch and return the same winning distance in both modes. If the winning distance remains non-finite with early abandoning disabled, the distance calculation itself did not produce a finite result for those inputs.

Custom Java distances may support the same contract through the cutoff-aware `compute(first, second, bestSoFar)` overload. Selected-dimension custom distances may support it through `compute(first, second, bestSoFar, selectedDimensions)`. Legacy custom distances may ignore the cutoff and compute the full distance.

A custom distance may abandon only after proving that its exact result must be strictly greater than `bestSoFar`. Exact equality must remain available so nearest-exemplar ties are preserved.

## Data compatibility

Evaluation data must use a representation and preprocessing contract compatible with the saved model, including:

- observation dimensionality;
- numeric or generic data type;
- numeric storage;
- reader and physical layout;
- separators and column selections;
- feature order;
- standardization; and
- missing-value handling.

OOD scoring does not convert incompatible evaluation data into the model's training representation.

## Missing feature values

Evaluation data with missing feature values must use a supported missing-data workflow. Configure imputation or compatible missing-aware handling before requesting OOD output.

See [Imputation](Imputation.md) and [Missing Values](../reference/Missing_Values.md).

## Standardization

When training uses standardization, later OOD evaluation must use the fitted training statistics rather than fitting new statistics from the evaluation data.

Example evaluation configuration:

```python
standardization="zscore"
standardization_scope="per_dimension"
standardization_variance="population"
standardization_stats="../output/ood_model/standardization_stats.json"
```

See [Standardization](Standardization.md) for supported methods, statistics persistence, and evaluation-time configuration.

## Reproducible OOD scoring

Record:

```python
seed = 42
num_workers = 4
early_abandon_distances = True
```

Also preserve:

- the PFGAP revision;
- the training and evaluation data and their ordering;
- the reader and representation settings;
- selected distances and their finite-distance behavior;
- the `early_abandon_distances` setting;
- standardization and missing-value settings;
- tree and split-candidate counts;
- the saved model containing split-distance summaries; and
- the OOD score type.

See [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md).

## Common problems

### The model does not contain OOD summaries

Train and save the model with:

```python
collect_split_distance_summaries=True
```

or:

```text
-collect_split_distance_summaries=true
```

### OOD output is not produced

Request:

```python
return_ood_scores=True
return_enhanced_outputs=True
```

or:

```text
-return_ood_scores=true
-return_enhanced_outputs=true
```

Then inspect the enhanced output described in [Outputs](../reference/Outputs.md).

### The OOD score type is rejected

Use:

```text
relative_support_exceedance
```

### OOD scoring reports a non-finite winning distance

The selected distance is not satisfying the finite-distance contract for the encountered inputs. Disable early abandoning and rerun the operation:

```python
early_abandon_distances = False
```

or:

```text
-early_abandon_distances=false
```

If the failure remains, inspect the distance implementation and the affected input representation. Do not convert a genuine or unresolved non-finite result into an arbitrary finite OOD score.

### Scores are near `1.0`

A score near `1.0` indicates strong relative exceedance. In particular, a positive evaluation winning distance against a branch whose recorded training maximum is zero contributes exactly `1.0` at that node. This is expected bounded behavior and is different from the former unbounded near-zero normalization.

### Some trees do not contribute a score

Use `ood_available_tree_count` together with `ood_total_tree_count` when interpreting an observation's OOD result.

### Evaluation data are parsed differently from training data

Use compatible reader, dimensionality, numeric storage, separators, column selections, feature ordering, standardization, and missing-value settings.

### The saved model cannot be found

Use the model path produced by the training run. Relative paths are resolved from the current working directory.

### OOD scores are confused with isolation scores

OOD scoring uses a trained forest's split-distance support. Isolation scoring uses `forest_mode="isolation"`. See [Outlier Scoring](Outlier_Scoring.md).

## Next steps

- Read [Standardization](Standardization.md) for fitted preprocessing statistics and evaluation-time reuse.
- Read [Outlier Scoring](Outlier_Scoring.md) for unsupervised isolation and supervised classification outlier scores.
- Read [Outputs](../reference/Outputs.md) for the enhanced OOD output schema.
- Read [Model Persistence](../reference/Model_Persistence.md) for saved-model compatibility.
- Read [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md) for repeatable execution.
