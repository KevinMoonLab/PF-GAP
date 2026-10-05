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

During training, PFGAP can collect split-distance summaries from the fitted forest. During evaluation, each observation is routed through the saved forest and compared with the support recorded at the visited splits.

The current OOD score type is:

```text
relative_support_exceedance
```

This score describes how strongly an evaluation observation's split distances exceed the corresponding training support. The output also records how many trees produced an available OOD score.

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
seed=42
num_workers=4
```

Also preserve:

- the PFGAP revision;
- the training and evaluation data and their ordering;
- the reader and representation settings;
- selected distances;
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
