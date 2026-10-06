# Quick Start

This guide walks through a small, complete PFGAP classification run using either the Python helper or the direct Java command line. Both examples use the same data and core configuration so that the two interfaces can be compared directly.

> **Prerequisite:** Complete the [Installation](Installation.md) guide first. This quick start assumes Java 21 is active and that `Application/PFGAP.jar` and `Application/PF_wrapper.py` come from the same PFGAP revision.

## What this example does

The example:

1. Creates a small numeric training dataset and test dataset.
2. Treats each row as one labeled, one-dimensional observation.
3. Trains a classification forest with 11 trees.
4. Uses five candidate distance configurations per split.
5. Fixes the random seed for repeatability.
6. Saves the fitted model.
7. Requests predictions for the test data.
8. Writes generated artifacts to an output directory.

The example is intentionally small and is intended to verify installation and demonstrate the basic workflow. It is not a benchmark or a recommended experimental design.

## Repository layout

Run the example from the repository root with this layout:

```text
PF-GAP/
├── Application/
│   ├── PFGAP.jar
│   └── PF_wrapper.py
├── quick-start-data/
│   ├── train.csv
│   └── test.csv
└── quick-start-output/
```

The output directory may not exist before the run. The Python helper creates a single missing output directory. For direct Java use, create it explicitly before launching PFGAP.

## Create the example data

Create `quick-start-data/`:

```bash
mkdir -p quick-start-data
```

On Windows PowerShell, use:

```powershell
New-Item -ItemType Directory -Force quick-start-data | Out-Null
```

Create `quick-start-data/train.csv` with the following contents:

```csv
1,0.0,0.1,0.2,0.1,0.0
1,0.1,0.2,0.3,0.2,0.1
1,0.0,0.0,0.1,0.2,0.1
1,0.2,0.1,0.2,0.1,0.0
2,1.0,0.9,0.8,0.9,1.0
2,0.9,0.8,0.7,0.8,0.9
2,1.0,1.0,0.9,0.8,0.9
2,0.8,0.9,0.8,0.9,1.0
```

Create `quick-start-data/test.csv` with the following contents:

```csv
1,0.1,0.1,0.2,0.2,0.1
1,0.0,0.2,0.2,0.1,0.0
2,0.9,0.9,0.8,0.8,0.9
2,1.0,0.8,0.8,0.9,1.0
```

Each row has this structure:

```text
label,value_1,value_2,value_3,value_4,value_5
```

The files have no header. The target is the first column, entries are comma-separated, and all feature values are numeric. With `data_dimension=1`, PFGAP treats each row as a one-dimensional numeric observation. Depending on the task and selected distance, this representation can describe either a fixed-width tabular feature vector or a univariate sequence. See [Dataset Representations](../data/Dataset_Representations.md) for the representation contract.

## Option 1: Run through Python

The Python helper constructs and launches the Java command. It does not reimplement training or prediction in Python.

Because the current helper launches `PFGAP.jar` by file name, run the script with `Application/` as the working directory. Dataset and output paths below are therefore relative to that directory.

Create `Application/quick_start.py`:

```python
import PF_wrapper as PF

exit_code = PF.train(
    train_file="../quick-start-data/train.csv",
    test_file="../quick-start-data/test.csv",
    exists_testlabels=True,
    return_predictions=True,
    save_model=True,
    model_name="quick_start_model",
    output_directory="../quick-start-output",
    forest_mode="classification",
    repeats=1,
    num_trees=11,
    r=5,
    seed=42,
    num_workers=1,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    verbosity=1,
)

if exit_code != 0:
    raise SystemExit(
        f"PFGAP exited with status {exit_code}. "
        "Review the Java output above for the reported cause."
    )

print("PFGAP quick-start run completed.")
```

Run it from `Application/`:

```bash
cd Application
python quick_start.py
cd ..
```

On systems where Python 3 is invoked as `python3`, substitute that command:

```bash
cd Application
python3 quick_start.py
cd ..
```

### Important Python-helper behavior

The helper maps Python-friendly arguments to the Java command-line names. In this example:

- `num_trees=11` becomes `-trees=11`.
- `return_predictions=True` becomes `-get_predictions=true`.
- `save_model=True` becomes `-savemodel=true`.
- `data_dimension=1` becomes `-is2D=false`.
- `numeric_data=True` becomes `-isNumeric=true`.
- `seed=42` adds `-seed=42`.
- `forest_mode="classification"` selects classification explicitly.

The helper also passes its remaining defaults. A direct Java command can omit options whose application defaults are appropriate, but the paired command below states the important values explicitly.

## Option 2: Run directly with Java

Create the output directory first:

```bash
mkdir -p quick-start-output
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force quick-start-output | Out-Null
```

From the repository root, run:

```bash
java -Xmx1g -jar Application/PFGAP.jar \
  -eval=false \
  -train=quick-start-data/train.csv \
  -test=quick-start-data/test.csv \
  -exists_testlabels=true \
  -forest_mode=classification \
  -repeats=1 \
  -trees=11 \
  -r=5 \
  -seed=42 \
  -num_workers=1 \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -savemodel=true \
  -modelname=quick_start_model \
  -out=quick-start-output/ \
  -verbosity=1
```

In PowerShell, use the backtick as the line-continuation character, or place the command on one line:

```powershell
java -Xmx1g -jar Application/PFGAP.jar `
  -eval=false `
  -train=quick-start-data/train.csv `
  -test=quick-start-data/test.csv `
  -exists_testlabels=true `
  -forest_mode=classification `
  -repeats=1 `
  -trees=11 `
  -r=5 `
  -seed=42 `
  -num_workers=1 `
  -is2D=false `
  -isNumeric=true `
  -entry_separator=, `
  -csv_has_header=false `
  -target_column=first `
  -get_predictions=true `
  -savemodel=true `
  -modelname=quick_start_model `
  -out=quick-start-output/ `
  -verbosity=1
```

PFGAP requires each direct command-line argument to use the exact `-name=value` form. Do not insert spaces around the equals sign.

## Understand the key options

### Execution mode

```text
-eval=false
```

Runs the training workflow. Evaluation-only execution, which loads an existing model and applies it to test data, uses `-eval=true` and is introduced below.

### Input data

```text
-train=quick-start-data/train.csv
-test=quick-start-data/test.csv
-exists_testlabels=true
```

These options identify the training and test inputs and indicate that labels are available for the test observations. This example stores labels in the first column of each file rather than in separate label files.

### Forest configuration

```text
-forest_mode=classification
-trees=11
-r=5
```

The forest is trained for classification with 11 trees and five candidate distance configurations considered at each split. The task guides discuss task-specific choices in more detail.

### Reproducibility and workers

```text
-seed=42
-num_workers=1
```

The seed controls PFGAP's configured randomness. A single worker is used in this first example to make the execution environment as simple as possible. Reproducibility can also depend on data ordering, implementation changes, custom extensions, and worker configuration. See [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md).

### Data representation

```text
-is2D=false
-isNumeric=true
-entry_separator=,
-csv_has_header=false
-target_column=first
```

These options select one-dimensional numeric observations, comma-delimited input, no header, and an embedded target in the first column.

### Requested artifacts

```text
-get_predictions=true
-savemodel=true
-modelname=quick_start_model
-out=quick-start-output/
```

These options request prediction output, save the fitted model under the supplied model name, and direct generated artifacts to `quick-start-output/`.

## Inspect the run

A successful invocation exits with status code `0`. Inspect the terminal output for the parsed configuration, progress, warnings, and completion status, then list the output directory:

```bash
find quick-start-output -maxdepth 2 -type f -print
```

On Windows PowerShell:

```powershell
Get-ChildItem quick-start-output -Recurse -File
```

The exact artifact set and file names depend on the export level, task, requested outputs, and current PFGAP revision. For this run, verify that:

- the process completed without an abort message;
- a saved model associated with `quick_start_model` is present;
- prediction-related output was produced for the test observations; and
- generated data was written under `quick-start-output/` rather than beside the input files.

See [Outputs](../reference/Outputs.md) for the authoritative artifact names and schemas.

## Apply the saved model later

Training with `save_model=True` or `-savemodel=true` creates a model that can be used in a later evaluation-only run. Use the actual saved model path produced by the training run.

### Python helper

From `Application/`, a later prediction call has this form:

```python
import PF_wrapper as PF

exit_code = PF.predict(
    model_name="../quick-start-output/quick_start_model",
    testfile="../quick-start-data/test.csv",
    exists_testlabels=True,
    return_predictions=True,
    output_directory="../quick-start-output/evaluation",
    forest_mode="classification",
    num_workers=1,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    file_has_header=False,
    target_column="first",
)

if exit_code != 0:
    raise SystemExit(exit_code)
```

Adjust `model_name` if the training run stores the model in a named subdirectory or with a revision-specific path convention.

### Direct Java

Create the nested evaluation output directory before the direct Java run:

```bash
mkdir -p quick-start-output/evaluation
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force quick-start-output/evaluation | Out-Null
```

The equivalent evaluation-only pattern is:

```bash
java -Xmx1g -jar Application/PFGAP.jar \
  -eval=true \
  -train=quick-start-data/test.csv \
  -test=quick-start-data/test.csv \
  -exists_testlabels=true \
  -modelname=quick-start-output/quick_start_model \
  -forest_mode=classification \
  -num_workers=1 \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -out=quick-start-output/evaluation/ \
  -verbosity=1
```

The current evaluation path receives the evaluation file through both `-train` and `-test`; the Python helper performs the same mapping internally. Model persistence behavior and compatibility boundaries are documented in [Model Persistence](../reference/Model_Persistence.md).

## Common first-run problems

### `Unable to access jarfile`

Run the direct command from the repository root, or replace `Application/PFGAP.jar` with the correct absolute path. When using the Python helper, run from `Application/` unless the helper's JAR location has been configured another way.

### `UnsupportedClassVersionError`

The active Java runtime is older than the runtime used to compile PFGAP. Verify that `java -version` reports Java 21.

### `Invalid command-line argument`

Every direct option must have the form `-name=value`. A missing equals sign, a space around the equals sign, or an unsupported option name causes argument parsing to fail.

### `Invalid Commandline Arguments`

One or more option names or values are not recognized by the current application revision. Ensure that the documentation, JAR, and Python helper come from compatible revisions.

### The output directory is missing

The Python helper creates one missing output directory with its configured path. Direct Java use should create the directory before execution. For nested output paths, create the parent hierarchy explicitly.

### The data are parsed incorrectly

Check all of the following together:

- `entry_separator` matches the file;
- `file_has_header` or `csv_has_header` matches the file;
- `target_column` matches the embedded label position;
- `data_dimension` or `is2D` matches the observation representation;
- `numeric_data` or `isNumeric` matches the feature type; and
- a compatible reader is being selected.

See [Data Formats](../data/Data_Formats.md), [Dataset Representations](../data/Dataset_Representations.md), and [Readers](../data/Readers.md).

## Where to go next

- Read [Configuration](Configuration.md) to understand the Python, command-line, and application configuration layers.
- Follow [Classification](../guides/Classification.md) for training, validation, evaluation, probability-style outputs, and classification-specific choices.
- Follow [Regression](../guides/Regression.md) for continuous targets.
- Follow [Imputation](../guides/Imputation.md) to replace supported missing values before or during a workflow.
- Read [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md) before working with large per-file datasets.
- Consult the [Configuration Reference](../reference/Configuration_Reference.md) and [CLI Reference](../reference/CLI_Reference.md) for complete option definitions.
