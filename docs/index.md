# PFGAP Documentation

PFGAP is a Java-based Proximity Forest framework and command-line application for classification, regression, imputation, outlier detection, out-of-distribution scoring, proximity analysis, and extensible distance-based learning on time series and structured data.

The Java application contains the primary implementation. A Python helper is available for scripts, notebooks, and Python-oriented experiment workflows.

> **Pre-v1 documentation**
>
> PFGAP is under active development toward version 1.0. These pages document the current public interface. Preserve the PFGAP revision, Java runtime, configuration, extension artifacts, and preprocessing state used by an experiment.

## Start here

If this is your first time using PFGAP, follow these pages in order:

1. [Installation](getting-started/Installation.md) explains Java, the Python helper, optional integrations, and build requirements.
2. [Quick Start](getting-started/Quick_Start.md) walks through a minimal end-to-end workflow.
3. [Configuration](getting-started/Configuration.md) introduces paths, options, precedence, and common configuration patterns.

For a systematic catalog of Python arguments and Java options, use the [Configuration Reference](reference/Configuration_Reference.md). For direct Java command syntax, use the [CLI Reference](reference/CLI_Reference.md).

## Choose an interface

### Python helper

`Application/PF_wrapper.py` constructs and launches the Java application.

Use it from:

- Python scripts;
- Jupyter or Marimo notebooks;
- experiment pipelines; or
- applications that already coordinate work in Python.

The helper does not reimplement PFGAP algorithms. Training, evaluation, proximity computation, imputation, scoring, and data processing remain in Java.

A typical call is:

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    forest_mode="classification",
    num_trees=101,
    r=5,
    seed=42,
    num_workers=4,
    output_directory="../output/classification",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java command line

`Application/PFGAP.jar` can be launched directly:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -forest_mode=classification \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -out=output/classification/
```

PFGAP application options use:

```text
-name=value
```

Task guides show Python and direct Java forms where both interfaces support the workflow.

## Guides

Guides are organized around tasks rather than individual classes.

### Learning and scoring

- [Classification](guides/Classification.md) explains training, validation, saved-model evaluation, labels, predictions, and classification metrics.
- [Regression](guides/Regression.md) explains numeric targets, regression forests, aggregate predictions, and regression metrics.
- [Imputation](guides/Imputation.md) explains initialization, iterative proximity updates, training and test imputation, and output choices.
- [Outlier Scoring](guides/Outlier_Scoring.md) distinguishes unsupervised isolation scoring from Breiman-style supervised classification outlier scores.
- [OOD Scoring](guides/OOD_Scoring.md) explains evaluation-time scoring from retained split-distance support.

### Data processing and execution

- [Standardization](guides/Standardization.md) covers methods, scopes, variance conventions, saved statistics, and inverse transformation.
- [Eager and Lazy Data](guides/Eager_and_Lazy_Data.md) explains materialized and deferred reader types, resources, reconstruction, and workflow compatibility.
- [Parallelism and Reproducibility](guides/Parallelism_and_Reproducibility.md) covers worker budgets, random seeds, repetitions, and reproducible archives.

## Data and I/O

PFGAP separates physical storage from logical observation representation.

- [Data Formats](data/Data_Formats.md) describes supported external layouts and file formats.
- [Dataset Representations](data/Dataset_Representations.md) defines numeric and generic in-memory observation shapes.
- [Readers](data/Readers.md) lists implemented reader types, required options, materialization behavior, and representation contracts.
- [Writers](data/Writers.md) describes complete dataset output, mirror-writer support, labels, headers, and inverse transformation.

For cross-cutting missing-data rules, see [Missing Values](reference/Missing_Values.md).

## Extensions

PFGAP can load user implementations without placing them in the main source tree.

- [Custom Distances](extensions/Custom_Distances.md) explains Java, Python, Maple, and meta distances, including separate JAR and fat JAR packaging.
- [Custom Readers](extensions/Custom_Readers.md) explains custom eager and deferred readers, separate reader JARs, thread safety, source discovery, and saved-model reconstruction.

Custom extensions execute with the PFGAP process's permissions. Use trusted implementations and retain their exact versions with reproducible experiment artifacts.

## Reference

Reference pages define current public contracts.

- [Configuration Reference](reference/Configuration_Reference.md) maps Python arguments to Java options and records defaults, accepted values, and dependencies.
- [CLI Reference](reference/CLI_Reference.md) defines direct Java syntax and every current command-line option.
- [Distances](reference/Distances.md) lists the distance registry, data-family conventions, parameter behavior, missing-aware measures, and extension descriptors.
- [Missing Values](reference/Missing_Values.md) defines missing representations, original-coordinate tracking, supported handling strategies, and unsupported cases.
- [Outputs](reference/Outputs.md) defines prediction, score, proximity, model, Matrix Market, and experiment-result artifacts.
- [Imputed-Only Output](reference/Imputed_Only_Output.md) defines output containing final values only at originally missing coordinates.
- [Model Persistence](reference/Model_Persistence.md) explains saved model contents, restored context, external dependencies, reconstruction, and compatibility.

## Common documentation paths

### Train a classifier

1. [Installation](getting-started/Installation.md)
2. [Quick Start](getting-started/Quick_Start.md)
3. [Classification](guides/Classification.md)
4. [Distances](reference/Distances.md)
5. [Outputs](reference/Outputs.md)

### Work with multivariate time series

1. [Dataset Representations](data/Dataset_Representations.md)
2. [Data Formats](data/Data_Formats.md)
3. [Readers](data/Readers.md)
4. [Distances](reference/Distances.md)
5. [Eager and Lazy Data](guides/Eager_and_Lazy_Data.md)

### Impute missing values

1. [Missing Values](reference/Missing_Values.md)
2. [Imputation](guides/Imputation.md)
3. [Standardization](guides/Standardization.md)
4. [Outputs](reference/Outputs.md)
5. [Imputed-Only Output](reference/Imputed_Only_Output.md)

### Add a proprietary format or distance

1. [Dataset Representations](data/Dataset_Representations.md)
2. [Custom Readers](extensions/Custom_Readers.md)
3. [Custom Distances](extensions/Custom_Distances.md)
4. [Model Persistence](reference/Model_Persistence.md)
5. [Parallelism and Reproducibility](guides/Parallelism_and_Reproducibility.md)

## Outputs at a glance

Depending on the task and configuration, PFGAP can write:

- saved forest models;
- validation or evaluation predictions;
- structured prediction and OOD CSV files;
- training and evaluation-to-training proximities;
- supervised classification outlier scores;
- isolation scores;
- complete imputed datasets;
- imputed-only Matrix Market output;
- saved standardization statistics; and
- compact experiment-result JSON with repetition summaries and artifact paths.

See [Outputs](reference/Outputs.md) for filenames, schemas, indexing conventions, and availability by workflow.

## Documentation map

```text
docs/
├── index.md
├── getting-started/
│   ├── Installation.md
│   ├── Quick_Start.md
│   └── Configuration.md
├── data/
│   ├── Data_Formats.md
│   ├── Dataset_Representations.md
│   ├── Readers.md
│   └── Writers.md
├── guides/
│   ├── Classification.md
│   ├── Regression.md
│   ├── Imputation.md
│   ├── Outlier_Scoring.md
│   ├── OOD_Scoring.md
│   ├── Standardization.md
│   ├── Eager_and_Lazy_Data.md
│   └── Parallelism_and_Reproducibility.md
├── extensions/
│   ├── Custom_Distances.md
│   └── Custom_Readers.md
└── reference/
    ├── Configuration_Reference.md
    ├── CLI_Reference.md
    ├── Distances.md
    ├── Missing_Values.md
    ├── Outputs.md
    ├── Imputed_Only_Output.md
    └── Model_Persistence.md
```

## Documentation conventions

- Guides explain how to complete a workflow.
- Reference pages define options, identifiers, schemas, and public behavior.
- Data pages define physical formats, logical representations, reading, and writing.
- Extension pages define interfaces, packaging, runtime behavior, and reproducibility requirements.
- Python and direct Java examples use equivalent settings where both interfaces support the workflow.
- Long option catalogs belong in reference pages rather than being duplicated throughout guides.
- Relative links are written so the Markdown sources work in the repository and on the documentation website.
- The Markdown files under `docs/` are the authoritative documentation source.

## Repository links

- [Project README](../README.md)
- [License](../LICENSE)
