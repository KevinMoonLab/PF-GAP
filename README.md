# PFGAP

PFGAP is a Java-based framework and command-line application for Proximity Forest learning and proximity-based analysis on time series and structured data. It extends the Proximity Forest family with generalized proximities, classification and regression, iterative imputation, outlier and out-of-distribution scoring, multivariate and unequal-length data support, flexible dataset readers, and extensible distance functions. 

**[Documentation](https://kevinmoonlab.github.io/PF-GAP/)** · **[Quick Start](https://kevinmoonlab.github.io/PF-GAP/getting-started/quick-start)** .

> **Project status:** PFGAP is under active development toward version 1.0. Compatibility-sensitive workflows should retain the PFGAP revision, Java runtime, preprocessing artifacts, external extension artifacts, and configuration used by an experiment.

## Key capabilities

- Time-series and structured-data classification
- Regression forests and numeric prediction
- Univariate and multivariate observations
- Equal-length and unequal-length series
- Generalized, Breiman, and depth-weighted proximities
- Iterative proximity-based imputation
- Complete and imputed-only output
- Supervised classification outlier scores
- Unsupervised isolation scoring
- Evaluation-time OOD scoring
- Eager and deferred dataset readers
- Delimited, long-form, per-file, HDF5, NPY, Parquet, and custom input sources
- Numeric feature storage using `float32`, `float64`, or automatic source-aware selection
- Standardization with reusable fitted statistics and per-series scopes
- Dense CSV and sparse Matrix Market outputs
- Bounded parallel execution through one application worker budget
- Built-in and custom Java, Python, Maple, and meta distances
- Custom eager and deferred readers from separate JARs
- Saved models with retained training data and reconstructible reader specifications

## Requirements

- Java 21 or later
- Python 3 when using `Application/PF_wrapper.py` or the top-level Python test scripts
- NumPy for Python-helper workflows that require it
- Optional format- or extension-specific dependencies

The Java Vector API is optional. In Python helper calls, set:

```python
use_vector_api=True
```

`PF_wrapper.py` then adds the required `jdk.incubator.vector` module to the Java launch command automatically.

For direct Java execution, add the module explicitly:

```bash
java --add-modules=jdk.incubator.vector \
  -Xmx4g \
  -jar Application/PFGAP.jar \
  -use_vector_api=true \
  ...
```

## Repository layout

The principal top-level structure is:

```text
PFGAP/
├── Application/
│   ├── PFGAP.jar
│   └── PF_wrapper.py
├── PFGAP/
│   ├── lib/
│   └── src/
├── docs/
│   ├── data/
│   ├── extensions/
│   ├── getting-started/
│   ├── guides/
│   ├── reference/
│   └── index.md
├── tests/
├── website/
├── LICENSE
└── README.md
```

- `Application/` contains the distributable JAR and Python helper.
- `PFGAP/src/` contains the Java source tree.
- `PFGAP/lib/` contains external Java libraries.
- `docs/` contains the authoritative user documentation, organized into getting-started, data, guides, extensions, and reference sections.
- `tests/` contains top-level Python scripts used to exercise application workflows and integrations.
- `website/` contains files which render the canonical content from `docs/` as a static documentation website.

## Interfaces

PFGAP supports two primary user interfaces.

### Python helper

`Application/PF_wrapper.py` validates Python-facing arguments, constructs the Java command, and launches `PFGAP.jar`.

Use it from:

- Python scripts;
- Jupyter or Marimo notebooks;
- experiment pipelines; or
- applications that coordinate work in Python.

The algorithms and application runtime remain implemented in Java.

### Direct Java command line

`Application/PFGAP.jar` accepts application options in this form:

```text
-name=value
```

The direct interface is suitable for shell scripts, Java-oriented deployments, scheduled jobs, and environments where Python is not needed.

## Quick start

The examples below train a classification forest, evaluate a labeled test set, write predictions, and save the model.

The paths are illustrative. Select a reader and separators that match the actual data layout.

### Python

Run from `Application/`, or otherwise place `PF_wrapper.py` and `PFGAP.jar` together in the working directory expected by the helper.

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    exists_testlabels=True,
    reader_type="NUMERIC_DELIMITED",
    forest_mode="classification",
    data_dimension=1,
    numeric_data=True,
    numeric_storage="float64",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    distances=["euclidean", "dtw", "erp"],
    num_trees=101,
    r=5,
    seed=42,
    num_workers=4,
    use_vector_api=False,
    return_predictions=True,
    save_model=True,
    model_name="classification_model",
    output_directory="../output/classification",
)

if status != 0:
    raise SystemExit(status)
```

Set `use_vector_api=True` when the selected operations should use supported Vector API implementations. The helper will add the incubator module to the Java command.

### Direct Java

From the repository root:

```bash
mkdir -p output/classification

java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -test=data/test.csv \
  -exists_testlabels=true \
  -reader_type=NUMERIC_DELIMITED \
  -forest_mode=classification \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=float64 \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -distances=[euclidean,dtw,erp] \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -get_predictions=true \
  -savemodel=true \
  -modelname=classification_model \
  -out=output/classification/
```

## Main workflows

### Classification

Train a classification forest, evaluate labeled or unlabeled data, save models, and export ordinary or structured predictions.

See [Classification](docs/guides/Classification.md).

### Regression

Train regression forests with numeric targets and export aggregate predictions, residuals, absolute errors, and structured tree-prediction dispersion.

See [Regression](docs/guides/Regression.md).

### Imputation

Initialize missing values, iteratively update them through forest proximities, and write either complete imputed datasets or imputed-only Matrix Market output.

See:

- [Imputation](docs/guides/Imputation.md)
- [Missing Values](docs/reference/Missing_Values.md)
- [Imputed-Only Output](docs/reference/Imputed_Only_Output.md)

### Outlier and isolation scoring

PFGAP separates:

- supervised classification training outlier scores; and
- unsupervised isolation-forest scoring.

See [Outlier Scoring](docs/guides/Outlier_Scoring.md).

### OOD scoring

A forest trained with retained split-distance summaries can score later evaluation observations relative to training split support.

See [OOD Scoring](docs/guides/OOD_Scoring.md).

### Proximities

PFGAP supports generalized, Breiman, and depth-weighted proximity definitions. It can write training-to-training and evaluation-to-training matrices in dense CSV or sparse Matrix Market form.

See [Outputs](docs/reference/Outputs.md).

## Data and I/O

PFGAP separates physical file layout from logical observation representation.

- [Data Formats](docs/data/Data_Formats.md) describes supported external formats.
- [Dataset Representations](docs/data/Dataset_Representations.md) defines one-dimensional and two-dimensional in-memory representations.
- [Readers](docs/data/Readers.md) lists implemented reader types and their option requirements.
- [Writers](docs/data/Writers.md) describes complete dataset output and mirror-writer support.
- [Eager and Lazy Data](docs/guides/Eager_and_Lazy_Data.md) explains materialized and deferred reader implementations.

## Standardization

PFGAP supports:

```text
none
z_score
mean_center
min_max
```

Available scopes are:

```text
global
per_dimension
per_series
per_series_per_dimension
```

Reusable statistics can be saved and restored for compatible scopes. Complete and imputed-only numeric outputs can be returned to the original feature scale.

See [Standardization](docs/guides/Standardization.md).

## Distances and extensions

The built-in distance registry includes vector, elastic, multivariate independent, multivariate dependent, missing-aware, graph, interoperability, and meta distances.

See [Distances](docs/reference/Distances.md) for exact case-sensitive identifiers and compatibility guidance.

### Custom distances

Custom distances can be implemented in:

- Java;
- Python;
- Maple; or
- pretrained-model and file-backed meta integrations.

Java distances can be distributed in a separate standard or fat JAR.

See [Custom Distances](docs/extensions/Custom_Distances.md).

### Custom readers

A separate JAR can implement a per-file `CustomSeriesReader` used by both eager and deferred custom reader types. A project-specific extension can also implement a direct whole-dataset reader and, when needed, a reconstructible deferred reader.

See [Custom Readers](docs/extensions/Custom_Readers.md).

## Parallel execution and reproducibility

Configure the worker budget with:

```text
-num_workers=1
```

for sequential execution, a positive value greater than one for bounded parallel execution, or:

```text
-num_workers=-1
```

for all processors visible to the JVM.

Set a seed for reproducible randomized behavior:

```text
-seed=42
```

Reproducibility also depends on data ordering, reader configuration, distances, preprocessing, external extensions, runtime versions, and worker settings.

See [Parallelism and Reproducibility](docs/guides/Parallelism_and_Reproducibility.md).

## Model persistence

A saved model is a Java serialization stream containing:

1. the fitted forest;
2. the retained training dataset; and
3. a snapshot of model-related application state.

Custom JARs, scripts, pretrained models, source data used by deferred readers, and other external dependencies remain external artifacts and must be retained with the experiment.

Load models only from trusted sources.

See [Model Persistence](docs/reference/Model_Persistence.md).

## Outputs

Depending on the workflow and configuration, PFGAP can write:

- ordinary classification or regression predictions;
- enhanced per-instance prediction details;
- OOD results;
- supervised outlier scores;
- isolation scores and diagnostics;
- dense or sparse proximities;
- complete imputed datasets;
- imputed-only Matrix Market values;
- standardization statistics;
- saved models; and
- `experiment_results.json` with repetition summaries and artifact paths.

See [Outputs](docs/reference/Outputs.md) for filenames, schemas, indexing, CSV behavior, repetition suffixes, and availability by workflow.

## Tests

The top-level `tests/` directory contains Python scripts for exercising current PFGAP workflows and integrations through the application interface.

These scripts complement the Java source tests and are useful for end-to-end checks involving:

- `PF_wrapper.py`;
- reader and format behavior;
- custom extensions;
- model training and evaluation; and
- generated artifacts.

Test scripts can depend on local data, optional runtimes, generated JARs, or environment-specific paths. Review each script before running it.

## Documentation

Start with [PFGAP Documentation](docs/index.md).

The documentation is organized into:

- `docs/getting-started/` for installation, first use, and configuration;
- `docs/data/` for formats, representations, readers, and writers;
- `docs/guides/` for task-oriented workflows;
- `docs/extensions/` for custom distance and reader development; and
- `docs/reference/` for options, registries, schemas, outputs, and persistence contracts.

### Getting started

- [Installation](docs/getting-started/Installation.md)
- [Quick Start](docs/getting-started/Quick_Start.md)
- [Configuration](docs/getting-started/Configuration.md)

### Data

- [Data Formats](docs/data/Data_Formats.md)
- [Dataset Representations](docs/data/Dataset_Representations.md)
- [Readers](docs/data/Readers.md)
- [Writers](docs/data/Writers.md)

### Guides

- [Classification](docs/guides/Classification.md)
- [Regression](docs/guides/Regression.md)
- [Imputation](docs/guides/Imputation.md)
- [Outlier Scoring](docs/guides/Outlier_Scoring.md)
- [OOD Scoring](docs/guides/OOD_Scoring.md)
- [Standardization](docs/guides/Standardization.md)
- [Eager and Lazy Data](docs/guides/Eager_and_Lazy_Data.md)
- [Parallelism and Reproducibility](docs/guides/Parallelism_and_Reproducibility.md)

### Extensions

- [Custom Distances](docs/extensions/Custom_Distances.md)
- [Custom Readers](docs/extensions/Custom_Readers.md)

### Reference

- [Configuration Reference](docs/reference/Configuration_Reference.md)
- [CLI Reference](docs/reference/CLI_Reference.md)
- [Distances](docs/reference/Distances.md)
- [Missing Values](docs/reference/Missing_Values.md)
- [Outputs](docs/reference/Outputs.md)
- [Imputed-Only Output](docs/reference/Imputed_Only_Output.md)
- [Model Persistence](docs/reference/Model_Persistence.md)

## Research citation

If you use PFGAP in research, cite the publication relevant to the functionality used.

- Ben Shaw, Jake S. Rhodes, Soukaina Filali Boubrahimi, and Kevin R. Moon. **Forest Proximities for Time Series.** IntelliSys 2025. [arXiv:2410.03098](https://arxiv.org/abs/2410.03098)
- Ben Shaw, Adam Rustad, Sofia Pelagalli Maia, Jake S. Rhodes, and Kevin R. Moon. **The Generalized Proximity Forest.** ACDSA 2026. [arXiv:2511.19487](https://arxiv.org/abs/2511.19487)

## License

See [LICENSE](LICENSE).
