# Eager and Lazy Data

This guide explains how to choose between eager and lazy data access in PFGAP. Access behavior is selected by `reader_type`: eager and lazy implementations have distinct `ReaderType` names and share the same logical dataset contracts.

For the full reader list and format-specific options, see [Readers](../data/Readers.md). For in-memory observation shapes, see [Dataset Representations](../data/Dataset_Representations.md).

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Configuration](../getting-started/Configuration.md) for path and option conventions.
3. Read [Data Formats](../data/Data_Formats.md) for physical file layouts.
4. Consult [Readers](../data/Readers.md) for the authoritative reader names and requirements.

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Eager and lazy access

PFGAP supports two data-access models.

### Eager access

An eager reader parses and materializes observations while constructing the dataset.

Eager access is appropriate when:

- the dataset fits comfortably in memory;
- observations will be accessed repeatedly;
- the workflow requires mutable materialized observations;
- complete imputed-data output is required;
- the selected format has only an eager reader; or
- predictable in-memory access is more important than reducing startup materialization.

### Lazy access

A lazy reader discovers logical observations and stores `LazySeriesRef` objects in the dataset. The corresponding `LazySeriesReader` materializes an observation when PFGAP requests it.

Lazy access is appropriate when:

- observations are large;
- full up-front materialization is undesirable;
- the source supports stable observation-level addressing;
- only part of the dataset may be visited;
- startup time or initial memory pressure matters; or
- the selected format has a dedicated lazy reader.

Lazy access changes when an observation is parsed. After materialization, the observation must satisfy the same representation contract expected by distances and downstream workflows.

## Select the access model through `reader_type`

Choose an eager or lazy `ReaderType` explicitly.

For example, eager and lazy numeric per-file delimited readers are:

```text
PER_FILE_NUMERIC_DELIMITED
LAZY_PER_FILE_NUMERIC_DELIMITED
```

Eager and lazy numeric per-file Parquet readers are:

```text
PER_FILE_NUMERIC_PARQUET
LAZY_PER_FILE_NUMERIC_PARQUET
```

Eager and lazy NPY readers are:

```text
NPY
LAZY_NPY
```

Custom per-file readers are:

```text
PER_FILE_CUSTOM
LAZY_PER_FILE_CUSTOM
```

The available pairings depend on the format. [Readers](../data/Readers.md) lists every implemented reader and its required companion options.

## Eager delimited example

A numeric delimited reader eagerly materializes one observation per source row:

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    reader_type="NUMERIC_DELIMITED",
    forest_mode="classification",
    data_dimension=1,
    numeric_data=True,
    numeric_storage="float64",
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    output_directory="../output/eager_run",
)

if status != 0:
    raise SystemExit(status)
```

Direct Java form:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -reader_type=NUMERIC_DELIMITED \
  -forest_mode=classification \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=float64 \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -out=output/eager_run/
```

## Lazy per-file delimited example

A lazy per-file reader discovers files immediately and defers parsing each file until its observation is requested:

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train-series",
    train_labels="../data/train_labels.csv",
    reader_type="LAZY_PER_FILE_NUMERIC_DELIMITED",
    file_pattern="series_{instance:05d}.csv",
    forest_mode="classification",
    data_dimension=2,
    numeric_data=True,
    numeric_storage="float32",
    entry_separator=",",
    file_has_header=False,
    output_directory="../output/lazy_run",
)

if status != 0:
    raise SystemExit(status)
```

Direct Java form:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train-series \
  -train_labels=data/train_labels.csv \
  -reader_type=LAZY_PER_FILE_NUMERIC_DELIMITED \
  '-file_pattern=series_{instance:05d}.csv' \
  -forest_mode=classification \
  -is2D=true \
  -isNumeric=true \
  -numeric_storage=float32 \
  -entry_separator=, \
  -csv_has_header=false \
  -out=output/lazy_run/
```

Quote pattern arguments in a shell when they contain wildcard characters or other shell metacharacters.

## Per-file datasets

In a per-file dataset, each discovered file represents one observation.

A typical directory is:

```text
train-series/
├── series_00000.csv
├── series_00001.csv
├── series_00002.csv
└── series_00003.csv
```

A matching pattern is:

```text
series_{instance:05d}.csv
```

Per-file discovery extracts the numeric placeholder and sorts observations deterministically by that number. Separate labels must use the same observation order.

The file set must remain stable while PFGAP is using the dataset. Do not add, remove, rename, or rewrite observation files during a run.

## Per-file representation contract

Per-file readers preserve the dimension axis. They return two-dimensional observations:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate per-file series is still two-dimensional:

```text
float[1][time]
double[1][time]
Object[1][time]
```

Per-file readers do not return the one-dimensional vector forms used by tabular data or ordinary univariate readers.

This restriction is specific to per-file readers. General readers may produce supported one-dimensional observations such as `float[]`, `double[]`, or `Object[]`.

## Lazy NPY

`LAZY_NPY` uses shared-array lazy access rather than one file per observation.

```python
status = PF.train(
    train_file="../data/train.npy",
    reader_type="LAZY_NPY",
    forest_mode="classification",
    numeric_data=True,
    numeric_storage="auto",
    output_directory="../output/lazy_npy_run",
)
```

Axis `0` is the observation axis. Materialized observations preserve supported `float32` or `float64` storage according to the NPY source and `numeric_storage` configuration.

## Lazy Parquet

Lazy per-file Parquet readers include:

```text
LAZY_PER_FILE_PARQUET
LAZY_PER_FILE_NUMERIC_PARQUET
```

Example:

```python
status = PF.train(
    train_file="../data/train-series",
    train_labels="../data/train_labels.csv",
    reader_type="LAZY_PER_FILE_NUMERIC_PARQUET",
    file_pattern="series_{instance:05d}.parquet",
    feature_columns=["x", "y", "z"],
    time_column="time",
    forest_mode="classification",
    data_dimension=2,
    numeric_data=True,
    numeric_storage="float32",
    output_directory="../output/lazy_parquet_run",
)
```

Each materialized observation is dimension-major. Feature projection and numeric storage are retained in the reader's reconstruction specification.

## Lazy observation references

A `LazySeriesRef` identifies one logical observation. It contains:

- the lazy reader registry key;
- the observation index; and
- the file or other source locator used by the materializer.

The dataset stores these references in observation order. When a distance, standardizer, predictor, or other consumer requests a value, PFGAP resolves the reference through the registered reader.

Consumers use the dataset interface rather than opening source files directly.

## Reader reconstruction

A `LazySeriesReaderSpec` stores the serializable settings required to reconstruct a runtime lazy reader. `LazySeriesReaderFactory` creates that runtime reader, and `AppContext` maintains the active registry.

A saved model stores reader specifications and observation references rather than live readers, open files, class loaders, mappings, or caches. Loading the model reconstructs the required readers from their saved specifications.

The source paths represented by those specifications must remain available when the model is used.

## Resource ownership

Lazy readers may retain:

- reusable Parquet projections;
- memory mappings;
- plugin instances;
- class loaders;
- shared decoders; or
- caches.

Resource-owning readers implement `AutoCloseable`. Decorators own their delegates, so closing the outer reader closes the full chain.

Application workflows manage readers created through the built-in factories and registry. Direct Java integrations must keep readers open while their references are in use and close them when the workflow finishes.

Replacing or clearing a registered reader closes the previous resource-owning reader.

## Concurrency

Lazy observations can be materialized concurrently when PFGAP uses multiple workers:

```python
num_workers=4
```

Direct Java form:

```text
-num_workers=4
```

Increasing the worker count can increase simultaneous file access, materialization, cache pressure, and temporary memory use.

Built-in readers provide their required concurrent-access behavior. A custom plugin should be marked thread-safe only when one plugin instance can safely serve concurrent reads. Otherwise, the custom runtime adapter serializes plugin invocation.

See [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md).

## Caching and repeated access

A lazy reader may cache decoded observations or reusable source structures. The practical memory cost depends on:

- observation size;
- access frequency;
- reader implementation;
- retained materializations;
- standardization behavior;
- distance workspaces; and
- worker count.

Lazy access avoids mandatory full-dataset materialization. It does not guarantee constant memory use.

When the complete dataset fits comfortably in memory and observations are revisited frequently, an eager reader can be faster because it avoids repeated resolution and parsing.

## Standardization with eager data

Eager readers parse an observation and apply prepared standardization before adding it to the dataset.

Reusable `global` and `per_dimension` statistics are fitted from compatible training data or loaded from a statistics file. The fitted training statistics are reused for test and later evaluation data.

Per-series scopes calculate local transformation parameters for each realized observation.

See [Standardization](Standardization.md).

## Standardization with lazy data

Lazy standardization is applied after an observation is materialized. PFGAP uses a `StandardizingLazySeriesReader` decorator so the underlying reader returns raw values and the standardization layer transforms the materialized observation exactly once.

Supported standardized lazy representations include:

```text
float[]
double[]
float[][]
double[][]
```

For reusable scopes:

```text
global
per_dimension
```

the lazy reader uses prepared training statistics.

For local scopes:

```text
per_series
per_series_per_dimension
```

parameters are calculated from the realized series.

Custom reader plugins must return raw values and must not apply PFGAP standardization themselves.

## Missing values and imputation

A reader can materialize an observation containing supported missing values without necessarily supporting iterative imputation or complete rewritten dataset output.

Imputation requires a compatible materialized and mutable workflow. Complete imputed output also requires a mirror writer for the source reader family.

Consult:

- [Readers](../data/Readers.md) for reader-specific support;
- [Imputation](Imputation.md) for the imputation workflow;
- [Missing Values](../reference/Missing_Values.md) for missing-value semantics; and
- [Writers](../data/Writers.md) for complete dataset output support.

Imputed-only Matrix Market output has its own numeric output contract. See [Imputed-Only Output](../reference/Imputed_Only_Output.md).

## Outputs and observation order

Predictions, enhanced results, scores, and proximities follow the dataset's stable instance ordering.

For per-file readers, preserve the source directory and naming pattern needed to relate each output row to its source observation.

Complete rewritten dataset output depends on the available mirror writer. Predictions and scoring artifacts use their dedicated output writers and are not limited to the complete-dataset writer formats.

See [Outputs](../reference/Outputs.md).

## Training and later evaluation

A model trained with one reader can be evaluated with another reader when both produce a compatible logical representation.

Compatibility includes:

- one-dimensional or two-dimensional shape;
- numeric or generic value type;
- primitive numeric storage where required;
- feature and dimension order;
- time order;
- variable-length behavior supported by the selected distances;
- standardization contract; and
- missing-value handling.

The physical source format does not need to be identical.

A model that stores lazy training references also stores reconstructible training-reader specifications where available. The referenced source data must remain accessible for operations that use the retained training dataset.

See [Model Persistence](../reference/Model_Persistence.md).

## Custom eager and lazy readers

PFGAP supports custom reader extensions at two levels.

### Custom per-file series plugin

A separate JAR can provide a class implementing `CustomSeriesReader`. The descriptor is:

```text
javareader:/absolute/path/to/readers.jar:example.reader.SeriesReader
```

Use:

```text
PER_FILE_CUSTOM
```

for eager materialization, or:

```text
LAZY_PER_FILE_CUSTOM
```

for deferred materialization.

The same plugin can support both modes. The built-in per-file coordinators handle discovery, ordering, references, validation, optional standardization, and resource lifecycle.

The plugin must have an accessible no-argument constructor and return a valid two-dimensional per-file representation.

### Custom whole-dataset reader

When discovery or grouping is proprietary, implement `DatasetReader` directly. A direct custom reader can return supported one-dimensional or two-dimensional observations according to its own source contract.

A custom whole-dataset lazy source also provides a `LazySeriesReader` and stable references. Saved-model reconstruction requires a serializable specification and a corresponding factory branch.

See [Custom Readers](../extensions/Custom_Readers.md) for full interfaces and examples.

## Choose eager access when

Use an eager reader when:

- the dataset fits comfortably in memory;
- observations will be revisited frequently;
- iterative imputation is required;
- complete rewritten output is required and a mirror writer exists;
- the selected format has no lazy implementation; or
- simple resource and performance behavior is preferred.

## Choose lazy access when

Use a lazy reader when:

- full materialization is undesirable;
- observations are large;
- access can be sparse;
- the source has stable observation locators;
- startup time and initial memory use matter; and
- the requested downstream operations are compatible with the reader.

## Common problems

### An eager reader was selected unintentionally

Use the corresponding `LAZY_...` reader type listed in [Readers](../data/Readers.md).

### A lazy implementation is unavailable

Choose an implemented lazy reader for the format or use its eager reader.

### No files match the pattern

Check the data directory, numeric placeholder, wildcard fragments, working directory, and shell quoting.

### Labels do not align with files

Ensure that label order matches deterministic file-discovery order.

### A per-file reader returns a one-dimensional array

Return a two-dimensional dimension-major observation. A univariate per-file series must use shape `[1][time]`.

### Materialization fails only during training

Lazy discovery can succeed before any observation is parsed. Check the source file, feature projection, data type, and representation returned when the failing reference is materialized.

### Memory use grows with a lazy reader

Concurrent materialization, caches, retained observations, distance workspaces, models, and output calculations still consume memory. Reduce the worker count or use an eager reader when the dataset fits in memory.

### Lazy execution is slower

Repeated resolution and parsing can cost more than one eager load. Use eager access for repeatedly visited data that fit in memory.

### Standardization is applied twice

Return raw values from the base reader or plugin. Let PFGAP's reader-side standardization layer apply the configured transformation.

### Complete imputed output is unavailable

Use a source reader family with a registered mirror writer, or request imputed-only output when that output contract meets the workflow requirements.

### Source files cannot be modified after a run

Ensure the workflow has completed and resource-owning readers have been closed.

### A saved model cannot reconstruct a reader

Ensure the model contains a reconstructible reader specification and that the referenced source paths, custom JARs, and dependencies are available.

## Related documentation

- [Readers](../data/Readers.md)
- [Data Formats](../data/Data_Formats.md)
- [Dataset Representations](../data/Dataset_Representations.md)
- [Writers](../data/Writers.md)
- [Standardization](Standardization.md)
- [Imputation](Imputation.md)
- [Parallelism and Reproducibility](Parallelism_and_Reproducibility.md)
- [Model Persistence](../reference/Model_Persistence.md)
- [Custom Readers](../extensions/Custom_Readers.md)
