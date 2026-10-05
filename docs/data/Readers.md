# Readers

PFGAP provides a unified reader subsystem for loading tabular data, time series, array formats, per-file datasets, and user-defined formats into `ListObjectDataset`. Readers are configured with `ReaderOptions`, selected by `ReaderType`, and normally constructed through `DatasetReaderFactory`.

This document describes the public reader model, supported in-memory representations, eager and lazy loading, numeric storage, standardization, built-in formats, and custom Java reader extensions.

For related details, see:

- [Dataset Representations](Dataset_Representations.md) for the complete in-memory data contract.
- [Data Formats](Data_Formats.md) for external file layouts and format-level conventions.

## 1. Core reader model

The principal reader classes are:

- `DatasetReader`: reads a configured source and returns a `ListObjectDataset`.
- `ReaderOptions`: contains common and format-specific reader settings.
- `ReaderType`: identifies the reader implementation selected by the factory.
- `DatasetReaderFactory`: constructs the appropriate `DatasetReader`.
- `ListObjectDataset`: stores labels and eager observations or lazy references.

Typical eager loading follows this pattern:

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.NUMERIC_DELIMITED)
        .setDataPath("data/train.csv")
        .setEntrySeparator(",")
        .setHasHeader(true)
        .setNumericStorageType(NumericStorageType.FLOAT32);

DatasetReader reader = DatasetReaderFactory.create(options);
ListObjectDataset dataset = reader.read();
```

Lazy readers use the same factory-facing pattern. Their `read()` methods discover observations and create references without materializing every observation immediately.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.LAZY_PER_FILE_NUMERIC_PARQUET)
        .setDataPath("data/train")
        .setFilePattern("series_{instance:05d}.parquet")
        .setFeatureColumns(List.of("x", "y", "z"))
        .setNumericStorageType(NumericStorageType.FLOAT32);

ListObjectDataset dataset =
        DatasetReaderFactory.create(options).read();
```

## 2. Observation representations

### 2.1 One-dimensional observations

PFGAP uses the same Java array types for one-dimensional tabular observations and one-dimensional ordered sequences. The meaning of the axis is determined by the reader contract, dataset metadata, selected distance, and workflow.

For tabular data, an array index identifies a feature or dimension:

```text
float[dimension]
double[dimension]
Object[dimension]
```

Examples include:

```java
float[] numericRow = {
        32.0f,
        1.82f,
        78.5f
};

Object[] genericRow = {
        "blue",
        "medium",
        12
};
```

Tabular workflows normally require every observation to use the same feature count and feature ordering.

For a univariate time series or another ordered sequence, an array index identifies time or sequence position:

```text
float[time]
double[time]
Object[time]
```

For example:

```java
double[] sequence = {
        1.2,
        1.4,
        1.1,
        0.8
};
```

Sequential workflows may support unequal observation lengths. The Java types remain `float[]`, `double[]`, and `Object[]`; the array type by itself does not distinguish tabular features from ordered positions.

### 2.2 Two-dimensional observations

Multivariate observations use dimension-major storage:

```text
float[dimension][position]
double[dimension][position]
Object[dimension][position]
```

For multivariate time series, the second axis is time:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

Each outer-array entry is one dimension or feature. Each inner array contains the values for that dimension.

```java
double[][] multivariateSeries = {
        {1.0, 1.1, 1.2},
        {5.0, 5.2, 5.1}
};
```

Recognized standard matrices are rectangular. Every dimension must be non-null and have the same length.

### 2.3 Per-file representations

Per-file readers always preserve the dimension axis. Their standard output representations are therefore:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate per-file time series remains two-dimensional:

```text
float[1][time]
double[1][time]
Object[1][time]
```

A custom per-file reader must not return `float[]`, `double[]`, or `Object[]`. Eager and lazy custom per-file paths validate this rule.

This restriction applies specifically to per-file readers. General and non-per-file readers may produce supported one-dimensional tabular or sequential observations.

### 2.4 Generic and proprietary representations

Generic readers use `Object[]` and `Object[][]` for categorical, textual, mixed, or otherwise nonprimitive values.

Custom readers may also return a proprietary representation when:

- Built-in standardization is disabled.
- The configured distance or downstream consumer understands that representation.
- The representation is not one of the standard one-dimensional arrays prohibited by the per-file contract.

### 2.5 Missing values

Numeric missing values are represented with the matching primitive `NaN` value:

```java
Float.NaN
Double.NaN
```

Generic missing values are normally represented by `null`.

Boxed numeric arrays are not standard numeric representations. Numeric readers should produce primitive `float` or `double` arrays.

## 3. Numeric storage policy

`NumericStorageType` controls the requested primitive representation:

- `AUTO`: use the reader's natural or default numeric type.
- `FLOAT32`: return primitive `float` observations.
- `FLOAT64`: return primitive `double` observations.

Typed binary formats can preserve a supported source type under `AUTO`. Untyped text readers generally use `FLOAT64` when no source type exists to preserve.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.NUMERIC_LONG_FORMAT_PARQUET)
        .setDataPath("data/series.parquet")
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

The storage choice is retained in lazy reconstruction specifications so that a saved model recreates the same materialization behavior.

## 4. Generic and optimized numeric readers

Several formats have both generic and optimized numeric readers.

Use a generic reader when observations may contain:

- Strings or categories
- Mixed Java values
- Generic object data
- A proprietary representation

Use the numeric variant when all selected features are numeric and primitive output is desired.

Optimized numeric readers avoid boxed intermediate storage where practical and support `NumericStorageType`. They should normally be preferred for large numeric datasets.

## 5. Delimited readers

### 5.1 `DELIMITED`

`DELIMITED` is the general CSV, TSV, or text reader. It supports generic and numeric input, optional headers, labels, missing values, and row-encoded arrays.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.DELIMITED)
        .setDataPath("data/train.csv")
        .setLabelPath("data/train-labels.csv")
        .setEntrySeparator(",")
        .setArraySeparator(":")
        .setHasHeader(true)
        .set2D(true)
        .setNumeric(false);
```

A wide row can represent either:

- A fixed-width tabular observation, such as `double[dimension]` or `Object[dimension]`.
- An equal-length univariate sequence, such as `double[time]` or `Object[time]`.

The file schema and configured workflow determine the interpretation.

### 5.2 `NUMERIC_DELIMITED`

`NUMERIC_DELIMITED` is the optimized path for numeric tabular data and univariate sequences. Each input row is one observation. It emits primitive numeric arrays and supports the numeric storage policy.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.NUMERIC_DELIMITED)
        .setDataPath("data/train.csv")
        .setEntrySeparator(",")
        .setHasHeader(true)
        .setTargetColumnIsFirst(true)
        .setNumericStorageType(NumericStorageType.FLOAT64);
```

## 6. TS files

`TS` reads `.ts` files such as those used by the UEA and UCR time-series archives. It supports:

- Univariate observations
- Multivariate observations
- Embedded labels or regression targets
- Numeric or generic values
- Missing values
- Unequal-length observations where supported

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.TS)
        .setDataPath("data/BasicMotions_TRAIN.ts")
        .setNumeric(true)
        .setHasMissingValues(false)
        .setRegression(false);
```

## 7. Long-format data

Long-format readers group multiple source rows into one observation. A typical source contains an instance identifier, optional time column, feature columns, and optional label columns.

```text
id,time,x,y,label
A,0,1.0,5.0,class1
A,1,1.2,5.1,class1
B,0,3.0,7.0,class2
```

The resulting observations are dimension-major matrices.

### 7.1 `LONG_FORMAT_DELIMITED`

Use `LONG_FORMAT_DELIMITED` for generic long-format text data.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.LONG_FORMAT_DELIMITED)
        .setDataPath("data/series.csv")
        .setEntrySeparator(",")
        .setHasHeader(true)
        .setIdColumn("id")
        .setTimeColumn("time")
        .setFeatureColumns(List.of("x", "y"))
        .setLabelColumns(List.of("label"));
```

If no time column is configured, source row order is preserved within each instance.

### 7.2 `NUMERIC_LONG_FORMAT`

Use `NUMERIC_LONG_FORMAT` for optimized numeric long-format delimited data.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.NUMERIC_LONG_FORMAT)
        .setDataPath("data/series.csv")
        .setEntrySeparator(",")
        .setHasHeader(true)
        .setIdColumn("id")
        .setTimeColumn("time")
        .setFeatureColumns(List.of("x", "y"))
        .setLabelColumns(List.of("label"))
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

### 7.3 `LONG_FORMAT_PARQUET`

Use `LONG_FORMAT_PARQUET` for generic long-format Parquet data.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.LONG_FORMAT_PARQUET)
        .setDataPath("data/series.parquet")
        .setIdColumn("id")
        .setTimeColumn("time")
        .setFeatureColumns(List.of("x", "y"))
        .setLabelColumns(List.of("label"));
```

### 7.4 `NUMERIC_LONG_FORMAT_PARQUET`

Use `NUMERIC_LONG_FORMAT_PARQUET` for the optimized primitive numeric Parquet path.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.NUMERIC_LONG_FORMAT_PARQUET)
        .setDataPath("data/series.parquet")
        .setIdColumn("id")
        .setTimeColumn("time")
        .setFeatureColumns(List.of("x", "y"))
        .setLabelColumns(List.of("label"))
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

## 8. Per-file datasets

In a per-file dataset, each file represents one observation. PFGAP discovers matching files, extracts the numeric placeholder from each filename, and sorts deterministically by that number.

Example directory:

```text
series_00000.csv
series_00001.csv
series_00002.csv
```

Example pattern:

```text
series_{instance:05d}.csv
```

Patterns contain exactly one numeric placeholder. Common forms include:

```text
{num}
{instance}
{run:05d}
```

Literal pattern fragments may also contain `*` and `?` wildcards. A directory source requires a pattern. A single regular file does not.

### 8.1 Delimited per-file readers

Available types are:

- `PER_FILE_DELIMITED`
- `PER_FILE_NUMERIC_DELIMITED`
- `LAZY_PER_FILE_DELIMITED`
- `LAZY_PER_FILE_NUMERIC_DELIMITED`

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.PER_FILE_NUMERIC_DELIMITED)
        .setDataPath("data/train")
        .setFilePattern("series_{instance:05d}.csv")
        .setEntrySeparator(",")
        .setHasHeader(true)
        .setFeatureColumns(List.of("x", "y"))
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

Change the type to `LAZY_PER_FILE_NUMERIC_DELIMITED` to discover files immediately but defer parsing each file until its observation is requested.

### 8.2 Parquet per-file readers

Available types are:

- `PER_FILE_PARQUET`
- `PER_FILE_NUMERIC_PARQUET`
- `LAZY_PER_FILE_PARQUET`
- `LAZY_PER_FILE_NUMERIC_PARQUET`

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.LAZY_PER_FILE_NUMERIC_PARQUET)
        .setDataPath("data/train")
        .setFilePattern("series_{instance:05d}.parquet")
        .setTimeColumn("time")
        .setFeatureColumns(List.of("x", "y"))
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

Per-file readers produce two-dimensional observations even when only one feature is selected.

## 9. NPY readers

### 9.1 `NPY`

`NPY` eagerly reads supported NumPy `.npy` arrays. It supports primitive `float32` and `float64` data and uses axis 0 as the observation axis.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.NPY)
        .setDataPath("data/train.npy")
        .setNumericStorageType(NumericStorageType.AUTO);
```

A materialized one-dimensional observation may be interpreted as a tabular feature vector or a univariate sequence.

### 9.2 `LAZY_NPY`

`LAZY_NPY` creates observation references into a shared `.npy` array and materializes an observation only when requested.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.LAZY_NPY)
        .setDataPath("data/train.npy")
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

Lazy NPY uses shared-array lazy access rather than one file per observation.

## 10. HDF5

`HDF5` reads an array-like HDF5 dataset and optionally a separate label dataset.

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.HDF5)
        .setDataPath("data/train.h5")
        .setHdf5DatasetPath("/features")
        .setHdf5LabelDatasetPath("/labels")
        .setNumericStorageType(NumericStorageType.FLOAT32);
```

The selected HDF5 dataset must have a supported array shape and compatible primitive type.

## 11. Lazy reader architecture

Lazy datasets store `LazySeriesRef` objects instead of materialized observations. A reference contains:

- The lazy reader registry key
- The observation index
- The file or other source locator used by the materializer

A `LazySeriesReader` converts one reference into one observation. `LazySeriesReaderSpec` stores the serializable configuration needed to reconstruct that runtime reader. `LazySeriesReaderFactory` performs reconstruction, and `AppContext` maintains the runtime registry.

A saved model stores specifications and references, not live readers, open files, class loaders, mappings, or caches. When the model is loaded in another JVM, PFGAP reconstructs the registered lazy readers from their specifications.

Lazy dataset construction normally sets dataset length to zero because observation lengths are not known until materialization and may differ.

### 11.1 Registration and reconstruction

A lazy coordinator normally:

1. Discovers logical observations.
2. Creates one `LazySeriesRef` for each observation.
3. Creates a serializable `LazySeriesReaderSpec`.
4. Registers the specification with `AppContext`.
5. Returns a dataset containing the references.

The factory reconstructs the runtime reader from the specification when the reader is first registered or when a saved model is loaded.

### 11.2 Resource ownership

Lazy readers may retain reusable projections, mapped files, plugin instances, class loaders, or caches. Resource-owning readers implement `AutoCloseable`.

Decorators own their delegates. Closing the outer reader closes the complete chain.

## 12. Standardization

Reader-side standardization uses prepared training statistics. Readers do not fit independent test-data statistics.

For eager readers, standardization occurs after an observation is parsed and before it is added to the dataset. For lazy readers, a `StandardizingLazySeriesReader` decorator applies prepared statistics after materialization.

The decorator calls the shared optimized entry point directly:

```java
Standardizer.transformInstanceInPlace(
        observation,
        standardizationStats
);
```

This preserves scalar, parallel, and Vector API dispatch in `Standardizer` without copying, boxing, transposing, or flattening the observation.

Standardization supports:

```text
float[]
double[]
float[][]
double[][]
```

This includes one-dimensional tabular observations and one-dimensional sequences. Per-file readers still enforce their narrower two-dimensional rule before standardization.

```java
ReaderOptions testOptions = trainOptions.copy()
        .setDataPath("data/test")
        .setTest(true)
        .setStandardizationStats(trainingStats);
```

Apply standardization exactly once. A custom plugin must return raw values and must not apply PFGAP standardization itself.

## 13. Custom Java readers

PFGAP exposes two custom extension levels:

1. `CustomSeriesReader` materializes one observation. The built-in eager and lazy custom coordinators use this contract for per-file datasets.
2. `DatasetReader` controls discovery and assembly of an entire dataset. Implement it directly when the source is not naturally one file per observation or requires proprietary coordination.

### 13.1 Custom plugin descriptor

A custom series plugin is loaded from a descriptor:

```text
javareader:/absolute/path/to/readers.jar:example.reader.DummySeriesReader
```

The implementation class must:

- Implement `datasets.readers.api.CustomSeriesReader`.
- Have an accessible no-argument constructor.
- Be present in the descriptor JAR with all required dependencies.

### 13.2 Dummy custom series reader

The same plugin can be used for eager and lazy per-file loading.

```java
package example.reader;

import datasets.NumericStorageType;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.api.CustomSeriesReader;
import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Minimal per-file custom series plugin. */
public final class DummySeriesReader implements CustomSeriesReader {

    /** Required by the reflective plugin loader. */
    public DummySeriesReader() {
    }

    @Override
    public Object read(
            LazySeriesRef reference,
            CustomReaderContext context
    ) throws IOException {
        Path file = reference.getFile();

        String delimiter = context.getParameter(
                "delimiter",
                ","
        );

        List<String> requestedFeatures =
                context.getFeatureColumns();

        NumericStorageType storage =
                context.getNumericStorageType();

        // Your parsing logic here:
        // 1. Open reference.getFile().
        // 2. Parse requested features in dimension-major order.
        // 3. Convert numeric missing values to primitive NaN.
        // 4. Return raw values.
        // 5. Do not apply PFGAP standardization in the plugin.

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Unreadable observation file: " + file
            );
        }

        int dimensions = requestedFeatures.isEmpty()
                ? 1
                : requestedFeatures.size();

        int timeLength = 8; // Replace with the parsed length.

        if (storage == NumericStorageType.FLOAT32) {
            float[][] values =
                    new float[dimensions][timeLength];

            // Fill values using your parsing logic here.
            return values;
        }

        double[][] values =
                new double[dimensions][timeLength];

        // Fill values using your parsing logic here.
        return values;
    }
}
```

For the current built-in custom coordinators, the plugin is per-file and must return a two-dimensional standard result:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate result must be `[1][time]`.

`CustomReaderContext` exposes:

- Configured data path
- Train/test state
- Classification/regression state
- Numeric and missing-value hints
- Requested feature columns
- Custom plugin parameters
- Requested numeric storage

### 13.3 Eager custom per-file loading

Use `PER_FILE_CUSTOM` to invoke the plugin during dataset construction:

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.PER_FILE_CUSTOM)
        .setDataPath("data/custom/train")
        .setFilePattern("sample_{instance:05d}.bin")
        .setCustomReaderDescriptor(
                "javareader:/opt/pfgap/readers.jar:"
                        + "example.reader.DummySeriesReader"
        )
        .setCustomReaderParameter("delimiter", ",")
        .setFeatureColumns(List.of("x", "y"))
        .setNumeric(true)
        .setNumericStorageType(NumericStorageType.FLOAT32)
        .setCustomReaderThreadSafe(true);

ListObjectDataset dataset =
        DatasetReaderFactory.create(options).read();
```

The eager custom coordinator:

1. Discovers matching files.
2. Extracts the numeric filename placeholder.
3. Orders files deterministically.
4. Creates one `LazySeriesRef` as the plugin locator for each file.
5. Invokes the plugin immediately.
6. Validates the per-file representation.
7. Applies optional prepared standardization.
8. Adds the materialized observation to the dataset.
9. Closes the plugin and plugin class loader.

### 13.4 Lazy custom per-file loading

Use `LAZY_PER_FILE_CUSTOM` with the same plugin and options:

```java
ReaderOptions options = new ReaderOptions()
        .setReaderType(ReaderType.LAZY_PER_FILE_CUSTOM)
        .setDataPath("data/custom/train")
        .setFilePattern("sample_{instance:05d}.bin")
        .setCustomReaderDescriptor(
                "javareader:/opt/pfgap/readers.jar:"
                        + "example.reader.DummySeriesReader"
        )
        .setCustomReaderParameter("delimiter", ",")
        .setFeatureColumns(List.of("x", "y"))
        .setNumeric(true)
        .setNumericStorageType(NumericStorageType.FLOAT32)
        .setCustomReaderThreadSafe(true);

ListObjectDataset dataset =
        DatasetReaderFactory.create(options).read();
```

The lazy coordinator discovers and orders files, creates references, and registers a serializable reconstruction specification. It does not invoke the plugin during `read()`.

When an observation is requested, the runtime chain is:

```text
JavaSeriesReader
    -> PerFileValidatingLazySeriesReader
        -> StandardizingLazySeriesReader, when configured
```

Validation occurs before standardization. Closing or replacing the registered outer reader closes the decorators, plugin instance, and plugin class loader.

Set `customReaderThreadSafe` to `true` only if one plugin instance can safely serve concurrent calls. If it is `false`, the runtime adapter synchronizes plugin invocation.

### 13.5 Dummy non-per-file eager reader

The built-in custom factory types are per-file coordinators. For a database, socket, archive, service, or proprietary shared container, implement `DatasetReader` directly.

```java
package example.reader;

import datasets.ListObjectDataset;
import datasets.readers.DatasetReader;

import java.io.IOException;

public final class DummyContainerReader implements DatasetReader {
    private final String source;

    public DummyContainerReader(String source) {
        this.source = source;
    }

    @Override
    public ListObjectDataset read() throws IOException {
        ListObjectDataset dataset =
                new ListObjectDataset(1);

        // Open the shared source here.
        // Discover records or observations here.
        // Parse each observation into a supported representation.
        // Add labels and observations in stable instance order.

        double[] tabularObservation = {
                32.0,
                1.82,
                78.5
        };

        dataset.add(
                "class-a",
                tabularObservation,
                0
        );

        dataset.setLength(tabularObservation.length);
        return dataset;
    }
}
```

Construct this reader directly or add a project-specific `ReaderType` and `DatasetReaderFactory` branch:

```java
DatasetReader reader =
        new DummyContainerReader(
                "data/proprietary.container"
        );

ListObjectDataset dataset = reader.read();
```

A non-per-file reader may return a supported one-dimensional tabular observation, one-dimensional sequence, two-dimensional observation, or a supported proprietary representation.

### 13.6 Dummy non-per-file lazy materializer

A non-per-file lazy source needs a `LazySeriesReader` that interprets each logical reference.

```java
package example.reader;

import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;

public final class DummyContainerSeriesReader
        implements LazySeriesReader, AutoCloseable {

    // Open a shared container, mapping, or reusable decoder here.

    @Override
    public Object read(LazySeriesRef reference) {
        int observationIndex = reference.getIndex();

        // Locate and parse one logical observation here.
        // Non-per-file readers may return a supported 1D or 2D form.

        return new double[] {
                observationIndex,
                observationIndex + 1.0
        };
    }

    @Override
    public void close() {
        // Close the shared container, mapping, decoder, or cache here.
    }
}
```

### 13.7 Dummy non-per-file lazy coordinator

The corresponding `DatasetReader` discovers logical observations, registers or owns the materializer, and creates stable references.

```java
package example.reader;

import datasets.ListObjectDataset;
import datasets.readers.DatasetReader;
import datasets.readers.lazy.LazySeriesRef;

import java.nio.file.Path;

public final class DummyLazyContainerReader
        implements DatasetReader {

    private final String readerKey;
    private final Path containerPath;
    private final int observationCount;

    public DummyLazyContainerReader(
            String readerKey,
            Path containerPath,
            int observationCount
    ) {
        this.readerKey = readerKey;
        this.containerPath = containerPath;
        this.observationCount = observationCount;
    }

    @Override
    public ListObjectDataset read() {
        // Register DummyContainerSeriesReader through AppContext,
        // or add a reconstructible LazySeriesReaderSpec/factory branch.

        ListObjectDataset dataset =
                new ListObjectDataset(observationCount);

        for (int index = 0;
             index < observationCount;
             index++) {

            LazySeriesRef reference =
                    new LazySeriesRef(
                            readerKey,
                            index,
                            containerPath
                    );

            dataset.add(
                    null,
                    reference,
                    index
            );
        }

        // Logical observation lengths are unknown until materialization.
        dataset.setLength(0);
        return dataset;
    }
}
```

For saved-model reconstruction, extend `LazySeriesReaderSpec` and `LazySeriesReaderFactory` with the serializable settings needed to recreate the materializer. Do not serialize live file handles, mappings, decoders, class loaders, or caches.

## 14. Resource lifecycle and concurrency

Lazy readers may retain reusable projections, mappings, plugin instances, class loaders, or caches. Resource-owning readers implement `AutoCloseable` and are closed when replaced or when the lazy-reader registry is cleared.

Decorators own their delegates. Closing the outer reader closes the complete chain.

Concurrent reads are permitted when the underlying implementation supports them. Lifecycle locks prevent close from racing with active materialization, validation, or standardization.

Custom plugins should keep per-read mutable state local. Declare a plugin thread-safe only when its shared state is immutable or safely concurrent.

## 15. Reader selection guidance

Choose a reader using these rules:

- Use optimized numeric readers when all selected features are numeric.
- Use generic readers for categorical, mixed, or proprietary values.
- Use eager loading when the dataset fits comfortably in memory and observations are reused frequently.
- Use lazy loading when observations are large, access is sparse, or startup time and memory pressure matter.
- Use wide delimited data for fixed-width tabular observations or equal-length univariate sequences.
- Use long format when multiple source rows belong to one observation.
- Use per-file readers when each observation is stored in a separate file.
- Use NPY or HDF5 for existing array-oriented scientific data.
- Use Parquet when typed columnar storage, projection, and efficient numeric decoding are valuable.
- Use a custom series plugin when only per-observation parsing is proprietary.
- Implement `DatasetReader` directly when discovery, grouping, labeling, or source coordination is also proprietary.

## 16. Current deferred work

The following work is intentionally deferred and is not part of the current reader subsystem:

- Lazy long-format readers
- Segmented or windowed NPY mapping for arrays too large for the current shared mapping strategy
- Additional custom per-file label extraction or label-file coordination
- Per-file writer support and eager/lazy output coordination

These items should be documented as future extensions rather than implied current functionality.
