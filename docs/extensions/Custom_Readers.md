# Custom Readers

PFGAP supports custom Java readers at two extension levels:

1. **Custom per-file series plugins**, where PFGAP handles discovery, ordering, eager or deferred materialization, validation, standardization, and lifecycle management.
2. **Custom whole-dataset readers**, where the extension controls source discovery and dataset assembly directly.

A custom reader can be packaged in a separate JAR. Use a per-file plugin when only the parsing of one observation is proprietary. Implement a whole-dataset reader when grouping, labeling, source coordination, or observation discovery is also proprietary.

For built-in formats and representation rules, see [Readers](../data/Readers.md). For eager and deferred access behavior, see [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md).

### Choose the extension level

#### Use `CustomSeriesReader` when

- one source file represents one observation;
- PFGAP can discover files from a directory and pattern;
- labels are supplied through the ordinary reader configuration;
- the plugin only needs to parse one file into one observation; and
- the same parser should support both eager and deferred materialization.

#### Implement `DatasetReader` directly when

- one observation spans multiple files;
- multiple observations share one proprietary container;
- observations come from a database, service, socket, archive, or stream;
- discovery and ordering are source-specific;
- labels must be extracted through proprietary logic;
- a one-dimensional custom representation is required; or
- per-file coordination does not match the data source.

## Custom per-file series plugins

A custom per-file plugin implements:

```java
package datasets.readers.api;

public interface CustomSeriesReader {
    Object read(
            LazySeriesRef reference,
            CustomReaderContext context
    ) throws IOException;
}
```

The plugin parses one observation identified by a `LazySeriesRef` and returns its raw in-memory representation.

The same implementation can be used by:

```text
PER_FILE_CUSTOM
LAZY_PER_FILE_CUSTOM
```

`PER_FILE_CUSTOM` invokes the plugin while the dataset is being built. `LAZY_PER_FILE_CUSTOM` stores references and invokes the plugin when an observation is requested.

### Separate-JAR descriptor

Package the implementation and its dependencies in a JAR, then identify the class with:

```text
javareader:/absolute/path/to/readers.jar:example.reader.SeriesReader
```

The implementation class must:

- implement `datasets.readers.api.CustomSeriesReader`;
- have an accessible no-argument constructor;
- be present in the descriptor JAR; and
- have every required dependency available to its class loader.

Use an absolute JAR path when the working directory may vary. A relative path is resolved from the PFGAP process working directory.

### Per-file observation contract

A custom per-file plugin must return a two-dimensional, dimension-major observation.

Supported standard forms are:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate series must retain the dimension axis:

```text
float[1][time]
double[1][time]
Object[1][time]
```

Do not return these one-dimensional forms from a per-file plugin:

```text
float[]
double[]
Object[]
```

General and whole-dataset custom readers can return supported one-dimensional observations. The two-dimensional restriction is specific to the built-in per-file coordinators.

### Raw-value contract

Return raw source values from the plugin.

The plugin must not apply PFGAP standardization. When standardization is configured, PFGAP applies it after the custom result has been validated.

For numeric output:

- return primitive `float[][]` or `double[][]` where possible;
- represent numeric missing values with `Float.NaN` or `Double.NaN`; and
- observe the requested `NumericStorageType` from the context.

For generic output:

- return a valid `Object[][]` or a supported proprietary two-dimensional representation; and
- represent generic missing values according to the plugin and downstream distance contract, normally with `null`.

### `CustomReaderContext`

The custom context supplies the configured information needed to parse one observation, including:

- data path;
- training or test state;
- classification or regression state;
- numeric-data setting;
- missing-value hint;
- requested feature columns;
- custom plugin parameters; and
- requested numeric storage.

Use context values rather than hard-coding one experiment configuration into the plugin.

Example parameter access:

```java
String delimiter = context.getParameter(
        "delimiter",
        ","
);

List<String> features =
        context.getFeatureColumns();

NumericStorageType storage =
        context.getNumericStorageType();
```

### Minimal per-file plugin

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

public final class SeriesReader
        implements CustomSeriesReader {

    public SeriesReader() {
    }

    @Override
    public Object read(
            LazySeriesRef reference,
            CustomReaderContext context
    ) throws IOException {
        Path file = reference.getFile();

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Unreadable observation file: " + file
            );
        }

        String delimiter = context.getParameter(
                "delimiter",
                ","
        );

        List<String> requestedFeatures =
                context.getFeatureColumns();

        NumericStorageType storage =
                context.getNumericStorageType();

        int dimensions = requestedFeatures.isEmpty()
                ? 1
                : requestedFeatures.size();

        int timeLength = discoverLength(
                file,
                delimiter
        );

        if (storage == NumericStorageType.FLOAT32) {
            float[][] values =
                    new float[dimensions][timeLength];

            readFloatValues(
                    file,
                    delimiter,
                    requestedFeatures,
                    values
            );

            return values;
        }

        double[][] values =
                new double[dimensions][timeLength];

        readDoubleValues(
                file,
                delimiter,
                requestedFeatures,
                values
        );

        return values;
    }

    private int discoverLength(
            Path file,
            String delimiter
    ) throws IOException {
        // Determine the realized series length.
        throw new UnsupportedOperationException(
                "Implement source-specific length discovery."
        );
    }

    private void readFloatValues(
            Path file,
            String delimiter,
            List<String> requestedFeatures,
            float[][] destination
    ) throws IOException {
        // Parse raw values in dimension-major order.
        // Convert missing numeric values to Float.NaN.
        throw new UnsupportedOperationException(
                "Implement source-specific parsing."
        );
    }

    private void readDoubleValues(
            Path file,
            String delimiter,
            List<String> requestedFeatures,
            double[][] destination
    ) throws IOException {
        // Parse raw values in dimension-major order.
        // Convert missing numeric values to Double.NaN.
        throw new UnsupportedOperationException(
                "Implement source-specific parsing."
        );
    }
}
```

The example intentionally leaves source-specific parsing unimplemented. The returned arrays show the required per-file shape and numeric-storage behavior.

### Build the plugin JAR

A minimal source layout is:

```text
custom-reader/
└── src/
    └── example/
        └── reader/
            └── SeriesReader.java
```

Compile against the same PFGAP API revision used at runtime:

```bash
mkdir -p out

javac \
  -cp /opt/pfgap/PFGAP.jar \
  -d out \
  src/example/reader/SeriesReader.java
```

Create the JAR:

```bash
jar --create \
  --file readers.jar \
  -C out .
```

The resulting descriptor is:

```text
javareader:/path/to/custom-reader/readers.jar:example.reader.SeriesReader
```

If the plugin uses third-party classes, package them into the JAR or otherwise make them available to the plugin class loader according to the deployment arrangement.

## Eager custom per-file reading

Use `PER_FILE_CUSTOM` to invoke the plugin while building the dataset.

### Python configuration

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/custom/train",
    train_labels="../data/custom/train_labels.csv",
    reader_type="PER_FILE_CUSTOM",
    file_pattern="sample_{instance:05d}.bin",
    custom_reader_descriptor=(
        "javareader:/opt/pfgap/readers.jar:"
        "example.reader.SeriesReader"
    ),
    custom_reader_parameters={
        "delimiter": ",",
        "strict": "true",
    },
    custom_reader_thread_safe=True,
    feature_columns=["x", "y"],
    forest_mode="classification",
    data_dimension=2,
    numeric_data=True,
    numeric_storage="float32",
    output_directory="../output/custom_eager",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java configuration

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/custom/train \
  -train_labels=data/custom/train_labels.csv \
  -reader_type=PER_FILE_CUSTOM \
  '-file_pattern=sample_{instance:05d}.bin' \
  '-custom_reader_descriptor=javareader:/opt/pfgap/readers.jar:example.reader.SeriesReader' \
  '-custom_reader_parameters=delimiter=,;strict=true' \
  -custom_reader_thread_safe=true \
  -feature_columns=[x,y] \
  -forest_mode=classification \
  -is2D=true \
  -isNumeric=true \
  -numeric_storage=float32 \
  -out=output/custom_eager/
```

Quote the complete descriptor, pattern, and parameter arguments when required by the shell.

### Eager coordinator behavior

The eager custom coordinator:

1. discovers files matching the configured pattern;
2. extracts and sorts the numeric filename placeholder;
3. creates a `LazySeriesRef` as the stable plugin locator for each file;
4. invokes the plugin immediately;
5. validates the returned per-file representation;
6. applies prepared standardization when configured;
7. adds the materialized observation to the dataset; and
8. closes the plugin and its class loader when reading is complete.

The returned dataset stores materialized observations rather than lazy references.

## Deferred custom per-file reading

Use `LAZY_PER_FILE_CUSTOM` with the same plugin to defer parsing until an observation is needed.

### Python configuration

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/custom/train",
    train_labels="../data/custom/train_labels.csv",
    reader_type="LAZY_PER_FILE_CUSTOM",
    file_pattern="sample_{instance:05d}.bin",
    custom_reader_descriptor=(
        "javareader:/opt/pfgap/readers.jar:"
        "example.reader.SeriesReader"
    ),
    custom_reader_parameters={
        "delimiter": ",",
        "strict": "true",
    },
    custom_reader_thread_safe=True,
    feature_columns=["x", "y"],
    forest_mode="classification",
    data_dimension=2,
    numeric_data=True,
    numeric_storage="float32",
    output_directory="../output/custom_lazy",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java configuration

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/custom/train \
  -train_labels=data/custom/train_labels.csv \
  -reader_type=LAZY_PER_FILE_CUSTOM \
  '-file_pattern=sample_{instance:05d}.bin' \
  '-custom_reader_descriptor=javareader:/opt/pfgap/readers.jar:example.reader.SeriesReader' \
  '-custom_reader_parameters=delimiter=,;strict=true' \
  -custom_reader_thread_safe=true \
  -feature_columns=[x,y] \
  -forest_mode=classification \
  -is2D=true \
  -isNumeric=true \
  -numeric_storage=float32 \
  -out=output/custom_lazy/
```

### Deferred coordinator behavior

The deferred custom coordinator:

1. discovers and orders the source files;
2. creates one `LazySeriesRef` for each observation;
3. creates a serializable `LazySeriesReaderSpec` containing the descriptor, context, storage, and plugin settings;
4. registers the specification with the application reader registry; and
5. returns a dataset containing references instead of parsed observations.

The plugin is not called by the coordinator's initial `read()` operation.

When an observation is requested, the runtime chain is:

```text
JavaSeriesReader
  -> PerFileValidatingLazySeriesReader
    -> StandardizingLazySeriesReader, when configured
```

Validation occurs before standardization.

Closing or replacing the registered outer reader closes the decorators, plugin instance, and plugin class loader.

## Thread safety

Configure:

```python
custom_reader_thread_safe=True
```

or:

```text
-custom_reader_thread_safe=true
```

only when one plugin instance can safely serve concurrent `read(...)` calls.

A thread-safe plugin should keep per-observation mutable state local to the `read(...)` call. Shared state must be immutable or safely concurrent.

When thread safety is false, the runtime adapter serializes plugin invocation. This protects a non-thread-safe plugin but limits concurrent materialization through that plugin instance.

## Custom parameters

Python accepts a dictionary:

```python
custom_reader_parameters={
    "delimiter": ",",
    "strict": "true",
}
```

The direct Java form is a semicolon-separated assignment list:

```text
-custom_reader_parameters=delimiter=,;strict=true
```

Parameter rules are:

- names cannot be blank;
- names cannot contain `;` or `=`;
- values cannot contain `;`; and
- duplicate parameter names are rejected.

The plugin retrieves values through `CustomReaderContext`.

## File discovery and ordering

Per-file custom readers use the same discovery contract as built-in per-file readers.

A pattern contains one numeric placeholder, such as:

```text
sample_{instance:05d}.bin
```

PFGAP extracts the numeric value and sorts observations deterministically by that number.

Literal pattern fragments can contain supported wildcard characters. A directory source requires a pattern. A single regular-file source does not.

Separate labels must follow the discovered observation order.

## Missing values

For numeric results, use primitive `NaN` values:

```java
Float.NaN
Double.NaN
```

For generic results, use the representation expected by the configured downstream workflow, normally `null` for missing object values.

The custom plugin is responsible for parsing its source missing-value representation. It can use context hints and custom parameters to determine which source tokens or sentinels are missing.

Do not perform iterative imputation inside the plugin. Return the parsed missing values and let PFGAP's imputation workflow handle them.

## Standardization

Custom plugins return raw values.

For eager custom reading, PFGAP applies prepared standardization after validating the returned observation and before adding it to the dataset.

For deferred custom reading, `StandardizingLazySeriesReader` applies the transformation after materialization.

Applying standardization inside the plugin causes double transformation when PFGAP standardization is enabled.

See [Standardization](../guides/Standardization.md).

## Saved-model reconstruction

A deferred custom per-file dataset stores references and a serializable reader specification. The specification retains the information required to rebuild the plugin reader, including:

- descriptor JAR and class name;
- data location and reader key;
- custom parameters;
- feature selection;
- numeric-storage request;
- thread-safety declaration; and
- standardization configuration needed by the decorator chain.

The saved model does not serialize a live plugin instance, class loader, open file, decoder, or cache.

When the model is loaded, PFGAP reconstructs the reader from the saved specification. The custom JAR, source data, and required dependencies must remain available at the referenced locations.

See [Model Persistence](../reference/Model_Persistence.md).

## Whole-dataset eager readers

The built-in custom reader types are per-file coordinators. For a proprietary shared source, implement `DatasetReader` directly.

### Minimal whole-dataset reader

```java
package example.reader;

import datasets.ListObjectDataset;
import datasets.readers.DatasetReader;

import java.io.IOException;

public final class ContainerReader
        implements DatasetReader {

    private final String source;

    public ContainerReader(String source) {
        this.source = source;
    }

    @Override
    public ListObjectDataset read()
            throws IOException {
        ListObjectDataset dataset =
                new ListObjectDataset(1);

        // Open the shared source.
        // Discover observations in stable order.
        // Parse labels and observations.

        double[] observation = {
                32.0,
                1.82,
                78.5
        };

        dataset.add(
                1,
                observation,
                0
        );

        dataset.setLength(observation.length);
        return dataset;
    }
}
```

A whole-dataset reader can return supported:

```text
float[]
double[]
Object[]
float[][]
double[][]
Object[][]
```

It can also return a proprietary representation when built-in standardization is disabled and the configured distance or downstream consumer understands the representation.

### Factory integration

A whole-dataset reader can be constructed directly in a Java integration:

```java
DatasetReader reader =
        new ContainerReader(
                "data/proprietary.container"
        );

ListObjectDataset dataset = reader.read();
```

To select it through ordinary application configuration, add a project-specific `ReaderType` and a corresponding `DatasetReaderFactory` branch.

That project integration must define how the reader receives paths, labels, options, and reconstruction state.

## Whole-dataset deferred readers

A proprietary shared container can support deferred materialization through a custom `LazySeriesReader` plus a coordinating `DatasetReader`.

### Minimal materializer

```java
package example.reader;

import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;

public final class ContainerSeriesReader
        implements LazySeriesReader, AutoCloseable {

    public ContainerSeriesReader() {
        // Open a shared container, mapping, or decoder.
    }

    @Override
    public Object read(LazySeriesRef reference) {
        int observationIndex =
                reference.getIndex();

        // Locate and parse one logical observation.
        return new double[] {
                observationIndex,
                observationIndex + 1.0
        };
    }

    @Override
    public void close() {
        // Close shared resources.
    }
}
```

Unlike a per-file plugin, a direct whole-dataset materializer can return a supported one-dimensional observation.

### Minimal deferred coordinator

```java
package example.reader;

import datasets.ListObjectDataset;
import datasets.readers.DatasetReader;
import datasets.readers.lazy.LazySeriesRef;

import java.nio.file.Path;

public final class LazyContainerReader
        implements DatasetReader {

    private final String readerKey;
    private final Path containerPath;
    private final int observationCount;

    public LazyContainerReader(
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

        dataset.setLength(0);
        return dataset;
    }
}
```

The coordinator must also register the matching runtime materializer through `AppContext`.

## Reconstructible whole-dataset deferred readers

Saved-model reconstruction requires a serializable `LazySeriesReaderSpec` containing the settings needed to recreate the materializer.

A project-specific reconstructible deferred reader therefore needs:

1. a serializable specification;
2. a `LazySeriesReaderFactory` branch that creates the runtime reader from that specification;
3. stable reader keys and references;
4. source paths or locators that remain valid after model loading; and
5. lifecycle handling for every recreated resource.

Do not place live file handles, mappings, sockets, class loaders, decoder instances, or caches in the serialized specification.

A runtime-only registration can support the current process but cannot reconstruct itself in a later process.

## Resource lifecycle

A resource-owning materializer should implement `AutoCloseable`.

Close resources when:

- eager reading finishes;
- a registered deferred reader is replaced;
- the reader registry is cleared;
- the application workflow finishes; or
- model reconstruction fails after creating replacement readers.

For a decorated deferred reader, the outer reader owns its delegate. Close only the outer chain once.

Do not close a reader while its references are still in use.

## Packaging recommendations

A separate reader JAR should contain:

- the custom implementation class;
- helper classes used by the implementation;
- compatible dependency classes or an appropriate dependency arrangement; and
- no duplicate PFGAP API classes.

Compile against the PFGAP revision used at runtime. Keep the JAR and any source-specific dependencies with the saved experiment artifacts.

For reproducibility, retain:

- the reader JAR;
- source revision or checksum;
- PFGAP revision;
- Java version;
- descriptor string;
- custom parameter values;
- reader type;
- feature selection;
- numeric storage; and
- source-data version or checksum.

## Security

A custom reader JAR contains executable Java code. Load reader JARs only from trusted sources.

The plugin can read files available to the PFGAP process and execute with that process's permissions. Review custom code and dependencies before use.

## Common problems

### The plugin class cannot be loaded

Check the JAR path, fully qualified class name, no-argument constructor, and PFGAP API compatibility.

### A dependency class cannot be found

Package the dependency with the reader JAR or make it available through the supported deployment class path. Do not package duplicate PFGAP API classes into the plugin.

### The plugin returns `double[]`

A custom per-file plugin must return a two-dimensional result. Wrap a univariate series as `double[1][time]`, or implement a whole-dataset reader when a one-dimensional representation is required.

### Numeric storage is ignored

Read `NumericStorageType` from `CustomReaderContext` and return `float[][]` for `FLOAT32` or `double[][]` for `FLOAT64`. Define a consistent policy for `AUTO`.

### Missing values fail later in the workflow

Convert source missing values to primitive `NaN` for numeric output or the supported generic missing representation. Configure compatible distances or imputation.

### Values are standardized twice

Remove standardization from the plugin. Return raw values and let PFGAP apply its configured standardization layer.

### Deferred reads fail under multiple workers

Set `custom_reader_thread_safe=false` unless one plugin instance is safe for concurrent calls. If true, keep mutable per-read state local or safely concurrent.

### Eager and deferred results differ

Use the same plugin, context parameters, feature ordering, numeric storage, missing-value conversion, and source files. Both coordinators should receive the same raw observation from the plugin.

### Labels do not align with observations

Ensure the separate label order matches deterministic file-discovery order.

### A saved model cannot recreate the plugin reader

Keep the JAR and source data available at the saved locations. Ensure the model contains a serializable reader specification rather than only a runtime registration.

### The plugin JAR remains locked or source files cannot be changed

Close the application workflow or clear the reader registry so the plugin instance and class loader are released.

### A whole-dataset deferred reader works only in the current process

Add a serializable reader specification and factory reconstruction branch. Runtime-only readers are not portable through model persistence.

### Related documentation

- [Readers](../data/Readers.md)
- [Dataset Representations](../data/Dataset_Representations.md)
- [Data Formats](../data/Data_Formats.md)
- [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md)
- [Standardization](../guides/Standardization.md)
- [Missing Values](../reference/Missing_Values.md)
- [Model Persistence](../reference/Model_Persistence.md)
- [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md)
