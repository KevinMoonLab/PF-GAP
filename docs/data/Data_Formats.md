# Data Formats

PFGAP supports multiple external dataset layouts for tabular observations, univariate sequences, multivariate sequences, equal-length data, unequal-length data, numeric values, and generic values. This page describes the structure and intended use of those formats. It does not define every reader option or imply that every format supports every dataset representation.

> **Status: Stable for v1, with format-specific verification items.** The format families described here are part of the current PFGAP data architecture. Exact reader capabilities, public option names, defaults, and unsupported combinations are documented in [Readers](Readers.md) and the interface references.

## 1. Formats, representations, and readers

Three related concepts must be kept separate:

- A **data format** is the external organization of a dataset.
- A **dataset representation** is the in-memory shape and element type used by PFGAP.
- A **reader** converts a supported external format into a supported dataset representation.

For example, a delimited row may be read as either:

```text
double[dimension]
```

for a tabular observation, or:

```text
double[time]
```

for a univariate sequence. Both use the Java type `double[]`; their axis semantics come from the reader configuration, source schema, and workflow.

A long-format Parquet file may instead be assembled into:

```text
float[dimension][time]
```

The file format alone does not determine every aspect of the resulting representation.

See [Dataset Representations](Dataset_Representations.md) for the in-memory data contract.

## 2. Choosing a format

The best format depends on the structure, scale, and access pattern of the dataset.

Use a **wide delimited format** when:

- Each observation fits naturally on one row.
- The data is a fixed-width tabular feature vector or an equal-length univariate sequence.
- Human readability is useful.
- The dataset is not so large that text parsing becomes a major bottleneck.

Use a **long format** when:

- Observations have multiple dimensions.
- Sequential observations have unequal lengths.
- Each value needs an explicit observation, dimension, time, or position coordinate.
- The same logical schema should be used across text or columnar storage.

Use a **per-file layout** when:

- Each observation already exists as a separate file.
- Series are large or naturally independent.
- Observation-level lazy access is desirable.
- Observation-specific metadata or lengths vary.

Use **NPY** when:

- Data originates in the NumPy ecosystem.
- Primitive numeric storage and efficient loading are priorities.
- The shape can be represented by the supported NPY reader contract.

Use **HDF5** when:

- Multiple arrays or dataset components should be stored in one container.
- Hierarchical organization is useful.
- Large numeric datasets require binary storage.
- The internal HDF5 schema matches a supported reader.

Use **Parquet** when:

- Typed columnar storage is beneficial.
- The data is produced by a data-engineering pipeline.
- Long-format or row-oriented schemas need efficient projection and decoding.
- The selected reader supports the required access pattern.

Use a **TS file** when:

- Data is distributed in UEA or UCR time-series archive style.
- The file contains time-series metadata and class or target declarations.
- The selected TS reader supports the dataset's task and structure.

Use a **custom reader** when:

- The source does not match a built-in schema.
- Data comes from a database, service, archive, object store, or domain-specific container.
- Existing data should be exposed without conversion to an intermediate format.

## 3. Delimited wide format

A delimited wide file normally stores one observation per row and one value per field.

For tabular data, each field is a feature or dimension:

```text
age,height,weight
32.0,1.82,78.5
45.0,1.75,81.2
```

The resulting observation is logically:

```text
double[dimension]
```

For a univariate sequence, each field is an ordered position:

```text
1.2,1.4,1.1,0.8
0.2,0.5,0.7,0.9
```

The resulting observation is logically:

```text
double[time]
```

These are both Java `double[]` observations. The schema determines whether an array index represents a feature or an ordered position.

### 3.1 Embedded targets

With the target in the first column:

```text
class_a,1.2,1.4,1.1,0.8
class_b,0.2,0.5,0.7,0.9
```

With the target in the last column:

```text
1.2,1.4,1.1,0.8,class_a
0.2,0.5,0.7,0.9,class_b
```

Labels or targets may be:

- Embedded in the first column
- Embedded in the last column
- Located in another supported configured column
- Supplied in a separate file
- Absent for an unlabeled prediction dataset

The task guide and reader reference define which combinations are valid.

### 3.2 Delimiters

Common delimiters include:

- Comma
- Tab
- Semicolon
- Space
- Another explicitly configured separator

The delimiter is part of reader configuration. A `.csv` extension does not by itself prove that the separator is a comma, and a `.txt` extension does not identify a schema.

### 3.3 Headers

A delimited file may contain a header row:

```text
label,t0,t1,t2,t3
class_a,1.2,1.4,1.1,0.8
class_b,0.2,0.5,0.7,0.9
```

Header handling must be configured consistently with the source file. When headers are enabled, a reader may use them for selection, validation, or metadata.

For tabular data, column ordering defines feature ordering unless the reader explicitly selects and reorders named columns.

### 3.4 Missing fields

A numeric delimited reader converts supported missing tokens or empty fields to primitive `NaN` values. A generic reader may represent supported missing fields with `null`.

Missing-token syntax is reader configuration. Users should not assume that every spelling, such as `NA`, `null`, `?`, or an empty string, is accepted unless configured or documented.

### 3.5 Shape considerations

Wide format most naturally represents:

- Fixed-width tabular observations
- Equal-length univariate sequences

Tabular rows normally require a consistent number and ordering of features. Sequential rows normally require equal length unless the selected reader documents support for trailing missing fields or another variable-length encoding.

A single wide row cannot reconstruct an arbitrary ragged multivariate observation unless additional shape information or a documented encoding is provided.

## 4. Generic delimited format

Generic delimited input uses the same row-and-field organization as numeric delimited input but does not require every observation value to be parsed as a primitive floating-point number.

Example generic tabular data:

```text
label,color,size,count
class_a,red,medium,3
class_b,blue,large,7
```

Example generic sequences:

```text
class_a,red,green,green,blue
class_b,small,medium,large,large
```

A generic row may produce:

```text
Object[dimension]
```

for tabular data, or:

```text
Object[time]
```

for a univariate generic sequence.

Generic parsing is appropriate for categorical, textual, mixed, or otherwise nonnumeric values. The selected distance and workflow must support the produced `Object[]` or `Object[][]` representation and its semantics.

Numeric data should normally use a numeric reader rather than a generic reader containing boxed numeric values.

## 5. Long format

Long format stores one logical measurement per row. Multiple rows are grouped to assemble each observation.

A feature-column schema can store one row per time or position with multiple feature values:

```text
observation_id,time,x,y,target
0,0,1.2,4.0,class_a
0,1,1.4,4.2,class_a
1,0,0.2,3.5,class_b
1,1,0.5,3.8,class_b
```

A value-column schema can store one row per dimension and time coordinate:

```text
observation_id,dimension,time,value,target
0,0,0,1.2,class_a
0,0,1,1.4,class_a
0,1,0,4.0,class_a
0,1,1,4.2,class_a
1,0,0,0.2,class_b
1,0,1,0.5,class_b
```

The exact supported schema is reader-specific. A long-format source may include:

- Observation or series ID
- Dimension or channel ID
- Time or position value
- One or more feature columns
- A single value column
- Label or regression target
- Additional metadata columns

### 5.1 Grouping

The reader groups rows by observation ID. A value-column schema may also require grouping by dimension ID. Row order alone should not substitute for explicit grouping unless the reader contract states that the source is order-dependent.

### 5.2 Ordering

Within each observation and dimension, values must be placed in a defined order. That order may come from:

- Source row order
- An explicit time column
- An explicit position column
- A reader-specific sorting policy

The reader documentation must state whether it sorts by time, relies on source order, or rejects ambiguous input.

### 5.3 Unequal lengths

Long format naturally supports unequal-length observations because each observation can contribute a different number of rows.

It can also express dimensions with different lengths inside one multivariate observation. Support for that internally ragged shape depends on downstream readers, distances, transformations, imputers, and writers, not only on the external format.

### 5.4 Repeated labels or targets

A long-format source may repeat an observation's label or target on every row, store it once, or provide it separately. Repeated values must be consistent within an observation.

A reader should reject conflicting labels or targets rather than silently choose one.

### 5.5 Numeric and generic long format

Numeric long-format readers produce primitive numeric observations. Generic long-format readers produce object-array observations.

Grouping and ordering rules can be shared even when value parsing, storage, and missingness differ.

## 6. Per-file datasets

A per-file dataset stores each observation in its own file. A directory structure, naming convention, manifest, or label file associates files with observation IDs and targets.

Conceptual layout:

```text
dataset/
|-- series_0001.csv
|-- series_0002.csv
|-- series_0003.csv
`-- labels.csv
```

A per-file observation may itself use a row-oriented table, a wide layout, or another reader-specific schema.

PFGAP per-file readers preserve the dimension axis. Their standard materialized representations are:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate per-file observation remains `[1][time]` rather than being collapsed to a one-dimensional array.

### 6.1 Advantages

- Natural support for unequal-length observations
- Observation-level lazy loading
- Independent access to large series
- Straightforward replacement or inspection of one observation
- Compatibility with one-recording-per-file datasets

### 6.2 Costs

- More filesystem operations
- Possible open-file and resource-management pressure
- Directory traversal and pattern-validation overhead
- More opportunities for missing, duplicated, or inconsistently named files

### 6.3 File discovery

A per-file reader must define how it discovers observations. Mechanisms may include:

- A manifest file
- A label file containing paths
- Deterministic directory enumeration
- A file-name pattern
- Explicit paths supplied by a custom reader specification

The current pattern-based readers use one numeric placeholder, such as:

```text
series_{instance:05d}.csv
```

The extracted numeric component defines deterministic observation order.

Discovery order must remain aligned with labels, predictions, scores, proximities, and output artifacts.

### 6.4 Lazy access

Per-file layouts are well suited to lazy materialization because one observation can be loaded independently. A lazy implementation must still define:

- Caching behavior
- Concurrency behavior
- Resource ownership and closing
- Deferred-read error handling
- Metadata availability
- Standardization behavior
- Saved-model reconstruction

Per-file writer support is separate from per-file reader support and may be intentionally deferred.

## 7. NumPy NPY format

NPY stores one NumPy array with dtype and shape metadata in a binary file.

PFGAP uses NPY for supported numeric array layouts. Compatibility depends on:

- Dtype
- Rank
- Shape
- Memory layout
- Axis conventions
- Whether labels or targets are stored separately

Primitive numeric NPY data maps naturally to PFGAP's standard observation families after applying the reader's dataset-axis interpretation:

```text
float[]
double[]
float[][]
double[][]
```

A one-dimensional materialized observation may be interpreted as:

```text
float[dimension]
double[dimension]
```

for tabular data, or:

```text
float[time]
double[time]
```

for a univariate sequence.

Object-dtype arrays are not equivalent to primitive numeric storage and should not be assumed to work with numeric readers.

Because one NPY file stores one array, a dataset may require multiple files, such as one for observations and one for labels.

The reader reference defines supported ranks, axis conventions, dtypes, and eager or lazy behavior.

## 8. HDF5 format

HDF5 is a hierarchical binary container that can store multiple named datasets and metadata entries.

A PFGAP HDF5 reader requires a supported schema describing where to find:

- Observation data
- Labels or targets
- Shape or length information for variable-length data
- Optional IDs or metadata

Conceptual structure:

```text
/data
/labels
/lengths
/ids
```

Actual dataset paths are configuration or reader-specific conventions.

HDF5 can represent dense equal-shape arrays directly. Unequal-length data may require:

- Variable-length datasets
- Grouped per-observation arrays
- Padding plus length metadata
- Another explicitly supported schema

The presence of an HDF5 file does not imply that its internal organization matches a built-in PFGAP reader. Unsupported schemas require conversion or a custom reader.

## 9. Parquet format

Parquet is a typed columnar format. PFGAP supports documented Parquet schemas rather than treating every Parquet table as interchangeable.

### 9.1 Long-format Parquet

Long-format Parquet stores the same logical organization as delimited long format using typed columns.

Feature-column example:

```text
observation_id | time | x | y | label
```

Value-column example:

```text
observation_id | dimension | time | value | label
```

The selected reader determines the accepted schema.

### 9.2 Per-file Parquet

A per-file Parquet dataset stores one logical observation per Parquet file. Each row normally represents one time or position, and selected feature columns become dimensions.

Conceptual file schema:

```text
time | x | y | z
```

The materialized observation is dimension-major:

```text
double[dimension][time]
```

or:

```text
float[dimension][time]
```

Generic per-file Parquet readers may produce `Object[dimension][time]`.

### 9.3 Schema requirements

Parquet readers must define:

- Required columns
- Accepted physical and logical types
- Observation grouping
- Dimension or feature selection
- Time ordering policy
- Label or target placement
- Null and missing-value handling
- Eager or lazy access behavior
- Numeric storage conversion

A reader should validate required columns and types before training begins.

## 10. TS format

The TS format is commonly used for time-series classification and regression datasets and can include metadata followed by observation records.

A TS file may describe:

- Problem name
- Whether timestamps are present
- Whether values are univariate or multivariate
- Whether series have equal lengths
- Whether class labels or targets are present
- Allowed class values
- Missing values

The data section contains one observation per logical record, with dimensions separated according to TS syntax.

Users should verify support for:

- Classification versus regression metadata
- Timestamped versus non-timestamped series
- Equal versus unequal lengths
- Missing values
- Multivariate dimensions
- String labels

Exact TS reader behavior belongs in the reader reference.

## 11. Separate label or target files

Some layouts store observations and targets in separate files.

Example observation file:

```text
1.2,1.4,1.1,0.8
0.2,0.5,0.7,0.9
```

Example label file:

```text
class_a
class_b
```

The number and order of labels must match the number and order of observations unless an explicit ID-based join is supported.

A separate target file should have an unambiguous schema. Extra columns, headers, or duplicate IDs must be handled according to the selected reader rather than silently ignored.

## 12. Training and evaluation sources

Training and evaluation data normally use compatible schemas, but they do not always contain the same target information.

Typical cases include:

- Labeled training data and labeled evaluation data
- Labeled training data and unlabeled prediction data
- Training and evaluation observations with separate target files
- A saved model loaded for evaluation without rereading the original training source

Reader configuration must interpret both datasets consistently. In particular, these should remain compatible:

- Tabular feature count and ordering
- Multivariate dimension count and meaning
- Generic value types
- Numeric precision
- Missing-value representation
- Standardization scope and fitted state

For tabular data, matching array lengths are insufficient if column meanings or ordering differ.

## 13. Missing values in external formats

External formats represent missingness differently. Possible forms include:

- Empty text fields
- A configured text token
- IEEE floating-point `NaN`
- Database or columnar null values
- TS-specific missing syntax
- A mask or index dataset

Numeric readers convert supported missing entries to primitive `NaN`. Generic readers may convert them to `null`.

PFGAP also records which positions were originally missing when required for imputation and imputed-only output. Replacing a missing value does not erase its original-missing status.

See [Missing Values](../reference/Missing_Values.md) for the complete missingness contract.

## 14. IDs, names, and observation order

Formats may provide explicit observation IDs, file names, row positions, or generated indices.

A reader must establish deterministic observation order because that order aligns:

- Observations
- Labels or targets
- Predictions
- Proximity matrix rows and columns
- Outlier and OOD scores
- Imputation outputs

An ID is not necessarily the same as the zero-based dataset index. Documentation and output schemas should distinguish them.

When a format does not provide IDs, a reader may derive stable IDs from row order, numeric file-pattern order, or another documented rule.

## 15. Time coordinates and feature coordinates

A source may include explicit feature names, dimension identifiers, time coordinates, or position coordinates.

### 15.1 Tabular features

For tabular data, column names and ordering define feature semantics. A reader must preserve or intentionally define a stable feature order.

A one-dimensional tabular observation is logically:

```text
observation[dimension]
```

Its index is not a time coordinate.

### 15.2 Sequential coordinates

A long-format or per-file source may include explicit time coordinates. A reader must define whether those coordinates are:

- Used only to sort values
- Preserved as metadata
- Passed to a timestamp-aware representation or distance
- Replaced by zero-based positions
- Written back by supported writers

Users should not assume that original time values survive a read-write cycle unless the selected reader and writer explicitly guarantee it.

See [Readers](Readers.md) and [Writers](Writers.md) for format-specific behavior.

## 16. Headers and metadata

Headers may contain field names, while binary or structured formats may contain schema metadata.

PFGAP may use metadata to:

- Locate target, ID, dimension, time, or value fields
- Validate types
- Select or order features
- Generate feature or dimension names
- Reconstruct shapes
- Select parsing behavior

Unknown metadata may be ignored, preserved, or rejected depending on the reader. Metadata preservation is not implied unless documented.

## 17. Compression

Some container formats support internal compression. Text files may also be externally compressed.

Built-in reader support for a format does not automatically imply support for every compression codec or archive layout. Required codec libraries and accepted extensions are documented per reader.

A compressed archive containing many observation files is not the same as a per-file directory unless the selected reader explicitly supports archive access.

## 18. Eager and lazy format support

Eager and lazy access are reader capabilities rather than intrinsic format properties.

A format may support:

- Eager reading only
- Lazy reading only
- Both through separate readers
- Both through one configurable reader

Lazy access is most useful when the source supports efficient observation-level lookup. A nominally lazy reader that repeatedly scans an entire file may save memory while increasing I/O cost.

Lazy implementations must also define resource ownership, concurrency, deferred errors, and saved-model reconstruction.

See [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md) for execution and resource-management behavior.

## 19. Numeric precision and source types

Source formats may distinguish integer, single-precision, and double-precision values. PFGAP's standard numeric dataset representations are floating point.

A numeric reader must define how it handles:

- Integer source values
- `float32` source values
- `float64` source values
- Higher-precision decimal types
- Mixed numeric columns
- Values outside the selected precision's range
- Automatic precision selection

`NumericStorageType` normally selects `AUTO`, `FLOAT32`, or `FLOAT64` materialization.

Conversions should be intentional and documented. Generic readers should not be used as an accidental substitute for numeric type conversion.

## 20. Reader and writer asymmetry

PFGAP may read a format without providing a matching writer. This is intentional.

Reasons include:

- The source format has many valid schemas but no single safe output schema.
- Writing requires metadata not retained by the reader.
- A reader is needed for training while another output format is better for results.
- Lazy input support does not determine output organization.
- A writer is deferred beyond the current release scope.

The [Readers](Readers.md) and [Writers](Writers.md) pages document their capability sets independently.

## 21. Output formats are documented separately

Input data formats and result artifacts overlap, but they are not the same topic.

For example:

- A delimited writer may write a complete imputed dataset.
- Matrix Market may store sparse proximities or imputed-only values.
- JSON may store experiment metadata rather than observations.
- A serialized model is not a dataset format.

See [Outputs](../reference/Outputs.md) for result artifacts and [Imputed-Only Output](../reference/Imputed_Only_Output.md) for sparse imputation packaging.

## 22. Custom formats

A custom reader can expose a dataset from any source that can be translated into a supported PFGAP representation or a documented proprietary representation understood by downstream components.

Examples include:

- Domain-specific binary recordings
- Databases
- Object stores
- Remote services
- Archive files
- Existing Java data structures

A custom format specification should define:

- Source identification
- Observation boundaries
- Produced Java type
- Numeric, generic, or proprietary value semantics
- Tabular, univariate sequential, or multivariate interpretation
- Dimension, feature, and time ordering
- Precision
- Missingness
- Labels or targets
- IDs
- Eager or lazy behavior
- Resource ownership
- Error handling

### 22.1 Per-file custom formats

The built-in custom coordinators discover one file per observation and invoke a `CustomSeriesReader`.

A standard per-file custom result must preserve the dimension axis:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate result is `[1][time]`.

### 22.2 Non-per-file custom formats

A custom `DatasetReader` can coordinate a shared source and may produce:

- One-dimensional tabular observations
- One-dimensional univariate sequences
- Two-dimensional observations
- Documented proprietary representations

A lazy non-per-file implementation additionally requires stable references, a materializer, resource ownership, and serializable reconstruction state when saved-model support is required.

See [Readers](Readers.md) for complete custom eager and lazy examples.

## 23. Validation checklist

Before configuring a reader, answer the following questions:

- Which external format and logical schema does the source use?
- Is each observation stored on one row, across many rows, in a shared array, or in a separate file?
- Is the observation tabular, univariate sequential, or multivariate?
- If one-dimensional, does its axis represent features or ordered positions?
- Are tabular feature counts and ordering consistent?
- Are sequential lengths equal or unequal across observations?
- Are multivariate observations internally rectangular or ragged?
- Are values numeric, generic, or proprietary?
- What source precision or dtype is used?
- Where are labels or targets stored?
- Does the evaluation source contain targets?
- Are observation IDs explicit?
- Is a time or position field present?
- How are missing values encoded?
- Is a header or schema definition present?
- Is eager or lazy access required?
- Does the selected built-in reader support this exact schema?
- Does the selected distance support the resulting representation and semantics?
- Is a matching writer actually required?

## 24. Reader capability matrix

The reader reference provides the authoritative capability matrix. It should record, for each reader:

- Format family
- Numeric or generic values
- Tabular, univariate sequential, or multivariate observations
- One-dimensional or two-dimensional output
- Equal-length, unequal-length, and ragged support
- Single and double precision
- Eager and lazy access
- Embedded or separate targets
- Missing values
- Explicit IDs and time coordinates
- Standardization compatibility
- Imputation compatibility
- Available matching writers

This format overview should remain conceptual. Exact support claims belong in the verified reader reference.

## 25. Related documentation

- [Dataset Representations](Dataset_Representations.md)
- [Readers](Readers.md)
- [Writers](Writers.md)
- [Missing Values](../reference/Missing_Values.md)
- [Configuration](../getting-started/Configuration.md)
- [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md)
- [Outputs](../reference/Outputs.md)
- [Custom Readers](../extensions/Custom_Readers.md)
