# Writers

This guide explains how PFGAP writes complete materialized datasets, including complete imputed training and test data. Dataset writers mirror supported reader families, preserve the logical dataset layout, optionally embed labels, and inverse-transform standardized numeric values while writing.

For prediction, score, proximity, model, and experiment-result artifacts, see [Outputs](../reference/Outputs.md). For imputed-only Matrix Market output, see [Imputed-Only Output](../reference/Imputed_Only_Output.md).

## Writer purpose

A dataset writer converts a materialized `ListObjectDataset` into one supported external dataset format.

Writers are used when PFGAP must emit a complete dataset, especially for:

- complete imputed training data;
- complete imputed validation or test data; and
- Java integrations that explicitly request dataset export.

A complete dataset output contains every represented feature value. This differs from imputed-only output, which contains only values at coordinates that were originally missing.

## Current writer support

PFGAP currently provides mirror writers for these reader families:

```text
NUMERIC_DELIMITED
DELIMITED
NUMERIC_LONG_DELIMITED
LONG_DELIMITED
```

The factory also accepts these equivalent names.

### Numeric delimited

```text
NUMERIC_DELIMITED
NUMERIC_DELIMITED_FILE
NUMERICDELIMITED
```

Uses `NumericDelimitedFileWriter`.

### Generic delimited

```text
DELIMITED
DELIMITED_FILE
GENERIC_DELIMITED
```

Uses `DelimitedFileWriter`.

### Numeric long-form delimited

```text
NUMERIC_LONG_DELIMITED
NUMERIC_LONG_FORM_DELIMITED
NUMERIC_LONG_FORM
NUMERIC_LONG_FORMAT
```

Uses `NumericLongFormatWriter`.

### Generic long-form delimited

```text
LONG_DELIMITED
LONG_FORM_DELIMITED
GENERIC_LONG_DELIMITED
LONG_FORM
LONG_FORMAT
```

Uses `LongFormatWriter`.

Writer-name matching is case-insensitive. Spaces and hyphens are normalized to underscores.

## Unsupported complete-dataset formats

A complete dataset writer is not currently registered for:

```text
NUMERIC_PER_FILE_DELIMITED_SERIES
PER_FILE_DELIMITED_SERIES
NPY
PARQUET
```

Other reader types without a registered mirror writer are also rejected.

PFGAP does not silently fall back to a generic representation. If a complete output format is unavailable, the writer factory reports that no mirror writer is registered for the source reader type.

This limitation applies to complete dataset export. Other artifacts, such as predictions, scores, proximities, and imputed-only Matrix Market output, use their own dedicated writers.

## Mirror writer selection

PFGAP selects a complete-dataset writer from the reader type that produced the source dataset.

Conceptually:

```text
source reader type -> mirror dataset writer
```

This preserves the source data family rather than converting every dataset to one universal text format.

Before writing, PFGAP verifies that the selected writer supports:

- the materialized runtime representation;
- the requested logical layout; and
- the supplied write options.

The complete dataset is validated again during writing because later observations can differ from the first observation.

## Materialized datasets

The writer interface accepts a materialized `ListObjectDataset`.

A writer can reject unresolved lazy references when its format does not support streaming from that source. Complete output therefore requires a dataset representation that the selected writer can encode.

Writers must not modify the supplied dataset.

## Logical data layouts

`DatasetWriteOptions` distinguishes the logical meaning of an observation from its Java runtime array type.

Supported layout declarations are:

```text
TABULAR
UNIVARIATE_SERIES
MULTIVARIATE_SERIES
LONG_FORM
AUTO
```

## Tabular layout

```text
TABULAR
```

Each dataset instance is one fixed-schema feature row.

A one-dimensional array position represents one feature. Header names, when enabled, correspond to those feature positions.

## Univariate-series layout

```text
UNIVARIATE_SERIES
```

Each dataset instance is one univariate series. Series may be variable length when the selected writer supports the representation.

Although tabular rows and univariate series can both use one-dimensional arrays, their logical meanings differ. Writers use the declared layout rather than inferring semantics only from the array class.

## Multivariate-series layout

```text
MULTIVARIATE_SERIES
```

Each instance is one dimension-major multivariate series.

Delimited multivariate output requires both:

- an entry separator; and
- a distinct array separator.

The array separator distinguishes nested dimensions within the serialized observation.

## Long-form layout

```text
LONG_FORM
```

Each logical observation is emitted through records with explicit instance and time or order metadata according to the selected long-form writer.

Use the numeric long-form writer for numeric observations and the generic long-form writer for supported object-backed observations.

## Automatic layout

```text
AUTO
```

The concrete writer may infer a supported layout or reject the request. Use an explicit layout when the same runtime representation could mean either tabular data or a time series.

## Output path

Every writer request requires an output path.

In Java:

```java
DatasetWriteOptions options =
        DatasetWriteOptions.builder(outputPath)
                .build();
```

The path cannot be null.

When complete imputed output is requested by the application workflow, the experiment output coordinator supplies the appropriate artifact path and mirror writer.

## Entry separator

The default entry separator in `DatasetWriteOptions` is a comma:

```text
,
```

For FastCSV-backed delimited writers, the separator must resolve to exactly one character and cannot be a line delimiter.

Escaped control forms are recognized:

```text
\t
\n
\r
```

A tab is valid. Newline and carriage-return separators are rejected because they are record delimiters.

## Array separator

The array separator is optional except for delimited multivariate-series output.

It must:

- resolve to one character for writers that require a character delimiter;
- differ from the entry separator; and
- not be a line delimiter.

Example:

```text
entry separator = ,
array separator = :
```

## Headers

Enable a header with:

```text
includeHeader = true
```

A header requires a nonempty feature-name list. Every feature name must be non-null and nonblank.

If no header is requested, feature names can be omitted.

The target header defaults to:

```text
target
```

A supplied blank target name is normalized to `target`.

## Labels and targets

A writer can embed labels or targets in the complete dataset output.

The relevant options are:

```text
embedLabels
labels
targetName
targetPlacement
```

Target placement can be:

```text
FIRST
LAST
```

The default is `LAST`.

### Embedded-label validation

If `embedLabels` is true, the label list must be nonempty.

If `embedLabels` is false, the label list must be empty. Supply labels to a separate label writer or enable embedding rather than silently discarding them.

The selected writer also validates that label count and observation count are compatible.

## Numeric and generic writers

### Numeric writers

Numeric writers encode supported primitive numeric or numeric object-backed observations.

They preserve primitive storage semantics where the destination format supports them. A float-backed dataset is not widened to a complete double-backed intermediate solely for export.

### Generic writers

Generic writers encode supported object-backed values according to the selected delimited or long-form format.

A generic writer still validates the materialized representation and layout before writing.

## Inverse standardization

Complete numeric dataset output can return values to their original feature scale.

`DatasetWriteOptions` accepts exactly one inverse-parameter source:

1. reusable `StandardizationStats`; or
2. a dataset-aligned list of `PerSeriesStandardizationState` values.

Supplying both is rejected.

If neither is supplied, values are written as currently materialized.

## Reusable statistics

Reusable statistics apply to standardization scopes such as:

```text
GLOBAL
PER_DIMENSION
```

The writer uses the stored centers and scales while emitting each numeric value.

## Per-series states

Per-series states apply to:

```text
PER_SERIES
PER_SERIES_PER_DIMENSION
```

The state list must align with dataset-instance order.

## Streaming inverse transformation

Writers inverse-transform values while emitting output. They must not:

- mutate source arrays in place; or
- construct a complete inverse-transformed dataset copy.

This keeps the current in-memory dataset unchanged and avoids unnecessary full-dataset duplication.

## Complete imputed output

Request a complete imputed training dataset with:

```python
return_imputed_training=True
```

Request a complete imputed test dataset with:

```python
return_imputed_testing=True
```

Direct Java forms:

```text
-impute_train=true
-impute_test=true
```

The application selects the mirror writer from the training or test reader type.

A complete imputed output includes:

- every originally observed feature value; and
- every final imputed feature value.

When standardization was applied, the writer can inverse-transform values during output using the fitted reusable statistics or per-series states supplied by the workflow.

## Complete output versus imputed-only output

### Complete dataset output

- Uses a mirror dataset writer.
- Includes observed and imputed values.
- Preserves an external dataset layout.
- Can include labels and headers.
- Can inverse-transform values while writing.

### Imputed-only output

- Uses Matrix Market coordinate format.
- Includes only originally missing coordinates.
- Retains exact-zero imputations.
- Uses a flattened rectangular matrix representation.
- Is documented in [Imputed-Only Output](../reference/Imputed_Only_Output.md).

## Java API example: numeric tabular CSV

```java
import datasets.ListObjectDataset;
import datasets.readers.ReaderType;
import datasets.writers.DatasetWriteOptions;
import datasets.writers.DatasetWriter;
import datasets.writers.DatasetWriterFactory;

import java.nio.file.Path;
import java.util.List;

DatasetWriteOptions options =
        DatasetWriteOptions.builder(
                        Path.of("output/imputed.csv")
                )
                .setDataLayout(
                        DatasetWriteOptions.DataLayout.TABULAR
                )
                .setEntrySeparator(",")
                .setIncludeHeader(true)
                .setFeatureNames(
                        List.of("feature_1", "feature_2")
                )
                .setEmbedLabels(true)
                .setLabels(dataset.getLabels())
                .setTargetName("target")
                .setTargetPlacement(
                        DatasetWriteOptions.TargetPlacement.FIRST
                )
                .build();

DatasetWriter writer =
        DatasetWriterFactory.createFor(
                ReaderType.NUMERIC_DELIMITED,
                dataset,
                options
        );

Path written = writer.write(dataset, options);
```

The writer returns the completed artifact path.

## Java API example: multivariate delimited output

```java
DatasetWriteOptions options =
        DatasetWriteOptions.builder(
                        Path.of("output/multivariate.csv")
                )
                .setDataLayout(
                        DatasetWriteOptions.DataLayout.MULTIVARIATE_SERIES
                )
                .setEntrySeparator(",")
                .setArraySeparator(":")
                .setEmbedLabels(false)
                .build();
```

The selected writer must support the materialized multivariate representation and nested delimited encoding.

## Java API example: inverse transformation

For reusable statistics:

```java
DatasetWriteOptions options =
        DatasetWriteOptions.builder(outputPath)
                .setDataLayout(
                        DatasetWriteOptions.DataLayout.TABULAR
                )
                .setReusableStatistics(standardizationStats)
                .build();
```

For per-series states:

```java
DatasetWriteOptions options =
        DatasetWriteOptions.builder(outputPath)
                .setDataLayout(
                        DatasetWriteOptions.DataLayout.UNIVARIATE_SERIES
                )
                .setPerSeriesStates(perSeriesStates)
                .build();
```

Do not set both inverse-parameter sources.

## Writer preflight

Java integrations can test format availability with:

```java
boolean available =
        DatasetWriterFactory.isImplemented(readerType);
```

This reports whether a writer is registered for the reader type. It does not validate a specific dataset.

For representation-aware validation, use:

```java
DatasetWriter writer =
        DatasetWriterFactory.createFor(
                readerType,
                dataset,
                options
        );
```

If unsupported, the factory reports the writer format, requested layout, and first available materialized instance type.

## Writer contract

Every dataset writer provides:

```text
write(dataset, options)
supports(dataset, options)
formatName()
```

### `write`

Writes the complete materialized dataset and returns the completed path.

It can throw:

- `IOException` for file creation or writing failure; or
- `IllegalArgumentException` for unsupported representation or incompatible options.

### `supports`

Performs preflight checking without beginning output. The writer still validates the complete dataset during `write`.

### `formatName`

Returns a stable short format name for diagnostics and manifests.

## Validation rules

`DatasetWriteOptions` enforces these common rules:

- output path cannot be null;
- layout cannot be null;
- entry separator cannot be null or empty;
- entry and array separators must differ;
- multivariate delimited output requires an array separator;
- a requested header requires feature names;
- feature names cannot be null or blank;
- embedded output requires a nonempty label list;
- labels cannot be supplied while embedding is disabled;
- target placement cannot be null;
- reusable statistics and per-series states cannot both be supplied; and
- character-based writers require a single non-line-delimiter separator.

Concrete writers additionally validate:

- supported runtime instance types;
- supported logical layouts;
- consistent observation shapes where required;
- label and instance counts;
- header width and feature names;
- finite numeric output where required; and
- representation-specific schema rules.

## File and dataset integrity

A writer must not mutate:

- the source dataset;
- primitive source arrays;
- labels;
- reusable standardization statistics; or
- per-series state objects.

Output creation should be treated as a terminal encoding operation over the materialized values and immutable write options.

## Common problems

### No mirror writer is registered

Use a currently supported complete-output reader family:

```text
NUMERIC_DELIMITED
DELIMITED
NUMERIC_LONG_DELIMITED
LONG_DELIMITED
```

Alternatively, request another artifact type that has its own writer, such as imputed-only Matrix Market output.

### The writer does not support the materialized instance type

Ensure that the reader, `numeric_data`, numeric storage, and logical layout produce a representation supported by the selected mirror writer.

### A header request fails

Supply a nonempty feature-name list with no null or blank names.

### Embedded labels fail

Enable label embedding and provide one compatible label or target per observation. If labels should be separate, do not pass them to the dataset writer.

### Multivariate delimited output rejects the options

Supply a nonempty array separator different from the entry separator.

### A separator is rejected

Use exactly one non-line-delimiter character for a character-based delimited writer. `\t` is accepted as an escaped tab.

### Inverse transformation configuration fails

Supply either reusable statistics or per-series states, not both. Ensure that the supplied state matches dataset shape and instance order.

### Complete imputed output is unavailable for the source format

Use a supported mirror-writer format or request imputed-only `.mtx` output where appropriate.

### Output values remain standardized

Ensure that the workflow supplies the fitted reusable statistics or per-series states to the writer options.

## Related documentation

- [Readers](Readers.md)
- [Data Formats](Data_Formats.md)
- [Dataset Representations](Dataset_Representations.md)
- [Imputation](../guides/Imputation.md)
- [Standardization](../guides/Standardization.md)
- [Outputs](../reference/Outputs.md)
- [Imputed-Only Output](../reference/Imputed_Only_Output.md)
- [Configuration Reference](../reference/Configuration_Reference.md)
