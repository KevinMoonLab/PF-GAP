# Imputed-Only Output

This reference defines PFGAP's imputed-only output: a sparse coordinate artifact containing final numeric values only at positions that were missing in the original input. It describes the output request, file selection, coordinate mapping, data types, exact-zero handling, standardization reversal, unequal-length behavior, and validation rules.

For the full imputation workflow, see [Imputation](../guides/Imputation.md). For general artifact naming and output conventions, see [Outputs](Outputs.md).

### Purpose

A complete imputed dataset contains both originally observed values and final imputed values. Imputed-only output instead contains one entry for every originally missing feature coordinate and no entries for originally observed coordinates.

This format is useful when the consumer already has the source data and needs only the replacement values produced by PFGAP.

Imputed-only output is not a complete-dataset representation. When the complete imputed dataset is requested, PFGAP continues to use the appropriate reader-matched writer from `datasets.writers`.

### Format selection

PFGAP selects the imputed-only format from the logical data rank:

- tabular or univariate data uses Matrix Market coordinate format (`.mtx`);
- multivariate data uses sparse tensor coordinate format (`.tns`).

The requested filename is treated as an artifact path. PFGAP replaces its final extension with `.mtx` or `.tns` according to the output data rank. This prevents a multivariate artifact from retaining a misleading `.mtx` extension.

### Request training output

Python:

```python
return_imputed_training_csr = True
training_imputed_csr_file = (
    "../output/training_imputed_values"
)
```

Direct Java:

```text
-output_train_imputed_csr=true
-train_imputed_csr_file=output/training_imputed_values
```

The default base filename is:

```text
training_imputed_values
```

PFGAP appends `.mtx` for tabular or univariate data and `.tns` for multivariate data.

Requesting this output enables training-data imputation and missing-value handling.

### Request test output

Python:

```python
return_imputed_testing_csr = True
testing_imputed_csr_file = (
    "../output/testing_imputed_values"
)
```

Direct Java:

```text
-output_test_imputed_csr=true
-test_imputed_csr_file=output/testing_imputed_values
```

The default base filename is:

```text
testing_imputed_values
```

PFGAP appends `.mtx` for tabular or univariate data and `.tns` for multivariate data.

Requesting this output enables test-data imputation and missing-value handling.

### Entry semantics

An entry means:

> This source feature coordinate was missing in the original input, and this is its final imputed numeric value.

Entry presence is based on original missingness, not on whether the final value is nonzero.

Therefore:

- originally observed cells are omitted;
- originally missing cells are included;
- an imputed value of `0.0` is included; and
- an absent coordinate does not mean an imputed zero.

This differs from conventional numerical sparsity, where zero values are normally omitted.

Both file formats use one-based coordinates. PFGAP's internal indices are zero-based and are incremented during writing.

## Tabular and univariate Matrix Market output

### File format

Tabular and univariate imputed-only output uses Matrix Market coordinate format:

```text
%%MatrixMarket matrix coordinate real general
```

The shape line is:

```text
row_count column_count entry_count
```

Each entry is:

```text
row column value
```

The file is UTF-8 text. The Matrix Market writer creates missing parent directories and replaces an existing target file.

### Row mapping

Each matrix row corresponds to one dataset instance.

The mapping is:

```text
matrix row = instance index
```

Internally, instances are zero-based. In the `.mtx` file, instance `0` is written as Matrix Market row `1`.

The matrix row count is the dataset instance count, including instances with no missing values. An instance with no originally missing coordinates contributes no entries but still occupies a matrix row.

### One-dimensional column mapping

For tabular or univariate observations, each matrix column corresponds directly to a feature or time position:

```text
column = position
```

The matrix column count is the maximum realized one-dimensional observation length across the dataset.

Internally, position `0` maps to column `0`. In the `.mtx` file, it is written as Matrix Market column `1`.

#### Example

Suppose three observations have maximum length `5`, and the originally missing coordinates are:

```text
instance 0, position 1 -> 4.5
instance 2, position 4 -> 0.0
```

The logical matrix has shape:

```text
3 rows x 5 columns
```

The coordinate entries are:

```text
1 2 4.5
3 5 0.0
```

The second entry is retained even though its value is exactly zero.

### CSR representation

Before writing tabular or univariate output, PFGAP builds an immutable compressed sparse row result.

The main CSR arrays are:

- `rowOffsets`, with length `rowCount + 1`;
- `columnIndices`, with one direct feature or time position for each originally missing coordinate; and
- primitive `float[]` or `double[]` values.

For row `r`, its entries occupy:

```text
rowOffsets[r] <= offset < rowOffsets[r + 1]
```

Column indices are strictly increasing within each row.

The CSR entry count is exactly the number of originally missing feature coordinates recorded for the dataset.

`ImputedValuesCSR` and `ImputedValuesCSRBuilder` support only tabular and univariate data. They do not flatten multivariate `(dimension, time)` coordinates.

## Multivariate sparse tensor output

### File format

Multivariate imputed-only output uses sparse tensor coordinate format with the `.tns` extension.

The file begins with comments describing the maximum tensor envelope, declared entry count, and coordinate order:

```text
% PFGAP sparse tensor coordinate file
% shape: instance_count maximum_dimension_count maximum_time_length
% entries: entry_count
% coordinates: instance dimension time value
```

Each non-comment entry is:

```text
instance dimension time value
```

All three coordinates are one-based in the file.

The file is UTF-8 text. The sparse tensor writer creates missing parent directories and replaces an existing target file.

### Direct coordinate mapping

Each `.tns` entry maps directly to the source coordinate:

```text
instance = source instance index
dimension = source dimension index
time = source time position
```

PFGAP does not flatten `(dimension, time)` into a matrix column and does not construct an intermediate CSR representation for multivariate output.

`ImputedValuesTensorWriter` traverses the original 2D `MissingIndices` groups and streams the logical coordinates directly to `SparseTensorWriter`.

#### Example

Suppose the originally missing coordinates are:

```text
instance 0, dimension 1, time 90 -> 4.5
instance 2, dimension 0, time 119 -> 0.0
```

A corresponding `.tns` artifact could be:

```text
% PFGAP sparse tensor coordinate file
% shape: 3 2 120
% entries: 2
% coordinates: instance dimension time value
1 2 91 4.5
3 1 120 0.0
```

The second entry is retained even though its value is exactly zero.

### Rectangular envelope

The `% shape:` comment describes the maximum rectangular envelope:

```text
instanceCount x maximumDimensionCount x maximumTimeLength
```

This envelope does not assert that every instance has every dimension or that every dimension reaches `maximumTimeLength`.

It supplies useful global bounds and preserves trailing tensor extents that might not appear among the sparse imputed coordinates.

### Unequal-length and ragged data

The `.tns` representation supports:

- different numbers of dimensions across instances;
- different time lengths across instances;
- different time lengths across dimensions within an instance; and
- combinations of unequal dimension counts and unequal lengths.

Only valid originally missing coordinates are emitted. Each time coordinate is interpreted relative to the corresponding original instance and dimension.

The original dataset supplies the complete ragged shape. The `.tns` patch alone is not a standalone reconstruction of the source dataset.

### Imputed-only scope

The `.tns` format is used only for multivariate imputed-only output.

It is not used when the user requests the entire imputed dataset. Complete-dataset output remains the responsibility of the configured reader-matched writer in `datasets.writers`.

## Numeric storage type

The tabular or univariate CSR result preserves the materialized dataset's primitive numeric storage:

- `FLOAT32` for homogeneous `float[]` observations; or
- `FLOAT64` for homogeneous `double[]` observations.

Numeric `Object[]` values are read through `Number.doubleValue()` and produce `FLOAT64` output.

Multivariate tensor output supports homogeneous:

- `float[][]` observations;
- `double[][]` observations; or
- numeric `Object[][]` observations.

Tensor values are written as finite real text values. No intermediate tensor value array is required.

`AUTO` is not a materialized CSR data type. All instances must have the same runtime representation class. Mixed float-backed and double-backed instances are rejected.

### Numeric-only output

Imputed-only output supports numeric imputed values.

Supported source representations are:

```text
double[]
float[]
Object[] containing Number values
double[][]
float[][]
Object[][] containing Number values
```

A null, string, boolean, or other nonnumeric imputed cell cannot be written to the imputed-only formats.

Categorical or generic mode imputation may be used in compatible PFGAP workflows, but imputed-only export requires the final selected cells to be numeric.

## Standardization reversal

Imputed-only values are written in the original feature scale when standardization state is available.

The output path applies the inverse affine transformation:

```text
original_value = standardized_value * scale + center
```

before writing each missing coordinate.

### Reusable standardization statistics

For reusable global or `per_dimension` statistics:

- `global` uses parameter group `0` for every coordinate;
- one-dimensional `per_dimension` uses the feature or time position as the parameter group; and
- multivariate `per_dimension` uses the source dimension as the parameter group.

The reusable center and scale counts must match the expected output groups.

### Per-series standardization state

For per-series transformation, each instance can supply one local standardization state.

If that state has one parameter group, group `0` is used throughout the series. Otherwise:

- one-dimensional output uses the feature or time position; and
- multivariate output uses the source dimension.

The per-series state count must equal the dataset instance count.

### Standardization-source exclusivity

Supply either:

- reusable dataset-level statistics; or
- per-series standardization states.

Supplying both is rejected.

If neither is supplied, values are written without inverse transformation.

## Finite-value requirement

Every imputed-only value must be finite after inverse transformation.

The output path rejects:

```text
NaN
Infinity
-Infinity
```

A non-finite result indicates incomplete imputation or an invalid inverse transformation.

For float-backed CSR output, the inverse-transformed double value must also remain finite after narrowing to `float`.

## Empty datasets and empty results

An empty dataset cannot produce an imputed-only result.

A nonempty dataset with no originally missing coordinates can produce an artifact with:

```text
entry_count = 0
```

Its matrix shape or tensor envelope still describes the source bounds.

## Validation rules

PFGAP validates the imputed-only result before and during writing.

The following shared conditions must hold:

- dataset instance count matches the missing-index instance count;
- dataset dimensionality matches the missing-index dimensionality;
- source instance runtime types are homogeneous;
- per-series state count matches dataset size when supplied;
- standardization parameter counts match the required groups;
- every emitted coordinate is within its output envelope;
- every emitted value is numeric and finite; and
- the emitted entry count equals the declared missing-value count.

For CSR output, PFGAP additionally validates:

- row and column counts are nonnegative;
- `rowOffsets` has length `rowCount + 1`;
- the first row offset is zero;
- the final row offset equals the entry count;
- row offsets do not decrease;
- value count equals column-index count;
- every column lies within the matrix shape; and
- columns are strictly increasing within a row.

For sparse tensor output, PFGAP additionally validates:

- instance, dimension, and time envelope counts are nonnegative;
- the declared entry count does not exceed the rectangular envelope;
- each instance, dimension, and time coordinate is globally in bounds; and
- `MissingIndices` coordinates exist in the corresponding source instance and dimension.

The output path also rejects a traversal that produces a different number of values than expected from the missing-index metadata.

## Reading the `.mtx` file

A Matrix Market reader returns the imputed matrix coordinates directly.

To apply the output back to source data:

- read the Matrix Market shape and entries;
- convert each one-based row and column to zero-based indices;
- use the row as the source instance index;
- use the column as the source feature or time position; and
- write the provided value only at that originally missing source coordinate.

Do not fill every absent matrix coordinate with zero. Absence means that the coordinate was not exported as an originally missing cell.

### Python reading example

A common Python reader is `scipy.io.mmread`:

```python
from scipy.io import mmread

imputed = mmread(
    "training_imputed_values.mtx"
).tocoo()

for row, column, value in zip(
    imputed.row,
    imputed.col,
    imputed.data,
):
    print(row, column, value)
```

SciPy exposes zero-based row and column arrays after reading the one-based Matrix Market file.

Retain explicit-zero entries. Do not call an operation that eliminates zeros if entry presence is being used to identify originally missing coordinates.

### One-dimensional reconstruction example

```python
from scipy.io import mmread

updates = mmread(
    "training_imputed_values.mtx"
).tocoo()

for instance, position, value in zip(
    updates.row,
    updates.col,
    updates.data,
):
    dataset[instance][position] = value
```

This example assumes the source is tabular or univariate and the output shape matches the source envelope.

## Reading the `.tns` file

The `.tns` file contains three coordinates followed by the imputed value.

To apply the output back to source data:

- ignore comment lines beginning with `%`;
- read the instance, dimension, time, and value columns;
- convert each one-based coordinate to a zero-based index;
- validate the coordinate against the corresponding original instance and dimension; and
- write the value only at that originally missing coordinate.

Do not create a complete rectangular array unless that is appropriate for the source dataset. For ragged data, apply the patch directly to the original per-instance representation.

### Python reading example

```python
import numpy as np

entries = np.loadtxt(
    "training_imputed_values.tns",
    comments="%",
    ndmin=2,
)

instances = entries[:, 0].astype(np.int64) - 1
dimensions = entries[:, 1].astype(np.int64) - 1
times = entries[:, 2].astype(np.int64) - 1
values = entries[:, 3]

for instance, dimension, time, value in zip(
    instances,
    dimensions,
    times,
    values,
):
    dataset[instance][dimension][time] = value
```

For a zero-entry tensor patch, inspect the `% entries:` comment or handle the empty `numpy.loadtxt` result explicitly.

## Difference from sparse proximities

Imputed-only output and sparse proximity matrices have different zero semantics.

### Imputed-only output

- Entry presence means the source cell was originally missing.
- Exact-zero values are retained.
- Omitted coordinates were not exported as missing cells.

### Sparse proximity output

- Entry presence represents a stored nonzero proximity.
- Retained exact-zero entries are rejected.
- Omitted coordinates represent numeric zero.

Do not process both file types with the same zero-elimination assumptions.

## Common problems

### The output file is not created

Enable the corresponding imputation operation and output request. Ensure the parent path is writable.

### The configured extension changes

This is expected. PFGAP selects `.mtx` for tabular or univariate data and `.tns` for multivariate data, replacing the supplied final extension if necessary.

### The entry count is smaller than the matrix or tensor envelope

This is expected. Only originally missing coordinates are stored.

### The output contains an explicit zero entry

This is valid. The corresponding source coordinate was originally missing and its final imputed value is zero.

### A multivariate time coordinate is beyond another instance's length

The tensor header records a global maximum envelope. Validate each coordinate against its own source instance and dimension rather than assuming every series reaches the global maximum time length.

### Float output fails after inverse standardization

The inverse-transformed value cannot be represented as a finite float. Use compatible data, statistics, and values.

### Writing reports a non-finite value

Imputation is incomplete or inverse standardization produced an invalid value. Every exported imputation must be finite.

### A generic value cannot be exported

Imputed-only output is numeric. Use a complete generic-data writer for nonnumeric imputed values.

### An external sparse library removes entries

Preserve explicit zero entries. Removing zeros destroys the original-missing-coordinate mask represented by entry presence.

### Multivariate reconstruction is ambiguous without the source data

This is expected. The `.tns` artifact is an imputed-only patch, not a standalone encoding of every ragged source length. Retain the original dataset when applying the patch.

## Documentation update responsibility

If the imputed-only representation, coordinate mapping, metadata contract, or output format changes, update this reference together with:

- [Imputation](../guides/Imputation.md);
- [Missing Values](Missing_Values.md);
- [Outputs](Outputs.md);
- [Configuration Reference](Configuration_Reference.md);
- [CLI Reference](CLI_Reference.md); and
- [Writers](../data/Writers.md).

## Related documentation

- [Imputation](../guides/Imputation.md)
- [Missing Values](Missing_Values.md)
- [Outputs](Outputs.md)
- [Standardization](../guides/Standardization.md)
- [Configuration Reference](Configuration_Reference.md)
- [CLI Reference](CLI_Reference.md)
- [Writers](../data/Writers.md)
