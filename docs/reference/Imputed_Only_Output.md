# Imputed-Only Output

This reference defines PFGAP's imputed-only output: a Matrix Market coordinate file containing final numeric values only at positions that were missing in the original input. It describes the output request, matrix shape, coordinate mapping, data types, exact-zero handling, standardization reversal, source-shape metadata, and validation rules.

For the full imputation workflow, see [Imputation](../guides/Imputation.md). For general artifact naming and Matrix Market conventions, see [Outputs](Outputs.md).

## Purpose

A complete imputed dataset contains both originally observed values and final imputed values. Imputed-only output instead contains one entry for every originally missing feature coordinate and no entries for originally observed coordinates.

This format is useful when the consumer already has the source data and needs only the replacement values produced by PFGAP.

## Request training output

Python:

```python
return_imputed_training_csr=True
training_imputed_csr_file=(
    "../output/training_imputed_values.mtx"
)
```

Direct Java:

```text
-output_train_imputed_csr=true
-train_imputed_csr_file=output/training_imputed_values.mtx
```

The default Java filename is:

```text
training_imputed_values.mtx
```

Requesting this output enables training-data imputation and missing-value handling.

## Request test output

Python:

```python
return_imputed_testing_csr=True
testing_imputed_csr_file=(
    "../output/testing_imputed_values.mtx"
)
```

Direct Java:

```text
-output_test_imputed_csr=true
-test_imputed_csr_file=output/testing_imputed_values.mtx
```

The default Java filename is:

```text
testing_imputed_values.mtx
```

Requesting this output enables test-data imputation and missing-value handling.

## File format

Imputed-only output uses Matrix Market coordinate format:

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

Matrix Market row and column coordinates are one-based. PFGAP's internal source and CSR indices are zero-based and are incremented during writing.

The file is UTF-8 text. The Matrix Market writer creates missing parent directories and replaces an existing target file.

## Entry semantics

An entry means:

> This source feature coordinate was missing in the original input, and this is its final imputed numeric value.

Entry presence is based on original missingness, not on whether the final value is nonzero.

Therefore:

- originally observed cells are omitted;
- originally missing cells are included;
- an imputed value of `0.0` is included; and
- an absent matrix coordinate does not mean an imputed zero.

This differs from conventional numerical sparsity, where zero values are normally omitted.

## Row mapping

Each matrix row corresponds to one dataset instance.

The mapping is:

```text
matrix row = instance index
```

Internally, instances are zero-based. In the `.mtx` file, instance `0` is written as Matrix Market row `1`.

The matrix row count is the dataset instance count, including instances with no missing values.

An instance with no originally missing coordinates contributes no entries but still occupies a matrix row.

## One-dimensional column mapping

For one-dimensional observations, each matrix column corresponds directly to a feature or time position:

```text
column = position
```

The matrix column count is the maximum realized one-dimensional observation length across the dataset.

Internally, position `0` maps to column `0`. In the `.mtx` file, it is written as Matrix Market column `1`.

### Example

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

## Two-dimensional column mapping

For dimension-major two-dimensional observations, PFGAP flattens each `(dimension, position)` coordinate into one global matrix column:

```text
column = dimension * maximumTimeLength + position
```

where:

- `dimension` is the zero-based source dimension;
- `position` is the zero-based position within that dimension; and
- `maximumTimeLength` is the maximum realized dimension length anywhere in the dataset.

The matrix column count is:

```text
maximumDimensionCount * maximumTimeLength
```

where `maximumDimensionCount` is the maximum realized number of dimensions in any instance.

### Example

Suppose:

```text
maximumDimensionCount = 3
maximumTimeLength = 5
```

Then the flattened zero-based column ranges are:

```text
dimension 0 -> columns 0 through 4
dimension 1 -> columns 5 through 9
dimension 2 -> columns 10 through 14
```

A missing value at:

```text
instance 2, dimension 1, position 3
```

maps internally to:

```text
column = 1 * 5 + 3 = 8
```

and is written to Matrix Market row `3`, column `9`.

## Rectangular envelope

The global flattening rule creates one valid rectangular matrix even when instances have:

- different numbers of dimensions;
- different dimension lengths; or
- both unequal dimension counts and unequal lengths.

Some columns in the rectangular envelope do not correspond to a realized source coordinate for a particular instance. They remain absent and must not be interpreted as source cells that were observed, missing, or imputed.

## Source-shape metadata

The in-memory `ImputedValuesCSR` retains the source shape needed to interpret the rectangular matrix correctly.

It stores:

```text
rowCount
columnCount
rowOffsets
columnIndices
values
storageType
twoDimensionalSource
maximumDimensionCount
maximumTimeLength
instanceDimensionOffsets
dimensionLengths
```

### `instanceDimensionOffsets`

This offset array identifies the range of dimension-length entries belonging to each instance.

For instance `i`:

```text
start = instanceDimensionOffsets[i]
end   = instanceDimensionOffsets[i + 1]
```

The dimension count is:

```text
end - start
```

### `dimensionLengths`

This array records the realized length of every dimension of every instance.

For one-dimensional data, each instance contributes one length entry.

For two-dimensional data, each instance contributes one length entry per realized dimension.

### Matrix Market file scope

The `.mtx` file contains the matrix shape and coordinate values. The richer per-instance source-shape arrays belong to the in-memory result object and are not encoded as additional Matrix Market arrays in the coordinate file.

A consumer that needs to reconstruct ragged two-dimensional source coordinates must retain or obtain the corresponding source-shape metadata.

## CSR representation

Before writing, PFGAP builds an immutable compressed sparse row result.

The main CSR arrays are:

- `rowOffsets`, with length `rowCount + 1`;
- `columnIndices`, with one column for each originally missing coordinate; and
- primitive `float[]` or `double[]` values.

For row `r`, its entries occupy:

```text
rowOffsets[r] <= offset < rowOffsets[r + 1]
```

Column indices are strictly increasing within each row.

The CSR entry count is exactly the number of originally missing feature coordinates recorded for the dataset.

## Numeric storage type

The imputed-only result preserves the materialized dataset's primitive numeric storage:

- `FLOAT32` for homogeneous `float[]` or `float[][]` observations; or
- `FLOAT64` for homogeneous `double[]` or `double[][]` observations.

Numeric `Object[]` or `Object[][]` values are read through `Number.doubleValue()` and produce `FLOAT64` output.

`AUTO` is not a materialized CSR data type.

All instances must have the same runtime representation class. Mixed float-backed and double-backed instances are rejected.

## Numeric-only output

Matrix Market imputed-only output supports numeric imputed values.

Supported source representations are:

```text
double[]
float[]
Object[] containing Number values
double[][]
float[][]
Object[][] containing Number values
```

A null, string, boolean, or other nonnumeric imputed cell cannot be written to this output format.

Categorical or generic mode imputation may be used in compatible PFGAP workflows, but imputed-only Matrix Market export requires the final selected cells to be numeric.

## Standardization reversal

Imputed-only values are written in the original feature scale when standardization state is available.

The builder applies the inverse affine transformation:

```text
original_value = standardized_value * scale + center
```

before adding each missing coordinate to the CSR.

## Reusable standardization statistics

For reusable `global` or `per_dimension` statistics:

- `global` uses parameter group `0` for every coordinate;
- one-dimensional `per_dimension` uses the feature or position as the parameter group; and
- two-dimensional `per_dimension` uses the source dimension as the parameter group.

The reusable center and scale counts must match the expected output groups.

## Per-series standardization state

For per-series transformation, each instance can supply one local standardization state.

If that state has one parameter group, group `0` is used throughout the series. Otherwise, two-dimensional output uses the source dimension as the local parameter group.

The per-series state count must equal the dataset instance count.

## Standardization-source exclusivity

Supply either:

- reusable dataset-level statistics; or
- per-series standardization states.

Supplying both is rejected.

If neither is supplied, values are written without inverse transformation.

## Finite-value requirement

Every imputed-only value must be finite after inverse transformation.

The builder rejects:

```text
NaN
Infinity
-Infinity
```

A non-finite result indicates incomplete imputation or an invalid inverse transformation.

For float-backed output, the inverse-transformed double value must also remain finite after narrowing to `float`.

## Empty datasets and empty results

An empty dataset cannot produce an imputed-only result.

A nonempty dataset with no originally missing coordinates can produce a matrix with:

```text
entry_count = 0
```

Its row and column shape still describe the source envelope.

## Validation rules

PFGAP validates the imputed-only result before writing.

The following must hold:

- row and column counts are nonnegative;
- maximum dimension count and time length are nonnegative;
- `rowOffsets` has length `rowCount + 1`;
- the first row offset is zero;
- the final row offset equals the entry count;
- row offsets do not decrease;
- value count equals column-index count;
- every column lies within the matrix shape;
- columns are strictly increasing within a row;
- every value is finite;
- source-shape offsets begin at zero and terminate at the dimension-length count;
- dataset instance count matches the missing-index instance count;
- dataset dimensionality matches the missing-index dimensionality;
- per-series state count matches the dataset size when supplied; and
- source instance runtime types are homogeneous.

The builder also rejects a result whose traversal of original missing coordinates produces a different number of values than expected from the missing-index metadata.

## Reading the `.mtx` file

A Matrix Market reader returns matrix coordinates, not the original ragged shape.

To apply the output back to source data:

1. Read the Matrix Market shape and entries.
2. Convert each one-based row and column to zero-based indices.
3. Use the row as the source instance index.
4. For one-dimensional data, use the column as the source position.
5. For two-dimensional data, calculate:

```text
dimension = column / maximumTimeLength
position  = column % maximumTimeLength
```

6. Verify that the coordinate exists in the corresponding source instance and dimension using the retained source-shape metadata.
7. Write the provided value only at that originally missing source coordinate.

Do not fill every absent matrix coordinate with zero. Absence means that the coordinate was not exported as an originally missing cell.

## Python reading example

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

## One-dimensional reconstruction example

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

This example assumes the source is one-dimensional and the output shape matches the source envelope.

## Two-dimensional coordinate decoding example

```python
from scipy.io import mmread

maximum_time_length = 120
updates = mmread(
    "training_imputed_values.mtx"
).tocoo()

for instance, column, value in zip(
    updates.row,
    updates.col,
    updates.data,
):
    dimension = column // maximum_time_length
    position = column % maximum_time_length
    dataset[instance][dimension][position] = value
```

Use the actual `maximumTimeLength` associated with the built result, and validate each decoded coordinate against the source-shape metadata before assignment.

## Difference from sparse proximities

Both imputed-only returns and sparse proximity matrices can use Matrix Market coordinate files, but their zero semantics differ.

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

### The entry count is smaller than the matrix size

This is expected. Only originally missing coordinates are stored.

### The matrix contains an explicit zero entry

This is valid. The corresponding source coordinate was originally missing and its final imputed value is zero.

### A two-dimensional column appears beyond a particular instance's length

The matrix uses a global rectangular envelope. Decode the column and consult that instance's source-shape metadata. Do not interpret envelope-only coordinates as realized source cells.

### Float output fails after inverse standardization

The inverse-transformed value cannot be represented as a finite `float`. Use compatible data, statistics, and values.

### Matrix Market writing reports a non-finite value

Imputation is incomplete or inverse standardization produced an invalid value. Every exported imputation must be finite.

### A generic value cannot be exported

Imputed-only Matrix Market output is numeric. Use a complete generic-data writer for nonnumeric imputed values.

### An external sparse library removes entries

Preserve explicit zero entries. Removing zeros destroys the original-missing-coordinate mask represented by entry presence.

### Two-dimensional reconstruction is ambiguous

Retain `maximumTimeLength`, `instanceDimensionOffsets`, and `dimensionLengths` with the source data or in-memory result. The `.mtx` coordinate file alone describes the rectangular matrix, not every ragged source shape.

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
