# Dataset Representations

PFGAP supports several in-memory observation representations so that the same dataset, proximity, learning, imputation, and output infrastructure can work with tabular data, univariate sequences, multivariate sequences, structured numeric observations, and generic observations.

This page defines the representation-level contract used throughout the PFGAP documentation. It describes the shape and meaning of an observation after it has been read. External storage layouts and reader selection are documented separately.

> **Status: Stable for v1, with identified verification items.** Primitive numeric arrays, generic object arrays, unequal-length sequential observations, primitive missing-value handling, and eager or lazy materialization are intended parts of the v1 data contract. Individual readers and algorithms may support only a subset of these representations.

## 1. Representation and file format are different concepts

A **dataset representation** describes the in-memory shape and element type presented to PFGAP algorithms. Examples include:

```text
double[]
float[]
Object[]
double[][]
float[][]
Object[][]
```

A **file format** describes how observations are stored outside the application. Examples include:

- Delimited wide files
- Delimited long-format files
- NPY
- HDF5
- Parquet
- Per-file datasets
- TS files
- Custom Java sources

A reader connects these concepts:

```text
external file or source
        |
        v
      reader
        |
        v
in-memory dataset representation
```

Multiple file formats may produce the same in-memory representation. Conversely, one format may support more than one representation depending on its schema and selected reader options.

For example, one delimited row may become either:

```text
double[dimension]
```

for tabular data, or:

```text
double[time]
```

for a univariate sequence. These are both represented by the Java type `double[]`; their semantics come from the dataset and reader contract.

## 2. Dataset-level model

Conceptually, a PFGAP dataset contains:

- A sequence of observations
- Optional labels or targets
- Dataset metadata
- Missing-value metadata where applicable
- Reader or resource state for lazy datasets
- Optional fitted transformation state

The concrete dataset classes manage these components. Algorithms should normally access observations through the dataset abstraction rather than assume that every dataset has already been materialized into one large array.

An observation representation describes one observation, not the complete dataset container.

## 3. Supported observation families

PFGAP separates standard observation representations into two broad families:

- **Numeric observations**, stored in primitive floating-point arrays
- **Generic observations**, stored in object arrays

This distinction affects:

- Reader compatibility
- Missing-value representation
- Distance compatibility
- Standardization
- Imputation
- Memory use
- Vectorization
- Output support

Within each family, observations may be one-dimensional or two-dimensional.

## 4. One-dimensional observations

A one-dimensional Java array can represent either tabular data or an ordered sequence. The array type alone does not encode which interpretation applies.

### 4.1 One-dimensional tabular observations

For tabular data, the array index identifies a feature or dimension:

```text
observation[dimension]
```

Standard numeric forms are:

```text
float[dimension]
double[dimension]
```

The standard generic form is:

```text
Object[dimension]
```

Example numeric tabular observation:

```java
double[] observation = {
        32.0,  // age
        1.82,  // height
        78.5   // weight
};
```

Example generic tabular observation:

```java
Object[] observation = {
        "blue",
        "medium",
        12
};
```

For tabular data:

- Feature order is part of the dataset schema.
- Every observation normally has the same feature count.
- Array indices do not represent time.
- Unequal array lengths normally indicate incompatible schemas unless a specialized workflow explicitly supports variable-width observations.

### 4.2 One-dimensional sequential observations

For a univariate time series or another ordered sequence, the array index identifies time or sequence position:

```text
observation[time]
```

or more generally:

```text
observation[position]
```

Standard numeric forms are:

```text
float[time]
double[time]
```

The standard generic form is:

```text
Object[time]
```

Example numeric sequence:

```java
double[] observation = {
        1.2,
        1.4,
        1.1,
        0.8
};
```

Example generic sequence:

```java
Object[] observation = {
        "red",
        "green",
        "green",
        "blue"
};
```

Sequential observations may have unequal lengths when the selected reader, distance, transformation, imputer, and writer support them.

### 4.3 Distinguishing tabular and sequential semantics

The following Java values have identical runtime types:

```java
double[] tabular = {
        32.0,
        1.82,
        78.5
};

double[] sequence = {
        1.2,
        1.4,
        1.1,
        0.8
};
```

The first may mean `observation[dimension]`; the second may mean `observation[time]`. PFGAP determines the intended semantics from:

- Reader configuration
- Dataset metadata
- Source schema
- Selected distance
- Selected task or workflow

Documentation should therefore use **one-dimensional** when discussing the Java shape and use **tabular** or **univariate sequential** when discussing axis semantics.

## 5. Two-dimensional observations

Two-dimensional observations use dimension-major storage.

### 5.1 Numeric two-dimensional observations

Standard numeric forms are:

```text
double[dimension][position]
float[dimension][position]
```

For a multivariate time series:

```text
double[dimension][time]
float[dimension][time]
```

The outer array contains dimensions, channels, or features. Each inner primitive array contains the ordered values for one dimension.

```java
double[][] observation = {
        {1.0, 1.1, 1.2},
        {4.0, 4.2, 4.1}
};
```

This observation has two dimensions and three positions per dimension.

### 5.2 Generic two-dimensional observations

The standard generic form is:

```text
Object[dimension][position]
```

For a multivariate generic sequence:

```text
Object[dimension][time]
```

Example:

```java
Object[][] observation = {
        {"low", "medium", "high"},
        {"off", "on", "on"}
};
```

The selected distance and downstream workflow must understand the actual element types.

### 5.3 Rectangular two-dimensional observations

A rectangular observation has:

- A nonzero dimension count
- No null dimension arrays
- A nonzero position count where required
- The same position count in every dimension

For example:

```text
Dimension 0 length: 100
Dimension 1 length: 100
Dimension 2 length: 100
```

Built-in per-file readers and per-file validation use this standard rectangular, dimension-major contract.

### 5.4 Ragged two-dimensional observations

The Java array type can express dimensions with different lengths:

```text
Dimension 0: [1.0, 1.1, 1.2, 1.3]
Dimension 1: [4.0, 4.2]
```

Such an observation is **ragged within the observation**.

Whether it is valid for a workflow depends on the selected:

- Reader
- Distance
- Transformation
- Imputer
- Writer

Support for `double[][]`, `float[][]`, or `Object[][]` does not automatically imply support for arbitrary inner-array lengths.

The standard per-file reader contract is rectangular. Specialized non-per-file or custom workflows may support ragged data only when that support is documented explicitly.

## 6. Per-file representation contract

Per-file readers preserve the dimension axis even for univariate data. Their standard representations are:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

A univariate per-file observation is represented as:

```text
float[1][time]
double[1][time]
Object[1][time]
```

A per-file reader must not collapse a univariate observation to:

```text
float[]
double[]
Object[]
```

This rule provides a stable per-file contract and prevents representation changes based only on the number of selected features.

Custom eager and lazy per-file paths validate:

- Non-null output
- Two-dimensional standard representation where standard arrays are used
- At least one dimension
- Non-null dimension arrays
- Nonempty dimensions where required
- Equal inner lengths

This restriction applies specifically to per-file readers. General and non-per-file readers may produce supported one-dimensional tabular or sequential observations.

## 7. Generic and proprietary observations

Generic arrays support categorical, textual, mixed, and other Java object values.

Standard generic representations are:

```text
Object[]
Object[][]
```

An `Object[]` may represent either:

```text
Object[dimension]
```

for tabular data, or:

```text
Object[time]
```

for a univariate generic sequence.

An `Object[][]` normally follows the dimension-major convention:

```text
Object[dimension][position]
```

Custom readers may return a proprietary observation object when:

- Built-in standardization is disabled.
- The configured distance or downstream component understands the type.
- The reader documents the type and lifecycle contract.
- A per-file reader does not return a prohibited one-dimensional standard array.

Returning an arbitrary object does not automatically make it compatible with the rest of PFGAP.

## 8. Boxed numeric arrays are not primitive numeric storage

PFGAP does not use these as standard primitive numeric observation representations:

```text
Double[]
Float[]
Double[][]
Float[][]
```

Numeric readers should produce:

```text
double[]
float[]
double[][]
float[][]
```

Primitive storage:

- Reduces per-value object overhead
- Improves memory locality
- Supports efficient numeric loops
- Enables compatible vectorized implementations
- Uses floating-point `NaN` for missing numeric entries

An `Object[]` or `Object[][]` may contain numeric wrapper objects as part of a genuinely generic dataset, but that does not make it a primitive numeric dataset.

## 9. Precision and numeric storage

Numeric observations may use:

- `float64`, represented by Java `double`
- `float32`, represented by Java `float`
- Automatic selection where supported by the reader pipeline

`NumericStorageType` normally exposes these choices as:

- `AUTO`
- `FLOAT32`
- `FLOAT64`

The selected precision affects:

- Memory use
- Reader materialization cost
- Lazy-cache size
- Arithmetic precision
- Distance implementations
- Standardization state
- Output representation

Algorithms and distances must either support the selected precision directly or use an intentional conversion path.

A lazy reader specification retains the numeric storage choice so that saved-model reconstruction preserves materialization behavior.

## 10. Equal-width, equal-length, and unequal-length datasets

PFGAP does not impose one universal shape rule across all representation families. Shape requirements depend on the semantics and consuming component.

### 10.1 Equal-width tabular data

```text
Observation 0: 12 features
Observation 1: 12 features
Observation 2: 12 features
```

Tabular workflows normally require a consistent feature count and feature ordering.

### 10.2 Variable-width tabular data

```text
Observation 0: 10 features
Observation 1: 12 features
Observation 2: 8 features
```

Although Java arrays can express this, ordinary tabular algorithms normally treat the observations as schema-incompatible. Variable-width tabular data requires a specialized and explicitly documented workflow.

### 10.3 Equal-length one-dimensional sequences

```text
Observation 0: length 100
Observation 1: length 100
Observation 2: length 100
```

### 10.4 Unequal-length one-dimensional sequences

```text
Observation 0: length 80
Observation 1: length 100
Observation 2: length 63
```

### 10.5 Equal-shape multivariate data

```text
Observation 0: 3 dimensions x 100 positions
Observation 1: 3 dimensions x 100 positions
```

### 10.6 Unequal-length multivariate data

```text
Observation 0: 3 dimensions x 80 positions
Observation 1: 3 dimensions x 100 positions
```

Each observation remains internally rectangular, but observation lengths differ.

### 10.7 Ragged multivariate data

```text
Observation 0 dimension lengths: [80, 75, 80]
Observation 1 dimension lengths: [100, 100, 92]
```

These cases are distinct. Documentation should not use “unequal length” ambiguously when the distinction between observation-level unequal lengths and dimension-level raggedness matters.

## 11. Dimension count

For a two-dimensional observation, the number of dimensions is:

```java
int dimensions = observation.length;
```

The representation itself can express different dimension counts across observations. However, many multivariate distances and transformations require:

- A consistent dimension count throughout the dataset
- Matching dimension counts between observations being compared
- Consistent feature meaning and ordering

Unless a workflow explicitly documents variable dimension counts, users should provide a consistent count across observations.

For one-dimensional tabular observations, the array length is the feature or dimension count.

For one-dimensional sequences, the array length is the sequence length.

## 12. Time, feature, and position semantics

Primitive arrays encode positions, but the meaning of a position depends on the representation contract.

### 12.1 Tabular semantics

For one-dimensional tabular data:

```text
observation[index] = feature value
```

The index identifies a feature or dimension. It is not a time coordinate.

### 12.2 Sequential semantics

For one-dimensional sequential data:

```text
observation[index] = value at an ordered position
```

For two-dimensional sequential data:

```text
observation[dimension][index] = value for one dimension at an ordered position
```

Array indices preserve order but do not necessarily represent original real-valued timestamps.

Depending on the reader and source format, an observation may have:

- Implicit positions only
- Explicit source time values preserved as metadata
- IDs and time coordinates used to assemble the observation but not retained as algorithm inputs
- Irregular sampling encoded by a specialized representation or compatible distance

A reader's documentation must state whether explicit time coordinates are preserved, transformed, or discarded. Algorithms operating only on primitive arrays see ordered positions unless additional metadata is supplied through a supported mechanism.

> **DOC-VERIFY:** Record time-coordinate preservation behavior for each long-format and per-file reader in the reader capability reference.

## 13. Labels and targets

Labels and targets are stored separately from observation values in the dataset abstraction, even when they originate from a column embedded in the same input file.

### 13.1 Classification targets

Classification labels may use supported object values, including integer-like or string labels. Algorithms must not assume class labels are contiguous zero-based integers.

### 13.2 Regression targets

Regression uses numeric target values. Reader and configuration rules determine how a target is parsed and stored.

### 13.3 Missing targets

Missing observation values and missing labels or targets are different problems. A reader may support missing feature values without supporting missing training targets.

Task documentation should state whether labels or targets are required for:

- Training
- Validation or evaluation
- Prediction-only workflows
- Outlier scoring
- OOD scoring
- Imputation

## 14. Missing numeric values

Primitive numeric observations represent missing entries with:

```java
Double.NaN
```

or:

```java
Float.NaN
```

This preserves primitive storage while allowing missingness to travel through the numeric pipeline.

```java
double[] observation = {
        1.0,
        Double.NaN,
        2.5,
        3.0
};
```

Readers may accept format-specific missing tokens, but numeric readers should convert supported tokens to primitive `NaN` values.

Numeric algorithms must use NaN-aware checks:

```java
Double.isNaN(value)
Float.isNaN(value)
```

Equality comparisons with `NaN` are not valid missing-value checks.

## 15. Missing generic values

Generic observations normally use `null` for missing values where the reader and workflow support it.

```java
Object[] observation = {
        "red",
        null,
        "blue"
};
```

A custom generic distance or imputer must define how it handles `null`. Generic null behavior should not be inferred from primitive numeric `NaN` behavior.

## 16. Original missing positions

PFGAP distinguishes between:

- The current value stored at a position
- Whether that position was originally missing in the source data

This distinction is necessary because an imputed value is no longer `NaN` or `null`, but the system may still need to:

- Update only originally missing positions in later imputation iterations
- Write only imputed values
- Preserve exact zero imputations
- Compare imputed output with the original missingness pattern

Dataset-level missing-index metadata tracks original missing coordinates independently of current values.

For numeric data, this metadata may use a sparse structure such as CSR. Users should treat it as dataset metadata rather than an alternative observation representation.

## 17. Flattening multivariate coordinates

Some sparse outputs require multidimensional positions to be mapped to matrix coordinates. A complete flattening contract must define:

- Observation index
- Dimension index
- Position within a dimension
- Dimension ordering
- Shape information required to reverse the mapping
- Behavior for unequal-length and ragged observations
- Whether coordinates are zero-based or one-based

Matrix Market files use one-based coordinates. Internal Java indexing ordinarily uses zero-based coordinates.

A flat sparse matrix alone may not preserve enough information to reconstruct arbitrary ragged multivariate observations. Portable output therefore needs appropriate shape metadata.

> **Status: Provisional.** Global sparse imputed-only output exists, but portable packaging for arbitrary ragged multivariate reconstruction remains a pre-v1 design item.

## 18. Eager and lazy datasets

Representation shape is independent of materialization strategy.

### 18.1 Eager datasets

An eager dataset materializes observations during reading and retains them in memory for later access.

Examples include an eagerly stored:

```text
double[dimension]
```

or:

```text
float[dimension][time]
```

### 18.2 Lazy datasets

A lazy dataset retains sufficient reference information and reader state to materialize observations on demand.

A lazy reader must still return a supported observation representation when an algorithm requests it. For example, a lazy numeric multivariate reader may return one `double[][]` without holding all observations in memory simultaneously.

Lazy implementations must define:

- Resource ownership and closing behavior
- Thread-safety or synchronized-access behavior
- Caching policy
- Error handling during deferred reads
- Shape and type metadata available before materialization
- Interactions with standardization and imputation
- Saved-model reconstruction behavior

These concerns are not encoded in the array type itself.

### 18.3 Lazy references

`LazySeriesRef` identifies an observation using:

- A reader registry key
- An observation index
- A file or other source locator

The referenced materializer determines whether the observation is tabular, sequential, one-dimensional, or two-dimensional.

## 19. Standardization and transformed views

Standardization changes numeric values but should not change logical observation shape.

It supports standard primitive representations:

```text
float[]
double[]
float[][]
double[][]
```

This includes:

- One-dimensional tabular feature vectors
- One-dimensional univariate sequences
- Two-dimensional dimension-major numeric observations

Prepared transformation state may be applied:

- Eagerly during reading
- During lazy observation access
- Through a transformed dataset view
- In reverse when writing output in original units

Fitted statistics belong to transformation state, not the primitive array type. Training-derived statistics used for evaluation data must be reused according to the selected standardization scope and method.

Generic observations are not automatically eligible for numeric standardization merely because some elements are numeric wrapper objects.

Custom proprietary representations are not automatically standardizable.

## 20. Distance compatibility

A distance must support the representation and semantics it receives.

A distance contract should state:

- Numeric or generic input
- Tabular, univariate sequential, or multivariate interpretation
- Supported precision
- Equal-length requirements
- Unequal-length support
- Ragged-dimension support
- Missing-value behavior
- Thread-safety expectations

A `double[]` distance intended for tabular vectors may differ conceptually from a `double[]` distance intended for ordered time series, even though both accept the same Java type.

Independent and dependent multivariate distances may interpret dimensions differently while receiving the same `double[][]` or `float[][]` representation.

Users should not select a distance solely by name or parameter type. They should confirm that its semantics and shape requirements match the selected reader and dataset.

## 21. Reader responsibilities

A dataset reader translates an external source into a valid dataset representation. Depending on the reader, this includes:

- Parsing values
- Selecting numeric or generic storage
- Selecting single or double precision
- Constructing tabular, univariate sequential, or multivariate observations
- Preserving unequal lengths where supported
- Preserving a stable tabular feature order
- Converting missing numeric tokens to `NaN`
- Recording original missing positions
- Parsing labels or targets
- Recording IDs, names, shape metadata, or time metadata
- Preparing eager storage or lazy access state
- Validating unsupported schemas and combinations

A reader should fail with a clear message when the requested representation cannot safely express the source data.

Per-file readers have the additional responsibility of preserving the two-dimensional dimension-major contract, including for univariate data.

## 22. Writer responsibilities

A writer translates dataset observations into an external format. Writers must not assume that every reader has a mirror-image writer.

A writer contract should state:

- Supported observation families
- Supported dimensionality
- Tabular or sequential semantics
- Equal-length or ragged-data restrictions
- Label or target handling
- Missing-value serialization
- Time and ID behavior
- Whether inverse transformation is applied
- Whether complete observations or imputed-only values are written
- Whether lazy observations can be streamed

Reader support and writer support are documented separately because their capability sets are intentionally not identical.

## 23. Custom readers and representations

A custom reader should normally produce one of the standard representation families described on this page. Doing so allows compatible distances, transformations, imputers, and writers to operate on its output.

Custom reader documentation should include:

- Produced Java type
- Numeric, generic, or proprietary classification
- Tabular, univariate sequential, or multivariate interpretation
- Precision
- Shape guarantees
- Missing-value representation
- Label or target behavior
- Eager or lazy behavior
- Resource ownership
- Compatible distances and workflows

### 23.1 Custom per-file readers

A custom per-file reader must return a two-dimensional standard representation when returning standard arrays:

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

It must not return:

```text
float[]
double[]
Object[]
```

A univariate per-file result is `[1][time]`.

### 23.2 Custom non-per-file readers

A non-per-file custom reader may return:

- A one-dimensional tabular observation
- A one-dimensional univariate sequence
- A two-dimensional observation
- A documented proprietary representation

The selected downstream workflow must support the result.

## 24. Representation capability summary

### 24.1 Primitive numeric

```text
double[]
float[]
double[][]
float[][]
```

Semantics:

- `double[]`: one-dimensional numeric, double precision; tabular feature vector or univariate sequence
- `float[]`: one-dimensional numeric, single precision; tabular feature vector or univariate sequence
- `double[][]`: two-dimensional dimension-major numeric, double precision
- `float[][]`: two-dimensional dimension-major numeric, single precision
- Numeric missing value: primitive `NaN`

### 24.2 Generic

```text
Object[]
Object[][]
```

Semantics:

- `Object[]`: one-dimensional generic; tabular feature vector or univariate sequence
- `Object[][]`: two-dimensional dimension-major generic
- Generic missing value: normally `null` where supported

### 24.3 Standard per-file output

```text
float[dimension][time]
double[dimension][time]
Object[dimension][time]
```

Univariate per-file output:

```text
float[1][time]
double[1][time]
Object[1][time]
```

### 24.4 Not standard primitive numeric storage

```text
Double[]
Float[]
Double[][]
Float[][]
```

## 25. Validation checklist

Before launching a PFGAP workflow, verify:

- Does the reader produce numeric, generic, or proprietary observations?
- Is the observation tabular, univariate sequential, or multivariate?
- Is the Java shape one-dimensional or two-dimensional?
- Does a one-dimensional axis represent features or ordered positions?
- Is the numeric precision supported by the selected distance?
- Are tabular feature counts and ordering consistent?
- Are sequential observation lengths equal or unequal?
- Are multivariate observations internally rectangular or ragged?
- Is the dimension count consistent where required?
- How are missing values represented?
- Are original missing coordinates recorded if imputation output is required?
- Are labels or targets embedded, separate, or absent?
- Does the selected task require evaluation labels?
- Does the selected distance support the representation, semantics, and shape?
- Does standardization support the representation?
- Does the requested writer support the representation?
- If the data is per-file, does every observation preserve the two-dimensional dimension axis?
- If access is lazy, are resource ownership and concurrency behavior supported?

## 26. Related documentation

- [Documentation home](../index.md)
- [Data Formats](Data_Formats.md)
- [Readers](Readers.md)
- `Distances.md`
- `Imputation.md`
- `Writers.md`
- `Missing_Values.md`
- `Standardization.md`
- `Eager_and_Lazy_Data.md`
- `Imputed_Only_Output.md`
