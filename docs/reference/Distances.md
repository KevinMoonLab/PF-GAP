# Distances

This reference lists the distance identifiers accepted by PFGAP, explains how distance sets are selected at tree splits, and summarizes the intended data families for the registered measures.

PFGAP does not automatically choose a compatible distance from an arbitrary data representation. Configure distances that match the logical observation type produced by the selected reader.

For custom implementations, see [Custom Distances](../extensions/Custom_Distances.md). For representation details, see [Dataset Representations](../data/Dataset_Representations.md).

## Configure forest distances

In Python:

```python
distances=["dtw", "erp", "lcss"]
```

Direct Java form:

```text
-distances=[dtw,erp,lcss]
```

The registry keys are case-sensitive. Use the identifiers exactly as listed on this page.

## Default distance set

When no user distance set is active, the application default set is:

```text
euclidean
dtw
dtwcv
ddtw
ddtwcv
wdtw
wddtw
lcss
erp
twe
msm
```

These defaults are primarily univariate numeric time-series measures, with Euclidean distance also suitable for fixed-width numeric vectors.

## Distance selection at splits

PFGAP considers `r` candidate distance configurations at each split:

```python
r=5
```

For every candidate:

1. PFGAP selects a distance from the configured distance set.
2. The selected distance receives its own parameter state.
3. Applicable distance parameters are randomized from the node data.
4. Candidate exemplars and branch assignments are evaluated.
5. The candidate with the best split score is retained.

When distance selection occurs per node, each candidate chooses from the configured set at that node. When the tree uses one distance type, every candidate still receives an independent evaluator and independently selected parameter state.

A fitted splitter stores its selected distance, parameters, exemplars, and selected dimension subset for later prediction.

## One-dimensional numeric distances

These identifiers are intended for one-dimensional numeric observations such as univariate time series or, where noted, fixed-width tabular vectors.

### Vector and pointwise distances

```text
euclidean
shifazEUCLIDEAN
manhattan
cosine
```

- `euclidean` and `shifazEUCLIDEAN` use the same Euclidean implementation.
- `manhattan` uses Manhattan distance.
- `cosine` uses cosine distance.
- These measures are appropriate for aligned fixed-width vectors. Use compatible lengths and feature ordering.

### Dynamic time warping family

```text
dtw
shifazDTW
dtwcv
shifazDTWCV
ddtw
shifazDDTW
ddtwcv
shifazDDTWCV
wdtw
shifazWDTW
wddtw
shifazWDDTW
```

- `dtw` and `shifazDTW` use full-window DTW.
- `dtwcv` and `shifazDTWCV` choose a random DTW window for each candidate.
- `ddtw` and `shifazDDTW` use full-window derivative DTW.
- `ddtwcv` and `shifazDDTWCV` choose a random derivative-DTW window.
- `wdtw` and `shifazWDTW` choose a random weighting parameter.
- `wddtw` and `shifazWDDTW` combine derivative and weighted DTW and choose a random weighting parameter.

The `shifaz...` names in this group are aliases that instantiate the same current implementation as their corresponding canonical names.

### Other elastic distances

```text
erp
shifazERP
lcss
shifazLCSS
msm
shifazMSM
twe
shifazTWE
```

- ERP chooses a random gap value and window.
- LCSS chooses a random epsilon and window.
- MSM chooses a random move-split-merge cost.
- TWE chooses random lambda and nu parameters.
- Each `shifaz...` identifier listed here is an alias for the same current implementation as the corresponding canonical identifier.

### Shape-based univariate distance

```text
shapeHoG1dDTW
```

Uses the registered one-dimensional shape-HoG DTW implementation.

## Registered legacy univariate identifiers

The registry also accepts:

```text
basicDTW
dtwDistance
dtwDistanceEfficient
pdtw
scdtw
francoisDTW
smoothDTW
dca_dtw
equality
shifazDTWCV
shifazDDTWCV
shifazWDTW
shifazWDDTW
```

Not every `MEASURE` value registered under a legacy name has a current dispatch case in `DistanceMeasure`. Prefer the canonical, actively dispatched identifiers documented in the preceding sections unless a project-specific workflow already uses and verifies a legacy identifier.

## Multivariate distance conventions

PFGAP uses two multivariate naming conventions.

### Independent distances: `_i`

An independent distance evaluates corresponding dimensions independently and combines their contributions.

Registered independent identifiers are:

```text
dtw_i
ddtw_i
shifazDDTW_I
wdtw_i
shifazWDTW_I
wddtw_i
shifazWDDTW_I
twe_i
shifazTWE_I
erp_i
shifazERP_I
euclidean_i
shifazEUCLIDEAN_I
lcss_i
shifazLCSS_I
msm_i
shifazMSM_I
manhattan_i
shifazMANHATTAN_I
cid_i
shifazCID_I
sbd_i
shifazSBD_I
```

The `shifaz..._I` identifiers are aliases for the corresponding current independent implementations.

### Dependent distances: `_d`

A dependent distance treats the multivariate observation jointly across dimensions at each aligned time position.

Registered dependent identifiers are:

```text
dtw_d
ddtw_d
wdtw_d
wddtw_d
shapeHoGdtw_d
```

`euclidean_d` and `manhattan_d` are not registered public distances.

## Multivariate time-series distances

### Independent DTW family

```text
dtw_i
ddtw_i
shifazDDTW_I
wdtw_i
shifazWDTW_I
wddtw_i
shifazWDDTW_I
```

These are the independent multivariate DTW, derivative-DTW, weighted-DTW, and weighted derivative-DTW variants.

### Dependent DTW family

```text
dtw_d
ddtw_d
wdtw_d
wddtw_d
```

These are the dependent multivariate DTW variants.

### Independent elastic measures

```text
twe_i
shifazTWE_I
erp_i
shifazERP_I
lcss_i
shifazLCSS_I
msm_i
shifazMSM_I
```

These use the same candidate-level parameter families as their univariate counterparts.

### Independent vector and shape distances

```text
euclidean_i
shifazEUCLIDEAN_I
manhattan_i
shifazMANHATTAN_I
cid_i
shifazCID_I
sbd_i
shifazSBD_I
```

These provide independent multivariate Euclidean, Manhattan, complexity-invariant, and shape-based distance variants.

### Shape-HoG multivariate distances

```text
shapeHoGdtw
shifazShapeHoGDTW
shapeHoGdtw_d
```

`shapeHoGdtw` and `shifazShapeHoGDTW` use the same current multivariate Shape-HoG DTW implementation. `shapeHoGdtw_d` is the registered dependent variant.

## Unequal-length observations

DTW-family measures use a full window when their configured window is nonpositive, and the distance implementations receive both input observations. The current multivariate DTW paths determine time length from the inner arrays rather than from the number of dimensions.

Use formats and representations that preserve each observation's realized time length. Within one multivariate observation, dimensions must satisfy the shape contract required by the selected distance.

Fixed-width vector distances such as Euclidean, Manhattan, and cosine require aligned compatible vector positions.

## Missing-compatible distances

The registered missing-aware distances are:

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

### One-dimensional missing-aware distances

```text
nan_euclidean
dtwarow
```

### Independent multivariate missing-aware distances

```text
nan_euclidean_i
dtwarow_i
```

### Dependent multivariate missing-aware distance

```text
dtwarow_d
```

These are the only identifiers accepted by `missing_proximity_distances` for proximity-first imputation:

```python
imputation_initialization="proximity_first"
missing_proximity_distances=["nan_euclidean"]
```

Direct Java form:

```text
-imputation_initialization=proximity_first
-missing_proximity_distances=[nan_euclidean]
```

Ordinary candidate distances are configured separately through `distances`.

See [Imputation](../guides/Imputation.md) and [Missing Values](Missing_Values.md).

## KNN-imputer distances

KNN initialization uses a separate distance set:

```python
initial_imputer="knn"
knn_distances=["dtw", "erp"]
```

Direct Java form:

```text
-initial_imputer=knn
-knn_distances=[dtw,erp]
```

KNN distances are instantiated through the same distance machinery. Select measures that support the representation supplied to the initial imputer. The KNN initializer currently uses five neighbors.

## Dimension subsampling

Node-level dimension subsampling passes a selected dimension subset to distance implementations that support it.

The current selected-dimension dispatch supports:

```text
euclidean
shifazEUCLIDEAN
manhattan
cosine
dtw_i
dtw_d
ddtw_i
shifazDDTW_I
ddtw_d
wdtw_i
shifazWDTW_I
wdtw_d
wddtw_i
shifazWDDTW_I
wddtw_d
shapeHoGdtw_d
cid_i
shifazCID_I
sbd_i
shifazSBD_I
msm_i
shifazMSM_I
twe_i
shifazTWE_I
erp_i
shifazERP_I
lcss_i
shifazLCSS_I
euclidean_i
manhattan_i
dtwarow_i
dtwarow_d
nan_euclidean_i
javadistance
```

A custom Java distance used with dimension subsampling must implement the selection-aware custom-distance interface.

If dimension subsampling is enabled with a distance not handled by the selected-dimension dispatch, PFGAP rejects the distance calculation rather than silently using every dimension.

For a one-dimensional array, dimension selection treats array positions as tabular features. Do not enable dimension subsampling when positions represent ordinary univariate time points.

## Graph distances

Registered graph-distance keys are case-sensitive:

```text
ApproximateGraphEditDistance
GraphEditDistance
GraphletDistance
HammingDistance
ShortestPathDistance
WLDistance
WLDistance2
```

They map to the graph measures:

```text
approximateGraphEditDistance
graphEditDistance
graphletDistance
hammingDistance
shortestPathDistance
wlDistance
wlDistance2
```

Use graph representations accepted by the corresponding implementation. Graph distances are not numeric-vector or time-series distances.

## Interoperability distances

PFGAP registers:

```text
python
maple
javadistance
```

These measures require an implementation descriptor rather than the bare registry key alone in ordinary CLI use.

### Python

```text
python:path/to/file[:FunctionName]
```

### Maple

```text
maple:path/to/file[:FunctionName]
```

### Java

```text
javadistance:path/to/class-or-jar[:ClassName]
```

The referenced file must exist when command-line arguments are parsed. See [Custom Distances](../extensions/Custom_Distances.md).

## Meta distances

Registered meta-distance types are:

```text
meta_classmatch
meta_file_classmatch
meta_regression
meta_file_regression
```

They compare observations through predictions produced by an external pretrained model or prediction source.

Use descriptor syntax:

```text
meta_classmatch:path/to/file[:method]
meta_file_classmatch:path/to/file[:method]
meta_regression:path/to/file[:method]
meta_file_regression:path/to/file[:method]
```

The external file must exist when arguments are parsed.

## Candidate parameters

PFGAP stores parameter values on each selected `DistanceMeasure` candidate.

### Window parameters

Candidate state includes windows for:

```text
dtw
ddtw
lcss
erp
```

Full-window DTW and DDTW use a nonpositive configured window. Cross-validated variants choose random windows from the data.

### Threshold and gap parameters

Candidate state includes:

```text
epsilonLCSS
gERP
```

LCSS selects epsilon and a window. ERP selects a gap value and a window.

### TWE parameters

Candidate state includes:

```text
nuTWE
lambdaTWE
```

### MSM parameter

Candidate state includes:

```text
cMSM
```

### Weighted-DTW parameters

Candidate state includes:

```text
weightWDTW
weightWDDTW
```

Parallel workers receive independent evaluators with copies of the selected candidate parameters. Parameters are not randomized again during branch assignment.

## Early abandoning and invalid results

Distance dispatch accepts a `bestSoFar` value for implementations that support early abandoning.

A distance may return positive infinity, including when early abandoning determines that a candidate cannot improve the current result. `NaN` is invalid and causes PFGAP to reject the distance result.

When multiple exemplars have exactly equal minimum distances, PFGAP selects among the tied branches using its seeded random source.

## Lazy materialization

Before built-in distance computation, PFGAP resolves stored lazy references to their materialized series representations. Candidate exemplars are materialized once per candidate assignment operation, and each query is materialized once before comparison with those exemplars.

A custom Java distance may implement the lazy-distance interface to receive stored representations and a resolver. Selection-aware lazy custom distances are not part of the current custom interface.

Reader behavior and materialized representation types are documented in [Readers](../data/Readers.md) and [Dataset Representations](../data/Dataset_Representations.md).

## Complete registry key list

The following keys are registered and may be recognized by command-line distance parsing. Keys are case-sensitive.

### Univariate and vector keys

```text
basicDTW
dtwDistance
dtwDistanceEfficient
erp
lcss
msm
pdtw
scdtw
twe
wdtw
francoisDTW
smoothDTW
dtw
dca_dtw
equality
dtwcv
ddtwcv
wddtw
ddtw
shifazDTW
shifazDTWCV
shifazDDTW
shifazDDTWCV
shifazWDTW
shifazWDDTW
shifazERP
shifazMSM
shifazLCSS
shifazTWE
shapeHoG1dDTW
euclidean
manhattan
cosine
shifazEUCLIDEAN
```

### Interoperability keys

```text
maple
python
javadistance
```

### Independent multivariate keys

```text
dtw_i
ddtw_i
shifazDDTW_I
wdtw_i
shifazWDTW_I
wddtw_i
shifazWDDTW_I
twe_i
shifazTWE_I
erp_i
shifazERP_I
euclidean_i
shifazEUCLIDEAN_I
lcss_i
shifazLCSS_I
msm_i
shifazMSM_I
manhattan_i
shifazMANHATTAN_I
cid_i
shifazCID_I
sbd_i
shifazSBD_I
```

### Dependent and shape-based multivariate keys

```text
dtw_d
ddtw_d
wdtw_d
wddtw_d
shapeHoGdtw_d
shapeHoGdtw
shifazShapeHoGDTW
```

### Missing-compatible keys

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

### Graph keys

```text
ApproximateGraphEditDistance
GraphEditDistance
GraphletDistance
HammingDistance
ShortestPathDistance
WLDistance
WLDistance2
```

### Meta-distance keys

```text
meta_classmatch
meta_file_classmatch
meta_regression
meta_file_regression
```

## Common errors

### Unknown distance

Use an exact registry key. Distance names are case-sensitive.

### A registered legacy key fails during initialization

Use an actively dispatched canonical distance from the main family sections. Registry presence and enum presence do not by themselves provide a current implementation dispatch case.

### A one-dimensional distance receives a matrix

Select an `_i` or `_d` multivariate measure, or use a reader that produces the intended one-dimensional representation.

### A multivariate distance receives a vector

Select the corresponding one-dimensional measure or correct the reader and `data_dimension` configuration.

### A vector distance receives unequal widths

Use observations with aligned compatible positions, or choose an elastic sequence distance whose implementation supports the required lengths.

### Proximity-first imputation rejects a distance

Use one of the five registered missing-compatible distances accepted by `missing_proximity_distances`.

### KNN initialization has no distances

Supply `knn_distances` when `initial_imputer="knn"`.

### Dimension subsampling rejects a distance

Choose a distance listed in the dimension-subsampling section, use a compatible selection-aware custom Java distance, or disable dimension subsampling.

### A custom descriptor file cannot be found

Use a path that resolves from the process working directory.

### A distance returns `NaN`

Correct the input, parameter, or custom-distance implementation. `NaN` is not an accepted distance result.

## Related documentation

- [Configuration Reference](Configuration_Reference.md)
- [CLI Reference](CLI_Reference.md)
- [Dataset Representations](../data/Dataset_Representations.md)
- [Readers](../data/Readers.md)
- [Imputation](../guides/Imputation.md)
- [Missing Values](Missing_Values.md)
- [Custom Distances](../extensions/Custom_Distances.md)
