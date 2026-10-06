# CLI Reference

This page defines the direct Java command-line interface for PFGAP. It lists every option currently accepted by `PFApplication`, accepted encodings, defaults from `AppContext`, and option relationships.

For Python-facing names and defaults, see [Configuration Reference](Configuration_Reference.md). For task-oriented commands, see [Quick Start](../getting-started/Quick_Start.md) and the pages in `guides/`.

## Invocation

From the repository root:

```bash
java -jar Application/PFGAP.jar -name=value
```

With an explicit heap limit:

```bash
java -Xmx4g -jar Application/PFGAP.jar -name=value
```

Every PFGAP argument must use the exact form:

```text
-name=value
```

The following forms are invalid:

```text
-name value
-name = value
name=value
```

Option names are case-sensitive. The parser splits each argument at the first equals sign, so a value may contain additional equals signs.

## Value conventions

### Booleans

Boolean options use `true` or `false`. Java boolean parsing is case-insensitive, but this documentation uses lowercase consistently.

### Null paths

The exact text `None` clears supported optional paths such as `-test`, `-train_labels`, and `-test_labels`. Several reader and preprocessing options also treat a blank value or case-insensitive `None` as absent.

### Lists

Distance lists use brackets:

```text
-distances=[dtw,erp,lcss]
```

String-column lists accept either bracketed or unbracketed comma-separated values:

```text
-feature_columns=[temperature,pressure]
```

```text
-feature_columns=temperature,pressure
```

Missing indicators use a bracketed comma-separated list:

```text
-MissingStrings=[,NA,NaN,null,nan,NAN]
```

### Shell quoting

Quote the complete argument when its value contains spaces, wildcards, semicolons, or shell metacharacters:

```bash
'-file_pattern=*.parquet'
'-custom_reader_parameters=encoding=UTF-8;strict=true'
'-train=/path/with spaces/train.csv'
```

Use the quoting rules of the active shell.

## Minimal training command

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -forest_mode=classification \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -out=output/run/
```

## Minimal saved-model evaluation command

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=true \
  -train=data/test.csv \
  -test=data/test.csv \
  -modelname=output/run/model \
  -forest_mode=classification \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -get_predictions=true \
  -out=output/evaluation/
```

The current saved-model evaluation path receives the evaluation data through both `-train` and `-test`.

## Execution and paths

## `-eval`

**Type:** Boolean
**Default:** Set by the invoking workflow
**Values:** `false` for training; `true` for saved-model evaluation

```text
-eval=false
```

## `-train`

**Type:** Path
**AppContext default:** `Data/GunPoint_TRAIN.tsv` under the working directory

Training input during training. During saved-model evaluation, identify the evaluation input here and through `-test`.

## `-test`

**Type:** Path or `None`
**AppContext default:** `Data/GunPoint_TEST.tsv` under the working directory

Optional test or validation input. `None` disables the test input.

## `-train_labels`

**Type:** Path or `None`
**Default:** `None`

Separate training labels or targets.

## `-test_labels`

**Type:** Path or `None`
**Default:** `None`

Separate test or evaluation labels or targets. Supplying a non-`None` path also sets `exists_testlabels` to true.

## `-exists_testlabels`

**Type:** Boolean
**Default:** `false`

Indicates whether known test or evaluation labels or targets are available. A supplied `-test_labels` path takes precedence and keeps this setting true.

## `-out`

**Type:** Directory path
**Default:** `output/`

Destination directory for generated artifacts. Create the directory before direct Java execution where required.

## `-modelname`

**Type:** Model name or saved-model path
**Default:** `Thor`

Names a model during training and identifies the saved model during evaluation.

## `-savemodel`

**Type:** Boolean
**Default:** `false`

Saves the trained model.

## `-repeats`

**Type:** Integer
**Default:** `1`

Number of experiment repetitions.

## `-export`

**Type:** Integer
**Default:** `1`

Artifact export level. Current application levels are `0`, `1`, and `2`.

## `-verbosity`

**Type:** Integer
**Default:** `0`

Console-reporting level. Current application levels are `0`, `1`, and `2`.

## Task selection

## `-forest_mode`

**Type:** Enumerated string
**Default:** `classification`
**Values:** `classification`, `regression`, `isolation`

Selects the forest task and synchronizes the legacy regression flag.

## `-isRegression`

**Type:** Boolean
**Default:** `false`

Legacy regression selector. Prefer `-forest_mode=regression` for new commands.

## `-purity_measure`

**Type:** String
**Default:** `gini`

Selects the purity or split objective. Task guides use `gini` for classification, `variance` for regression, and `isolation_path_length` for isolation.

## `-purity_threshold`

**Type:** Floating-point value
**Default:** `1e-6`

Purity stopping threshold.

## `-voting`

**Type:** String
**Default:** `mean`

Tree-prediction aggregation used by regression.

## `-regression_num_branches`

**Type:** Integer
**Default:** `2`

Number of branches used by regression splits.

## `-isolation_num_branches`

**Type:** Integer
**Default:** `2`

Number of branches used by isolation splits.

## `-isolation_min_leaf_size`

**Type:** Integer
**Default:** `1`

Minimum isolation-tree leaf size.

## Forest structure and randomness

## `-trees`

**Type:** Integer
**Default:** `11`

Number of trees.

## `-r`

**Type:** Integer
**Default:** `1`

Number of candidate distance configurations considered at each split.

## `-on_tree`

**Type:** Boolean
**Default:** `true`

Controls random distance-measure selection per node through the existing application setting.

## `-max_depth`

**Type:** Integer
**Default:** `0`

Maximum tree-depth setting.

## `-bootstrap_trees`

**Type:** Boolean
**Default:** `true`

Enables bootstrap sampling for individual trees.

## `-shuffle`

**Type:** Boolean
**Default:** `false`

Shuffles the dataset for the invocation.

## `-seed`

**Type:** Long integer
**Default:** No explicit seed

Initializes the application random-number generator.

## Parallel execution

## `-num_workers`

**Type:** Integer
**Default:** `1`

Accepted values:

- `1` for sequential execution;
- a positive integer greater than one for bounded parallel execution; or
- `-1` for every processor visible to the JVM.

Zero and values below `-1` are invalid.

## `-use_vector_api`

**Type:** Boolean
**Default:** `false`

Enables supported Vector API execution. When true, launch Java with the incubator vector module:

```bash
java --add-modules=jdk.incubator.vector -Xmx4g -jar Application/PFGAP.jar ...
```

See [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md).

## Reader selection

## `-reader_type`

**Type:** `ReaderType` name
**Default:** `DELIMITED`

Common reader type for training and test data. Reader names are case-insensitive because input is normalized to uppercase.

## `-train_reader_type`

**Type:** `ReaderType` name or blank
**Default:** No override

Overrides `-reader_type` for training data.

## `-test_reader_type`

**Type:** `ReaderType` name or blank
**Default:** No override

Overrides `-reader_type` for test data.

## `-file_pattern`

**Type:** String or `None`
**Default:** `None`

Common file-discovery pattern for compatible multi-file readers.

## `-train_file_pattern`

**Type:** String or `None`
**Default:** `None`

Training-specific file pattern.

## `-test_file_pattern`

**Type:** String or `None`
**Default:** `None`

Test-specific file pattern.

## `-id_column`

**Type:** Column name or `None`
**Default:** `None`

Instance identifier column for compatible long-form readers.

## `-time_column`

**Type:** Column name or `None`
**Default:** `None`

Time or order column for compatible long-form readers.

## `-feature_columns`

**Type:** Comma-separated string list
**Default:** Empty list

Feature columns for compatible readers.

## `-label_columns`

**Type:** Comma-separated string list
**Default:** Empty list

Label or target columns for compatible readers.

## `-hdf5_dataset_path`

**Type:** HDF5 dataset path
**Default:** `/X`

Feature dataset location for HDF5 readers.

## `-hdf5_label_dataset_path`

**Type:** HDF5 dataset path
**Default:** `/y`

Label dataset location for HDF5 readers.

Reader names, layouts, eager or lazy behavior, and required companion options are documented in [Readers](../data/Readers.md).

## Custom readers

## `-custom_reader_descriptor`

**Type:** Descriptor string or `None`
**Default:** `None`

Identifies a custom reader implementation.

## `-custom_reader_parameters`

**Type:** Semicolon-separated assignments
**Default:** Empty map

Format:

```text
-custom_reader_parameters=name1=value1;name2=value2
```

Blank names and duplicate names are rejected. Each entry must contain an equals sign.

## `-custom_reader_thread_safe`

**Type:** Boolean
**Default:** `false`

Declares whether the custom reader supports concurrent calls.

See [Custom Readers](../extensions/Custom_Readers.md).

## Observation representation

## `-is2D`

**Type:** Boolean
**Default:** `false`

`false` selects one-dimensional observations. `true` selects two-dimensional observations.

## `-isNumeric`

**Type:** Boolean
**Default:** `true`

Selects numeric or supported generic observations.

## `-numeric_storage`

**Type:** Enumerated string
**Default:** `auto`

Canonical values:

```text
auto
float32
float64
```

Accepted aliases:

```text
float
single
fp32
double
fp64
```

`auto` preserves a supported source dtype when available. Untyped text input defaults to `float64` unless its reader documents otherwise.

## `-entry_separator`

**Type:** String
**Default:** Tab

Entry delimiter for compatible text readers. To pass an escaped tab through a shell, use the form required by that shell and reader.

## `-array_separator`

**Type:** String
**Default:** `:`

Nested-array delimiter for compatible two-dimensional delimited layouts.

## `-csv_has_header`

**Type:** Boolean
**Default:** `false`

Indicates that compatible delimited input contains a header row.

## `-target_column`

**Type:** Enumerated string
**Default:** `first`
**Values:** `first`, `last`

Position of an embedded target in supported delimited input.

## Dimension selection

## `-subsample_dimensions`

**Type:** Boolean
**Default:** `false`

Enables node-level dimension subsampling.

## `-dimension_selection_strategy`

**Type:** Enumerated string
**Default:** `ALL`

Values:

```text
ALL
SQRT
LOG2
FIXED_COUNT
PROPORTION
```

Input is normalized to uppercase.

## `-dimension_selection_count`

**Type:** Integer
**Default:** `1`

Positive count required by `FIXED_COUNT` when dimension subsampling is active.

## `-dimension_selection_proportion`

**Type:** Floating-point value
**Default:** `1.0`

Finite proportion in `(0, 1]` required by `PROPORTION` when dimension subsampling is active.

Strategy-specific numeric settings are ignored when subsampling is disabled or the strategy is `ALL`.

## Distances

## `-distances`

**Type:** Bracketed distance list
**Default:** No user distance list

```text
-distances=[dtw,erp,lcss]
```

An empty list leaves the registered application defaults available:

```text
-distances=[]
```

The application default set is:

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

The main distance parser accepts built-in names and supported custom descriptors. See [Distances](Distances.md).

## Custom-distance descriptors

Supported descriptor patterns inside `-distances` include:

```text
javadistance:path/to/file[:ClassName]
python:path/to/file[:FunctionName]
maple:path/to/file[:FunctionName]
meta_type:path/to/file[:method]
```

The referenced file must exist when the arguments are parsed. See [Custom Distances](../extensions/Custom_Distances.md).

## `-knn_distances`

**Type:** Bracketed distance list
**Default:** No KNN distance list

Distances used by the KNN initial imputer:

```text
-knn_distances=[dtw,erp]
```

KNN initialization requires at least one entry.

## `-missing_proximity_distances`

**Type:** Bracketed distance list
**Default:** No missing-aware list

Allowed values:

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

Used by proximity-first imputation initialization.

## Proximities

## `-getprox`

**Type:** Boolean
**Default:** `false`

Requests supported proximity output.

## `-proximity_type`

**Type:** Enumerated string
**Default:** `PFGAP`

Values:

```text
PFGAP
BREIMAN
DEPTH_WEIGHTED
```

Input is normalized to uppercase.

## Missing values

## `-hasMissingValues`

**Type:** Boolean
**Default:** `false`

Declares that the workflow contains or handles missing feature values.

## `-MissingStrings`

**Type:** Bracketed string list
**Default:** Set by the caller

Example:

```text
-MissingStrings=[,NA,NaN,null,nan,NAN]
```

The leading empty entry represents an empty field.

See [Missing Values](Missing_Values.md).

## Imputation

## `-perform_train_imputation`

**Type:** Boolean
**Default:** `false`

Performs training-data imputation.

## `-perform_test_imputation`

**Type:** Boolean
**Default:** `false`

Performs test or evaluation-data imputation.

## `-numImputes`

**Type:** Integer
**Default:** `0`

Number of iterative proximity-imputation updates.

## `-imputation_initialization`

**Type:** Enumerated string
**Default:** `impute_first`

Values:

```text
impute_first
proximity_first
```

`proximity_first` requires compatible `-missing_proximity_distances`.

## `-initial_imputer`

**Type:** Enumerated string
**Default:** Mean imputer

Values:

```text
mean
global_mean
linear
median
global_median
mode
global_mode
knn
```

`knn` requires `-knn_distances`. The KNN initializer uses five neighbors.

## `-gap_update`

**Type:** Enumerated string
**Default:** Derived when omitted

Values:

```text
standard
dtw_alignment
```

This option synchronizes the older `DTWImpute` flag.

## `-DTWImpute`

**Type:** Boolean
**Default:** `false`

Compatibility flag for DTW-aligned updates. Prefer `-gap_update=dtw_alignment` in new commands.

## `-impute_train`

**Type:** Boolean
**Default:** `false`

Requests complete imputed training-data output.

## `-impute_test`

**Type:** Boolean
**Default:** `false`

Requests complete imputed test-data output.

## `-output_train_imputed_csr`

**Type:** Boolean
**Default:** `false`

Requests imputed-only training values in Matrix Market `.mtx` form. This also enables training imputation and missing-value handling.

## `-output_test_imputed_csr`

**Type:** Boolean
**Default:** `false`

Requests imputed-only test values in Matrix Market `.mtx` form. This also enables test imputation and missing-value handling.

## `-train_imputed_csr_file`

**Type:** Path or `None`
**Default:** `training_imputed_values.mtx`

Output path for imputed-only training values.

## `-test_imputed_csr_file`

**Type:** Path or `None`
**Default:** `testing_imputed_values.mtx`

Output path for imputed-only test values.

See [Imputation](../guides/Imputation.md) and [Imputed-Only Output](Imputed_Only_Output.md).

## Standardization

## `-standardization`

**Type:** Standardization method
**Default:** `none`

Implemented methods:

```text
none
z_score
mean_center
min_max
```

Accepted aliases are documented in [Standardization](../guides/Standardization.md).

## `-standardization_scope`

**Type:** Standardization scope
**Default:** `per_dimension`

Values:

```text
global
per_dimension
per_series
per_series_per_dimension
```

## `-standardization_variance`

**Type:** Variance convention
**Default:** `population`

Values:

```text
population
sample
```

## `-standardization_stats`

**Type:** JSON path or `None`
**Default:** `None`

Loads reusable statistics for `global` or `per_dimension` standardization.

## `-save_standardization_stats`

**Type:** Boolean
**Default:** `false`

Saves newly fitted reusable statistics.

## `-standardization_stats_output`

**Type:** JSON path or `None`
**Default:** `None`

Optional path for saved fitted statistics.

The complete standardization constraints are documented in [Standardization](../guides/Standardization.md).

## Prediction, scoring, and enhanced output

## `-get_predictions`

**Type:** Boolean
**Default:** `false`

Requests ordinary aggregate prediction artifacts.

## `-return_enhanced_outputs`

**Type:** Boolean
**Default:** `false`

Requests structured per-instance prediction details.

## `-get_training_outlier_scores`

**Type:** Boolean
**Default:** `false`

Requests Breiman-style supervised training outlier scores from a classification run. This does not select isolation mode.

## `-return_ood_scores`

**Type:** Boolean
**Default:** `false`

Requests evaluation-time OOD output. OOD output is independent of ordinary predictions and enhanced prediction details.

## `-ood_score_type`

**Type:** OOD scorer name
**Default:** `relative_support_exceedance`

Current value:

```text
relative_support_exceedance
```

## `-collect_split_distance_summaries`

**Type:** Boolean
**Default:** `false`

Retains branch-local distance summaries during training so the saved model can support distance-based OOD scoring.

See [Outlier Scoring](../guides/Outlier_Scoring.md), [OOD Scoring](../guides/OOD_Scoring.md), and [Outputs](Outputs.md).

## Option relationships

## Training and evaluation

Use:

```text
-eval=false
```

for training, and:

```text
-eval=true
```

for saved-model evaluation.

During current saved-model evaluation, supply the evaluation input through both `-train` and `-test`.

## Forest mode and regression flag

`-forest_mode=regression` sets the regression flag true. Classification and isolation set it false. Prefer `-forest_mode` over setting only `-isRegression`.

## Test labels

A non-`None` `-test_labels` path marks test labels as present. A later `-exists_testlabels=false` argument does not clear that state.

## Imputed-only output

`-output_train_imputed_csr=true` enables training imputation and missing-value handling. `-output_test_imputed_csr=true` enables test imputation and missing-value handling.

## Gap update compatibility

`-gap_update=dtw_alignment` sets `DTWImpute` true. `-gap_update=standard` sets it false. If only `-DTWImpute` is supplied, true selects DTW alignment and false selects standard when no strategy has already been set.

## OOD capability

A model intended for later OOD scoring must be trained with:

```text
-collect_split_distance_summaries=true
```

Requesting OOD output does not require `-return_enhanced_outputs=true`.

## Reader precedence

Training reader selection uses `-train_reader_type` when present, otherwise `-reader_type`. Test reader selection uses `-test_reader_type` when present, otherwise `-reader_type`.

Training and test file patterns follow the same override rule.

## Rejected and unavailable options

## `-jvmwarmup`

The former parser case is disabled. It is not a current CLI option.

## Former parallel booleans

These older options are not accepted:

```text
-parallelTrees
-parallelProx
-parallelPredict
-parallelSplit
-parallelSplitThreshold
```

Use `-num_workers` for the shared worker budget.

## Unknown options

Any option not matched by the parser aborts with an invalid-command-line error.

## Exit behavior and errors

The application catches configuration and runtime exceptions at its main entry point and reports them through the application abort utility.

Common command-line errors include:

- an argument without `=`;
- an unknown option name;
- an unknown reader type;
- an unknown numeric storage type;
- an invalid forest mode;
- an invalid target position;
- an unknown distance;
- an incompatible missing-aware distance;
- a malformed distance list;
- a missing custom-distance file;
- malformed custom-reader parameters;
- invalid dimension-selection settings;
- KNN imputation without KNN distances;
- incompatible standardization settings; and
- an invalid OOD score type.

## Related documentation

- [Configuration Reference](Configuration_Reference.md)
- [Configuration](../getting-started/Configuration.md)
- [Readers](../data/Readers.md)
- [Distances](Distances.md)
- [Missing Values](Missing_Values.md)
- [Outputs](Outputs.md)
- [Model Persistence](Model_Persistence.md)
