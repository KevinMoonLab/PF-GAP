# Configuration Reference

This page defines the public configuration properties exposed by PFGAP's Python helper and Java application. It records current names, types, defaults, accepted values, option relationships, and task applicability.

For task-oriented examples, see [Configuration](../getting-started/Configuration.md) and the pages in `guides/`. For exact direct-command syntax, see [CLI Reference](CLI_Reference.md).

## Configuration interfaces

PFGAP has two public configuration interfaces:

- `PF_wrapper.train(...)` for training and optional same-run evaluation;
- `PF_wrapper.predict(...)` for evaluation with a saved model; and
- direct Java arguments passed to `PFGAP.jar` as `-name=value`.

The Python helper constructs and launches a Java command. Python argument names sometimes differ from their Java equivalents, and the helper supplies many values explicitly even when the caller accepts the Python default.

## General conventions

### Direct Java syntax

Every Java application argument uses:

```text
-name=value
```

Option names are case-sensitive. Boolean values use `true` or `false`.

### Python status value

`PF_wrapper.train(...)` and `PF_wrapper.predict(...)` return the Java process exit status. A successful process returns `0`.

### Paths

Relative paths are resolved from the process working directory. The current Python helper launches `PFGAP.jar` by filename, so the standard arrangement is to run Python from `Application/`, with `PF_wrapper.py` and `PFGAP.jar` together.

### Lists

Python sequences are encoded as bracketed comma-separated values:

```python
distances=["dtw", "erp"]
```

```text
-distances=[dtw,erp]
```

### Optional Python values

Python `None` is used for optional values. The helper forwards `None` as text for some path arguments and omits other arguments entirely. The tables below identify the public default and behavior.

## Task and execution mode

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `forest_mode` | `-forest_mode` | `None` | `None` | `classification`, `regression`, `isolation` | Selects the forest task. In training, `None` becomes `regression` when `regressor=True`; otherwise it becomes `classification`. During prediction, the saved-model task should be used. |
| `regressor` | `-isRegression` | `False` | Not exposed | Boolean | Legacy regression selector. Explicit `forest_mode` is preferred. Training helper synchronizes it with `forest_mode`. |
| Not exposed separately | `-eval` | Helper sends `false` | Helper sends `true` | Boolean | Selects training or saved-model evaluation. Set by the helper function rather than a public Python keyword. |
| `repeats` | `-repeats` | `1` | Not exposed | Positive integer | Number of training repetitions. Direct Java evaluation also accepts `-repeats`; the current Python `predict(...)` interface does not expose it. |

## Input paths and labels

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `train_file` | `-train` | Required | Not used by name | Path | Training observations. |
| `test_file` | `-test` | `None` | Not used by name | Path or `None` | Optional validation or test observations supplied during training. |
| `testfile` | `-train` and `-test` | Not applicable | Required | Path | Evaluation observations. The helper sends the same path through both Java options. |
| `train_labels` | `-train_labels` | `None` | Not exposed | Path or `None` | Separate training labels or targets. |
| `test_labels` | `-test_labels` | `None` | `None` | Path or `None` | Separate test, validation, or evaluation labels or targets. Supplying a Java `-test_labels` path also marks test labels as present. |
| `exists_testlabels` | `-exists_testlabels` | `False` | `False` | Boolean | Indicates whether known evaluation labels or targets are available. |
| `target_column` | `-target_column` | `first` | `first` | `first`, `last` | Position of an embedded target in supported delimited layouts. |
| `file_has_header` | `-csv_has_header` | `False` | `False` | Boolean | Indicates whether supported delimited input has a header row. |

Classification currently uses one integer class label per labeled observation. Regression uses one numeric target per labeled observation.

## Reader selection

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `reader_type` | `-reader_type` | `None` | `None` | `ReaderType` name or `None` | Selects the common reader type. When omitted by the helper, the Java default is `DELIMITED`. |
| Not exposed | `-train_reader_type` | Java default `None` | Not applicable | `ReaderType` name | Overrides the common reader type for training data. |
| Not exposed | `-test_reader_type` | Java default `None` | Not applicable | `ReaderType` name | Overrides the common reader type for test data. |
| `file_pattern` | `-file_pattern` | `None` | `None` | String or `None` | Common discovery pattern for readers that consume multiple files. |
| Not exposed | `-train_file_pattern` | Java default `None` | Not applicable | String or `None` | Training-specific file pattern. |
| Not exposed | `-test_file_pattern` | Java default `None` | Not applicable | String or `None` | Test-specific file pattern. |
| `id_column` | `-id_column` | `None` | `None` | Column name or `None` | Instance identifier column for compatible long-form readers. |
| `time_column` | `-time_column` | `None` | `None` | Column name or `None` | Time or order column for compatible long-form readers. |
| `feature_columns` | `-feature_columns` | `None` | `None` | Sequence of column names | Explicit feature columns for compatible readers. |
| `label_columns` | `-label_columns` | `None` | `None` | Sequence of column names | Explicit label or target columns for compatible readers. |
| `hdf5_dataset_path` | `-hdf5_dataset_path` | `/X` | `/X` | HDF5 dataset path | Feature dataset location for HDF5 readers. |
| `hdf5_label_dataset_path` | `-hdf5_label_dataset_path` | `/y` | `/y` | HDF5 dataset path | Label dataset location for HDF5 readers. |

Reader type names and their requirements are documented in [Readers](../data/Readers.md).

### Eager and lazy selection

The current public wrapper and Java parser do not expose a standalone `lazy` option. Eager or lazy access is selected through a reader type whose implementation has the corresponding access behavior. See [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md) and [Readers](../data/Readers.md).

## Custom readers

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `custom_reader_descriptor` | `-custom_reader_descriptor` | `None` | `None` | Descriptor string or `None` | Identifies the custom reader implementation. |
| `custom_reader_parameters` | `-custom_reader_parameters` | `None` | `None` | Dictionary, encoded string, or `None` | Reader-specific parameters. Dictionaries are encoded as `name=value` entries separated by semicolons. |
| `custom_reader_thread_safe` | `-custom_reader_thread_safe` | `False` | `False` | Boolean | Declares whether the custom reader may be called concurrently. |

Custom parameter names cannot be blank or contain `;` or `=`. Values cannot contain `;`. Duplicate Java-side names are rejected. See [Custom Readers](../extensions/Custom_Readers.md).

## Observation representation

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `data_dimension` | `-is2D` | `1` | `1` | Python: `1` or `2`; Java: Boolean | `1` maps to `false` and selects one-dimensional observations. `2` maps to `true` and selects two-dimensional observations. |
| `numeric_data` | `-isNumeric` | `True` | `True` | Boolean | Selects numeric or supported generic observations. |
| `numeric_storage` | `-numeric_storage` | `auto` | Defined but not forwarded by current `predict(...)` | `auto`, `float32`, `float64` and aliases | Selects primitive numeric feature storage. Training helper forwards this option. |
| `entry_separator` | `-entry_separator` | `,` | `,` | String | Separates entries in supported delimited input. Python converts a literal tab to `\t` for Java. |
| `array_separator` | `-array_separator` | `:` | `:` | String | Separates nested arrays in compatible delimited 2D layouts. |

`numeric_storage` aliases are:

- `float`, `single`, and `fp32` for `float32`;
- `double` and `fp64` for `float64`.

`auto` preserves a supported source dtype when the format provides one. Untyped text readers default to `float64` unless documented otherwise. Numeric feature storage does not require distances, proximities, predictions, or scores to use the same primitive precision.

## Forest structure

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `num_trees` | `-trees` | `11` | Not exposed | Positive integer | Number of trees in the forest. |
| `r` | `-r` | `5` | Not exposed | Positive integer | Candidate distance configurations considered at each split. Java's context default is `1`; the Python helper explicitly sends `5`. |
| `on_tree` | `-on_tree` | `True` | Not exposed | Boolean | Controls random distance-measure selection per node through the existing application setting. |
| `max_depth` | `-max_depth` | `0` | Not exposed | Integer | Maximum tree depth setting. The Java field initializes to `0`. |
| `bootstrap_trees` | `-bootstrap_trees` | `True` | Not exposed | Boolean | Enables bootstrap sampling for trees. |
| `shuffle` | `-shuffle` | `False` | `False` | Boolean | Shuffles the dataset for the invocation. |

## Dimension selection

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `subsample_dimensions` | `-subsample_dimensions` | `False` | Not exposed | Boolean | Enables node-level dimension subsampling. |
| `dimension_selection_strategy` | `-dimension_selection_strategy` | `all` | Not exposed | `ALL`, `SQRT`, `LOG2`, `FIXED_COUNT`, `PROPORTION` | Strategy used when subsampling dimensions. |
| `dimension_selection_count` | `-dimension_selection_count` | `1` | Not exposed | Positive integer | Dimension count for `FIXED_COUNT`. |
| `dimension_selection_proportion` | `-dimension_selection_proportion` | `1.0` | Not exposed | Finite value in `(0, 1]` | Dimension proportion for `PROPORTION`. |

Python aliases include `fixed` and `count` for `FIXED_COUNT`, `prop` for `PROPORTION`, and `log_2` for `LOG2`.

Strategy-specific numeric values are ignored when subsampling is disabled or the strategy is `ALL`.

## Classification, regression, and isolation settings

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `purity` | `-purity_measure` | `gini` | Not exposed | Purity name | Classification helper default. The helper substitutes `variance` for regression and `isolation_path_length` for isolation when the caller leaves `gini`. |
| `purity_threshold` | `-purity_threshold` | `1e-6` | Not exposed | Floating-point value | Purity stopping threshold. |
| `regressor_aggregation` | `-voting` | `mean` | Not exposed | Aggregation name | Combines tree-level regression predictions. |
| `regression_num_branches` | `-regression_num_branches` | `2` | Not exposed | Positive integer | Number of branches for regression splits. |
| `isolation_num_branches` | `-isolation_num_branches` | `2` | Not exposed | Positive integer | Number of branches for isolation splits. |
| `isolation_min_leaf_size` | `-isolation_min_leaf_size` | `1` | Not exposed | Positive integer | Minimum isolation leaf size. |

## Distances

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `distances` | `-distances` | `None` | `None` | Sequence, encoded list, or descriptors | Candidate forest distances. The helper encodes `None` as an empty list, which delegates to application behavior. |
| `knn_distances` | `-knn_distances` | `None` | `None` | Distance sequence | Required when `initial_imputer="knn"`. |
| `missing_proximity_distances` | `-missing_proximity_distances` | `None` | `None` | Missing-compatible distance sequence | Distances used for `proximity_first` initialization. |

Valid missing-proximity distances are:

```text
nan_euclidean
nan_euclidean_i
dtwarow
dtwarow_i
dtwarow_d
```

The Java application default distance set, used when no user distances are active, is:

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

See [Distances](Distances.md) for representation compatibility and [Custom Distances](../extensions/Custom_Distances.md) for descriptor syntax.

## Proximities

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `return_proximities` | `-getprox` | `False` | `False` | Boolean | Requests supported proximity output. |
| `proximity_type` | `-proximity_type` | `PFGAP` | `PFGAP` | `PFGAP`, `BREIMAN`, `DEPTH_WEIGHTED` | Selects the proximity definition. Python normalizes the value to uppercase. |

## Missing-value recognition

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `has_missing_values` | `-hasMissingValues` | `None` | `None` | Boolean or `None` | Explicit missing-data indicator. When `None`, the helper derives it from imputation, sparse imputed output, proximity-first initialization, and missing-aware distance settings. |
| `missing_indicators` | `-MissingStrings` | `("", "NA", "NaN", "null", "nan", "NAN")` | Same | Sequence of strings | Tokens interpreted as missing by compatible readers. |

See [Missing Values](Missing_Values.md).

## Imputation operations

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `impute_training_data` | `-perform_train_imputation` | `False` | Not exposed | Boolean | Performs training-data imputation. |
| `impute_testing_data` | `-perform_test_imputation` | `False` | `False` | Boolean | Performs test or evaluation-data imputation. |
| `impute_iterations` | `-numImputes` | `5` | `5` | Nonnegative integer | Number of proximity-imputation iterations. Java's context default is `0`; the helper explicitly sends `5`. |
| `imputation_initialization` | `-imputation_initialization` | `impute_first` | `impute_first` | `impute_first`, `proximity_first` | Selects initialization before iterative updates. |
| `initial_imputer` | `-initial_imputer` | `mean` | `mean` | `mean`, `global_mean`, `linear`, `median`, `global_median`, `mode`, `global_mode`, `knn` | Initial value strategy for impute-first handling and applicable workflows. |
| `gap_update` | `-gap_update` | `None` | `None` | `standard`, `dtw_alignment`, or `None` | Proximity update strategy. `None` is derived from `DTWImpute`. |
| `DTWImpute` | `-DTWImpute` | `False` | `False` | Boolean | Compatibility flag. Selects `dtw_alignment` when `gap_update` is omitted. |

KNN initialization uses five neighbors and requires at least one `knn_distances` entry.

## Imputation outputs

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `return_imputed_training` | `-impute_train` | `False` | Not exposed | Boolean | Requests a complete imputed training dataset and enables training imputation. |
| `return_imputed_testing` | `-impute_test` | `False` | `False` | Boolean | Requests a complete imputed test dataset and enables test imputation. |
| `return_imputed_training_csr` | `-output_train_imputed_csr` | `False` | Not exposed | Boolean | Requests imputed-only training output in Matrix Market `.mtx` form and enables training imputation. |
| `return_imputed_testing_csr` | `-output_test_imputed_csr` | `False` | `False` | Boolean | Requests imputed-only test output in Matrix Market `.mtx` form and enables test imputation. |
| `training_imputed_csr_file` | `-train_imputed_csr_file` | `None` | Not exposed | Path or `None` | Output path for imputed-only training values. Java default: `training_imputed_values.mtx`. |
| `testing_imputed_csr_file` | `-test_imputed_csr_file` | `None` | `None` | Path or `None` | Output path for imputed-only test values. Java default: `testing_imputed_values.mtx`. |

The option names retain `csr`; the supported file output is Matrix Market `.mtx`. See [Imputed-Only Output](Imputed_Only_Output.md).

### Lazy per-file Parquet restriction

The current Python helper requires `file_pattern` for `PER_FILE_PARQUET` and rejects training or test imputation and imputed-data output with this reader type.

## Standardization

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `standardization` | `-standardization` | `none` | `none` | `none`, `z_score`, `mean_center`, `min_max` plus aliases | Selects the numeric transformation. |
| `standardization_scope` | `-standardization_scope` | `per_dimension` | `per_dimension` | `global`, `per_dimension`, `per_series`, `per_series_per_dimension` plus aliases | Selects which values share parameters. |
| `standardization_variance` | `-standardization_variance` | `population` | `population` | `population`, `sample` plus aliases | Variance convention for z-score standardization. |
| `standardization_stats` | `-standardization_stats` | `None` | `None` | JSON path or `None` | Loads reusable `global` or `per_dimension` statistics. |
| `save_standardization_stats` | `-save_standardization_stats` | `False` | `False` | Boolean | Saves newly fitted reusable statistics. |
| `standardization_stats_output` | `-standardization_stats_output` | `None` | `None` | JSON path or `None` | Optional statistics output path. |

### Current Python `predict(...)` forwarding

The current `predict(...)` signature accepts the six standardization arguments, but its command builder does not append them to the Java invocation. Saved-model evaluation therefore uses the preprocessing state restored by the Java model-loading workflow rather than Python-side standardization overrides. Direct Java evaluation can supply these options explicitly.

See [Standardization](../guides/Standardization.md) for method formulas, aliases, scope semantics, and configuration constraints.

## Model and output paths

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `save_model` | `-savemodel` | `True` | Not exposed | Boolean | Saves the trained model. |
| `model_name` | `-modelname` | `PF` | Required | Name during training; path during prediction | The training helper reduces the value to its final path component. Prediction passes the supplied saved-model path. |
| `output_directory` | `-out` | Empty string | Empty string | Directory path | Python converts an empty value to the current directory, creates one missing directory, and appends `/`. |
| `export` | `-export` | `1` | `1` | Integer export level | Controls application artifact export level. Java context documents levels `0`, `1`, and `2`. |
| `verbosity` | `-verbosity` | `1` | `1` | Integer verbosity level | Controls runtime reporting. Java context documents levels `0`, `1`, and `2`. |

The Java context default output directory is `output/`. The helper always sends its normalized output path.

## Prediction and structured outputs

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `return_predictions` | `-get_predictions` | `False` | `False` | Boolean | Requests ordinary aggregate prediction artifacts. |
| `return_enhanced_outputs` | `-return_enhanced_outputs` | `False` | `False` | Boolean | Requests structured per-instance prediction details. |
| `return_training_outlier_scores` | `-get_training_outlier_scores` | `False` | Not exposed | Boolean | Requests Breiman-style supervised classification training outlier scores. |

Prediction, enhanced output, and OOD output are independent requests.

## OOD configuration

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `return_ood_scores` | `-return_ood_scores` | `False` | `False` | Boolean | Requests OOD output. It does not require enhanced prediction output. |
| `ood_score_type` | `-ood_score_type` | `relative_support_exceedance` | Same | `relative_support_exceedance` | Selects the evaluation-time OOD scorer. |
| `collect_split_distance_summaries` | `-collect_split_distance_summaries` | `False` | Not exposed | Boolean | Retains branch-local distance summaries during training so the model can support distance-based OOD scoring. |

When `return_ood_scores=True` is passed to `train(...)`, the helper enables split-distance summary collection for same-run validation.

See [OOD Scoring](../guides/OOD_Scoring.md).

## Parallelism and randomness

| Python argument | Java option | Train default | Predict default | Type and accepted values | Description |
|---|---|---:|---:|---|---|
| `num_workers` | `-num_workers` | `1` | `1` | `-1` or a positive integer | Worker budget. `-1` uses all processors visible to the JVM; `1` is sequential. |
| `seed` | `-seed` | `None` | Not exposed | Long integer or `None` | Seeds the application random-number generator during training. When omitted, no seed argument is sent. |
| `use_vector_api` | `-use_vector_api` | `False` | `False` | Boolean | Records and enables supported Vector API execution. The helper also adds the incubator vector module when true. |
| `memory` | JVM `-Xmx` argument | `1g` | `1g` | JVM heap-size string | Python-only helper setting used to construct `-Xmx`, for example `4g`. |

### Current Vector API prediction command

The current training helper supplies `--add-modules=jdk.incubator.vector` as a Java argument pair-compatible string entry. The current prediction helper constructs `"--add-modules jdk.incubator.vector"` as one argument. Direct Java users should use:

```bash
java --add-modules=jdk.incubator.vector -Xmx4g -jar PFGAP.jar ...
```

See [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md).

## Java-only application settings

The following settings exist in the Java application context but are not exposed by the current Python helper signature.

| Java option or context setting | Default | Description |
|---|---:|---|
| `-train_reader_type` | `None` | Training-specific reader override. |
| `-test_reader_type` | `None` | Test-specific reader override. |
| `-train_file_pattern` | `None` | Training-specific file pattern. |
| `-test_file_pattern` | `None` | Test-specific file pattern. |
| `warmup_java` context setting | `False` | Java warmup control. The parser's former `-jvmwarmup` case is disabled. |
| `garbage_collect_after_each_repetition` context setting | `True` | Requests repetition-level garbage collection in applicable orchestration. |
| `print_test_progress_for_each_instances` context setting | `100` | Evaluation progress interval. |
| `useSparseProximities` context setting | `True` | Selects sparse internal proximity storage. |

These context-only values are not guaranteed as command-line options unless listed in [CLI Reference](CLI_Reference.md).

## Derived Python-helper behavior

The helper applies these configuration rules before launching Java:

1. `forest_mode=None` becomes `regression` when `regressor=True`; otherwise `classification`.
2. Regression and isolation modes synchronize the legacy regression flag.
3. Default `gini` purity becomes `variance` for regression and `isolation_path_length` for isolation.
4. `return_ood_scores=True` during training enables split-distance summary collection.
5. Complete or imputed-only output requests enable the corresponding imputation operation.
6. `has_missing_values=None` is derived from imputation and missing-aware settings.
7. `gap_update=None` becomes `dtw_alignment` when `DTWImpute=True`; otherwise `standard`.
8. `data_dimension=1` maps to `-is2D=false`; `2` maps to `-is2D=true`.
9. The training model name is reduced to its final path component.
10. The output directory is created when one directory level is missing and normalized with a trailing slash.

## Training-only Python arguments

The following public helper arguments are currently available only on `train(...)`:

```text
train_file
test_file
train_labels
collect_split_distance_summaries
save_model
repeats
num_trees
r
isolation_num_branches
isolation_min_leaf_size
regression_num_branches
bootstrap_trees
seed
on_tree
max_depth
subsample_dimensions
dimension_selection_strategy
dimension_selection_count
dimension_selection_proportion
return_training_outlier_scores
regressor
purity
purity_threshold
regressor_aggregation
impute_training_data
return_imputed_training
return_imputed_training_csr
training_imputed_csr_file
numeric_storage
```

Some corresponding Java options may remain usable during direct evaluation, but they are not exposed by the current Python prediction signature.

## Prediction-only Python arguments

The public prediction interface uses:

```text
model_name
testfile
```

as required arguments. It does not expose a separate training input because the evaluation file is sent as both Java `-train` and `-test`.

## Configuration errors

Common causes of rejection include:

- an unknown or incorrectly capitalized Java option;
- a direct argument not written as `-name=value`;
- `data_dimension` other than `1` or `2`;
- an invalid forest mode;
- `num_workers` equal to `0` or below `-1`;
- an unknown reader type;
- an unknown numeric storage type;
- a malformed list;
- an unknown distance;
- a non-missing-compatible entry in `missing_proximity_distances`;
- KNN initialization without KNN distances;
- invalid dimension-selection count or proportion;
- incompatible standardization statistics;
- a custom reader parameter with invalid delimiters;
- `PER_FILE_PARQUET` without a file pattern in the Python helper; or
- imputation requested for Python-helper `PER_FILE_PARQUET` access.

## Related documentation

- [Configuration](../getting-started/Configuration.md)
- [CLI Reference](CLI_Reference.md)
- [Readers](../data/Readers.md)
- [Distances](Distances.md)
- [Missing Values](Missing_Values.md)
- [Outputs](Outputs.md)
- [Model Persistence](Model_Persistence.md)
- [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md)
