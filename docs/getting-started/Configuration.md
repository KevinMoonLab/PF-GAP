# Configuration

This guide explains how PFGAP is configured through the Python helper and the direct Java command line. It focuses on configuration workflow, naming, value encoding, precedence, validation, and reusable configuration patterns; the complete option catalogs belong in the [Configuration Reference](../reference/Configuration_Reference.md) and [CLI Reference](../reference/CLI_Reference.md).

> **Version status:** PFGAP is under active development toward version 1.0. Keep `Application/PFGAP.jar`, `Application/PF_wrapper.py`, and this documentation on compatible revisions. Pre-v1 option names, defaults, and compatibility rules may change.

## Configuration interfaces

PFGAP currently exposes two primary configuration interfaces:

1. **Python helper:** Call `PF_wrapper.train(...)` or `PF_wrapper.predict(...)` with Python keyword arguments.
2. **Direct Java command line:** Launch `PFGAP.jar` with arguments in the exact form `-name=value`.

The Python helper validates and normalizes selected values, translates Python-facing names to Java option names, and launches the same Java application. It does not implement a separate training, prediction, imputation, or scoring engine.

PFGAP does not currently use a general YAML, JSON, TOML, or properties file as its primary public configuration interface. For reproducible experiments, store Python calls in scripts or store direct Java commands in shell scripts.

## Choose an interface

Use the **Python helper** when:

- the surrounding workflow is written in Python;
- experiments are launched from notebooks or Python scripts;
- Python data preparation or result analysis is required;
- Python keyword arguments are more readable than a long shell command; or
- helper-side validation and normalization are useful.

Use the **direct Java command line** when:

- no Python environment is desired;
- PFGAP is launched from a shell script, scheduler, container, or Java-oriented pipeline;
- the exact Java option names need to be visible; or
- an option is accepted by the Java application but is not yet exposed by the Python helper.

The interfaces should produce equivalent application behavior when they ultimately supply the same effective configuration, input files, seed, worker count, application revision, and extension implementations. They are not textually identical: the helper uses Python-facing names and supplies many defaults explicitly.

## Python configuration

The helper exposes separate functions for training and evaluation:

```python
import PF_wrapper as PF

training_status = PF.train(
    train_file="data/train.csv",
    test_file="data/test.csv",
    exists_testlabels=True,
    forest_mode="classification",
    num_trees=101,
    r=5,
    seed=42,
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    target_column="first",
    return_predictions=True,
    save_model=True,
    model_name="classification_model",
    output_directory="output/train",
)

if training_status != 0:
    raise SystemExit(training_status)
```

Evaluation of a saved model uses `predict(...)`:

```python
import PF_wrapper as PF

prediction_status = PF.predict(
    model_name="output/train/classification_model",
    testfile="data/new_data.csv",
    exists_testlabels=False,
    forest_mode="classification",
    num_workers=4,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    target_column="first",
    return_predictions=True,
    output_directory="output/predict",
)

if prediction_status != 0:
    raise SystemExit(prediction_status)
```

The exact saved-model path is determined by the training output contract. Use the path produced by the training run rather than assuming that the model name alone is sufficient. See [Model Persistence](../reference/Model_Persistence.md).

### Python argument names

Python names are designed to be readable in Python and do not always match Java option names. Common mappings include:

| Python argument | Java option |
|---|---|
| `num_trees` | `-trees` |
| `return_proximities` | `-getprox` |
| `return_predictions` | `-get_predictions` |
| `save_model` | `-savemodel` |
| `model_name` | `-modelname` |
| `output_directory` | `-out` |
| `file_has_header` | `-csv_has_header` |
| `data_dimension=1` | `-is2D=false` |
| `data_dimension=2` | `-is2D=true` |
| `numeric_data` | `-isNumeric` |
| `return_training_outlier_scores` | `-get_training_outlier_scores` |
| `impute_training_data` | `-perform_train_imputation` |
| `impute_testing_data` | `-perform_test_imputation` |
| `impute_iterations` | `-numImputes` |
| `return_imputed_training` | `-impute_train` |
| `return_imputed_testing` | `-impute_test` |
| `return_imputed_training_csr` | `-output_train_imputed_csr` |
| `return_imputed_testing_csr` | `-output_test_imputed_csr` |
| `missing_indicators` | `-MissingStrings` |
| `regressor_aggregation` | `-voting` |

This table illustrates the translation layer; it is not the complete option catalog.

### Helper return value

`train(...)` and `predict(...)` return the status code from the launched Java process. A successful process conventionally returns `0`. Scripts should check this value instead of assuming that the run completed successfully.

### Helper working directory

The current helper launches `PFGAP.jar` by file name. The simplest supported arrangement is to keep `PF_wrapper.py` and `PFGAP.jar` together and run the Python process with `Application/` as its working directory.

Paths supplied to the helper are passed to the launched Java process. Relative dataset, model, extension, statistics, and output paths therefore need to be valid from that working directory.

## Direct Java configuration

A direct command begins with:

```bash
java -jar Application/PFGAP.jar -name=value
```

For example:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -test=data/test.csv \
  -exists_testlabels=true \
  -forest_mode=classification \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -get_predictions=true \
  -savemodel=true \
  -modelname=classification_model \
  -out=output/train/
```

Every application argument must use the exact form:

```text
-name=value
```

These forms are invalid:

```text
-name value
-name = value
name=value
```

The Java parser splits each argument at the first equals sign. This allows the value portion to contain additional equals signs, which is important for encoded custom-reader parameters.

### Option names are exact

Direct Java option names are case-sensitive. For example, the current interface uses both lower-case names and historically mixed-case names such as:

```text
-is2D=true
-isNumeric=true
-hasMissingValues=true
-MissingStrings=[,NA,NaN,null,nan,NAN]
-DTWImpute=false
```

Copy names from the current reference documentation or application interface rather than changing capitalization for consistency.

### Boolean values

Use `true` or `false`:

```text
-savemodel=true
-shuffle=false
```

The Java boolean parser is case-insensitive, but lower-case values are used throughout the documentation for consistency.

### Numeric values

Integer and floating-point options are written without additional quoting:

```text
-trees=101
-purity_threshold=1e-6
-dimension_selection_proportion=0.5
```

Invalid numeric text causes parsing to fail. Strategy-specific ranges may be validated after all arguments have been read.

### Null-like values

Some path-like or optional values treat `None` as absent:

```text
-test=None
-train_labels=None
-test_labels=None
```

This behavior is option-specific. Do not assume that every option accepts `None`. In Python, prefer the actual value `None` for optional helper arguments unless a documented string encoding is required.

## Configuration layers and precedence

Understanding which layer supplies a value prevents accidental dependence on defaults.

### Direct Java

For direct Java runs, the effective configuration is constructed as follows:

1. PFGAP starts with values initialized by the application context.
2. Each supplied command-line argument overrides or supplements the corresponding value.
3. Cross-option normalization and validation are applied after parsing where required.
4. The selected workflow may derive additional behavior from the final settings.

If the same direct option is supplied more than once, arguments are processed from left to right, so the later occurrence normally replaces the earlier value. Do not rely on duplicates as a configuration technique; specify each option once.

### Python helper

For helper-driven runs:

1. Python function defaults are applied to arguments the caller omits.
2. The helper validates and normalizes selected combinations.
3. The helper converts values to Java command-line arguments.
4. The Java application parses those arguments and performs its own validation and setup.

This means a Python helper default can override an internal Java default because the helper often passes the Python value explicitly. When comparing Python and direct Java runs, compare the final options sent to Java rather than comparing only the arguments written by the user.

### Loaded models

A saved model may contain fitted structures and task state that must remain compatible with evaluation-time data and options. Runtime settings such as input paths, requested outputs, reader configuration, worker count, and evaluation-time imputation still need to be supplied where applicable. Do not deliberately contradict the model's training contract. See [Model Persistence](../reference/Model_Persistence.md) for the authoritative compatibility rules.

## Group configuration by responsibility

Long calls are easier to review when options are grouped conceptually.

### Inputs and targets

```python
train_file="data/train.csv"
test_file="data/test.csv"
train_labels=None
test_labels=None
exists_testlabels=True
target_column="first"
```

Targets can be embedded in supported input files or supplied separately where the selected reader and workflow permit it. `exists_testlabels` describes whether evaluation targets are available; it does not identify the target location by itself.

### Task and forest

```python
forest_mode="classification"
repeats=1
num_trees=101
r=5
bootstrap_trees=True
max_depth=0
```

`forest_mode` should be set explicitly to `classification`, `regression`, or `isolation` in reusable scripts. The older `regressor`/`-isRegression` setting remains part of the current interface, but explicit forest mode communicates intent more clearly.

Task-specific guides explain valid and useful combinations:

- [Classification](../guides/Classification.md)
- [Regression](../guides/Regression.md)
- [Outlier Scoring](../guides/Outlier_Scoring.md)
- [OOD Scoring](../guides/OOD_Scoring.md)

### Data representation and reading

```python
data_dimension=1
numeric_data=True
numeric_storage="auto"
reader_type=None
entry_separator=","
array_separator=":"
file_has_header=False
target_column="first"
```

`data_dimension` describes the shape of one observation, not the rank of the whole dataset:

- `1` selects a one-dimensional observation such as a tabular feature vector or univariate sequence.
- `2` selects a two-dimensional observation such as a multivariate series.

`numeric_storage` accepts canonical values `auto`, `float32`, and `float64`; the Java interface also recognizes documented aliases. Reader-specific options such as `file_pattern`, `id_column`, `time_column`, `feature_columns`, HDF5 paths, and custom-reader descriptors should be supplied only when required by the selected reader.

See [Dataset Representations](../data/Dataset_Representations.md), [Data Formats](../data/Data_Formats.md), and [Readers](../data/Readers.md).

### Distances

```python
distances=["dtw", "erp", "lcss"]
```

Distance lists select the candidate measures available to the forest. Names must be registered or use a supported custom-distance descriptor. An omitted or empty distance list delegates selection to the application's current default behavior.

Direct Java lists use brackets and comma-separated entries:

```text
-distances=[dtw,erp,lcss]
```

Do not add shell-interpreted whitespace or quoting unless required by the shell. See [Distances](../reference/Distances.md) and [Custom Distances](../extensions/Custom_Distances.md).

### Outputs and persistence

```python
return_predictions=True
return_enhanced_outputs=False
return_proximities=False
save_model=True
model_name="experiment_01"
output_directory="output/experiment_01"
export=1
verbosity=1
```

Request only the artifacts needed by the workflow. Some outputs require compatible task state or additional collection during training. `export` controls the application's export level, while individual booleans request specific artifacts. `verbosity` controls runtime reporting rather than which scientific results are computed.

See [Outputs](../reference/Outputs.md) and [Model Persistence](../reference/Model_Persistence.md).

### Execution resources

```python
memory="8g"
num_workers=4
use_vector_api=False
seed=42
```

`memory` is a Python-helper option used to construct the JVM heap argument, such as `-Xmx8g`; it is not a PFGAP application option. Direct Java users place the heap argument before `-jar`:

```bash
java -Xmx8g -jar Application/PFGAP.jar ...
```

`num_workers` controls the shared worker configuration. A fixed seed is recommended for repeatable experiments, but exact reproducibility may also depend on worker count, data order, implementation revision, and custom code. See [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md).

## Value encoding

Several option families use compact string encodings.

### Lists

Python callers should normally pass a Python sequence:

```python
distances=["dtw", "erp"]
feature_columns=["temperature", "pressure"]
missing_indicators=["", "NA", "NaN", "null"]
```

The helper encodes these as bracketed, comma-separated values:

```text
-distances=[dtw,erp]
-feature_columns=[temperature,pressure]
-MissingStrings=[,NA,NaN,null]
```

A string can also be passed to helper list encoders, but a Python list or tuple is clearer and less error-prone.

Current list encodings do not provide a general escaping mechanism for commas inside individual values. Avoid column names, missing indicators, and similar entries that contain the list delimiter.

### Separators

For a literal tab separator in Python, use:

```python
entry_separator="\t"
```

The helper converts it to the escaped form expected by the Java argument layer. The same normalization is applied to `array_separator`.

For ordinary delimiter characters:

```python
entry_separator=","
array_separator=":"
```

The shell may interpret characters such as `|`, `;`, `*`, or whitespace. Quote the complete `-name=value` argument when necessary:

```bash
java -jar Application/PFGAP.jar '-entry_separator=|' ...
```

Use the quoting rules of the active shell.

### Paths

Python callers pass paths as Python strings:

```python
train_file="../data/train.csv"
output_directory="../output/run_01"
```

For direct Java use, quote the complete argument when the path contains spaces:

```bash
java -jar Application/PFGAP.jar \
  '-train=/path/with spaces/train.csv' \
  '-out=/path/with spaces/output/'
```

Relative paths are resolved from the process working directory. Prefer absolute paths in schedulers and automation where the working directory may vary.

### Custom-reader parameters

The Python helper accepts custom-reader parameters as a dictionary:

```python
custom_reader_parameters={
    "encoding": "UTF-8",
    "strict": "true",
}
```

It encodes the dictionary as semicolon-separated assignments:

```text
-custom_reader_parameters=encoding=UTF-8;strict=true
```

Parameter names cannot be blank or contain `;` or `=`. Values cannot contain `;`. The Java parser rejects blank or duplicate names. Quote the full direct argument in shells where semicolons terminate commands:

```bash
java -jar Application/PFGAP.jar \
  '-custom_reader_parameters=encoding=UTF-8;strict=true' ...
```

See [Custom Readers](../extensions/Custom_Readers.md).

### Custom-distance descriptors

Custom distances are included in the distance list using descriptor forms such as:

```text
javadistance:path/to/file[:ClassName]
python:path/to/file[:FunctionName]
maple:path/to/file[:FunctionName]
```

Meta-distance descriptors use their registered `meta_...` prefix and a file path, with an optional method where supported. Paths are validated when arguments are parsed. Descriptor syntax and packaging are documented in [Custom Distances](../extensions/Custom_Distances.md).

## Derived and conditional behavior

Some options affect other settings or require companion options.

### Forest mode and purity

In the Python helper, omitting `forest_mode` selects regression when `regressor=True`; otherwise it selects classification. When explicit mode is `regression` or `isolation`, the helper adjusts the legacy regression flag accordingly. It also replaces the default `gini` purity with a mode-appropriate default for regression or isolation.

For clarity, set `forest_mode` explicitly and choose task-specific purity only when departing from the documented defaults.

### Missing values and imputation

When `has_missing_values` is omitted in Python, the helper derives it from selected imputation, CSR output, proximity-first initialization, and missing-aware distance options.

Requesting a full or sparse imputed output causes the corresponding imputation operation to be enabled. For example:

```python
return_imputed_training_csr=True
```

implies training-data imputation even if `impute_training_data` was initially false.

Missing-aware proximity distances are restricted to registered measures that support missing observations. The current Java argument validation accepts `nan_euclidean`, `nan_euclidean_i`, `dtwarow`, `dtwarow_i`, and `dtwarow_d` in `missing_proximity_distances`.

See [Imputation](../guides/Imputation.md), [Missing Values](../reference/Missing_Values.md), and [Imputed-Only Output](../reference/Imputed_Only_Output.md).

### DTW-aligned GAP updates

The legacy `DTWImpute` boolean and the newer `gap_update` option describe related behavior. Unless `gap_update` is set explicitly, the helper chooses `dtw_alignment` when `DTWImpute=True` and `standard` otherwise.

Prefer the explicit strategy form in new configurations:

```python
gap_update="dtw_alignment"
```

or:

```text
-gap_update=dtw_alignment
```

Valid current strategy values are `standard` and `dtw_alignment`.

### OOD output

When `return_ood_scores=True` during Python-driven training, the helper also enables collection of split-distance summaries because same-run validation needs those summaries. The currently accepted helper score type is `relative_support_exceedance`.

OOD scoring has additional training-time and evaluation-time requirements. See [OOD Scoring](../guides/OOD_Scoring.md).

### Dimension subsampling

Dimension selection is inactive when `subsample_dimensions=False` or the strategy is `ALL`. Current strategies are:

- `ALL`
- `SQRT`
- `LOG2`
- `FIXED_COUNT`
- `PROPORTION`

The Python helper also accepts selected aliases such as `fixed`, `count`, `prop`, and `log_2`, then sends the canonical value to Java.

`FIXED_COUNT` requires a positive integer count. `PROPORTION` requires a finite value in `(0, 1]`.

### Reader-specific requirements

Reader options form a compatibility contract rather than an independent collection of toggles. Examples include:

- `PER_FILE_PARQUET` requires `file_pattern` in the current Python helper.
- Lazy `PER_FILE_PARQUET` does not currently support imputation or imputed-data output through the helper.
- Long-form readers may require `id_column`, `time_column`, and feature or label column selections.
- HDF5 readers use `hdf5_dataset_path` and, where applicable, `hdf5_label_dataset_path`.
- Custom readers require an appropriate descriptor and may accept custom parameters.

Consult [Readers](../data/Readers.md) before selecting reader-specific options.

## Recommended configuration patterns

### Use scripts as experiment records

Store each reproducible run in a version-controlled Python or shell script. Include:

- input paths or dataset identifiers;
- forest mode;
- tree and candidate counts;
- distance selection;
- seed and worker count;
- representation and reader settings;
- missing-data and standardization settings;
- requested outputs; and
- a unique output directory and model name.

Do not rely only on interactive shell history or notebook state.

### Set important defaults explicitly

For durable experiments, explicitly set options whose defaults could change or whose meaning matters scientifically:

```python
forest_mode="classification"
num_trees=101
r=5
seed=42
num_workers=4
data_dimension=1
numeric_data=True
standardization="none"
```

There is no need to restate every implementation default. Prioritize task definition, data interpretation, randomness, performance-sensitive behavior, preprocessing, and requested outputs.

### Use a configuration dictionary in Python

A dictionary makes closely related runs easier to compare:

```python
import PF_wrapper as PF

common = {
    "train_file": "../data/train.csv",
    "test_file": "../data/test.csv",
    "exists_testlabels": True,
    "forest_mode": "classification",
    "num_trees": 101,
    "r": 5,
    "seed": 42,
    "num_workers": 4,
    "data_dimension": 1,
    "numeric_data": True,
    "entry_separator": ",",
    "file_has_header": False,
    "target_column": "first",
    "return_predictions": True,
    "save_model": True,
}

status = PF.train(
    **common,
    model_name="dtw_run",
    output_directory="../output/dtw_run",
    distances=["dtw"],
)

if status != 0:
    raise SystemExit(status)
```

Create a new dictionary or use keyword overrides for each run. Avoid mutating one shared dictionary across unrelated experiments in ways that make the final configuration difficult to reconstruct.

### Keep training and prediction data contracts aligned

A saved model does not remove the need to describe evaluation input correctly. Keep these settings compatible between training and prediction:

- observation dimensionality;
- numeric versus generic representation;
- numeric storage where relevant;
- reader and dataset layout;
- separators and header handling;
- target placement or separate-label arrangement;
- feature ordering and column selection;
- standardization statistics; and
- missing-value handling.

A prediction file may contain different numbers of observations or valid variable-length sequences, but it must still satisfy the fitted model's supported representation and preprocessing contract.

### Separate outputs by run

Use one output directory per experiment or evaluation:

```text
output/
├── classification_dtw_seed42/
├── classification_erp_seed42/
└── classification_dtw_seed43/
```

This prevents later runs from overwriting models, statistics, predictions, proximities, or metadata from earlier runs.

## Validation and errors

Configuration is validated in more than one place:

- Python checks selected types, enumerations, numerical ranges, and incompatible helper combinations before launching Java.
- Java parses every supplied option and rejects unknown names.
- Java performs cross-option validation after parsing.
- Readers, preprocessors, tasks, and exporters may perform additional validation when the corresponding component is initialized or used.

A successful helper call construction does not guarantee that the input data, model, or task combination is valid.

### Unknown direct option

An option not recognized by the current Java application causes the run to abort. Check spelling, capitalization, revision compatibility, and whether the option is Python-only.

### Invalid Python keyword

An unsupported helper keyword raises a Python `TypeError` before Java is launched. Compare the call with the current function signature.

### Valid option, invalid combination

Options may be individually valid but incompatible together. Common causes include:

- a reader without its required path or column settings;
- imputation requested for a lazy representation that does not support mutation;
- a missing-aware workflow with an incompatible distance;
- KNN initialization without `knn_distances`;
- an invalid dimension-selection count or proportion;
- OOD output without required training summaries;
- standardization statistics incompatible with the evaluation data; or
- outputs requested for a task that cannot produce them.

### Python and Java runs differ

Compare:

1. application and helper revisions;
2. effective option values, including helper defaults;
3. working directory and resolved paths;
4. seed and worker count;
5. input ordering and labels;
6. selected reader and numeric storage;
7. custom extension versions; and
8. model and standardization artifacts.

The helper builds an argument list and launches Java without a shell, so shell quoting differences affect direct commands but normally do not affect helper calls.

## Configuration checklist

Before launching a substantive run, verify:

- [ ] `PFGAP.jar`, `PF_wrapper.py`, documentation, and custom extensions are revision-compatible.
- [ ] `forest_mode` matches the task.
- [ ] Input and label paths resolve from the actual working directory.
- [ ] The selected reader matches the physical data layout.
- [ ] `data_dimension`, `numeric_data`, and `numeric_storage` match the observation representation.
- [ ] Separators, header handling, target placement, and column selections match the files.
- [ ] Distance names and custom descriptors are valid.
- [ ] Missing-data and imputation settings are compatible with the reader and distances.
- [ ] Standardization settings and statistics are appropriate for training or evaluation.
- [ ] Seed and worker count are recorded.
- [ ] Requested outputs are supported by the task.
- [ ] The output directory and model name are unique for the run.
- [ ] The Java heap is suitable for the dataset and requested artifacts.

## Next steps

- Use the [Configuration Reference](../reference/Configuration_Reference.md) for the complete public property catalog.
- Use the [CLI Reference](../reference/CLI_Reference.md) for exact Java option names and encodings.
- Follow [Classification](../guides/Classification.md), [Regression](../guides/Regression.md), or [Imputation](../guides/Imputation.md) for task-specific configuration.
- Read [Standardization](../guides/Standardization.md) before enabling preprocessing.
- Read [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md) and [Readers](../data/Readers.md) before choosing a reader for a large or per-file dataset.
- Consult [Outputs](../reference/Outputs.md) before building automation around generated artifacts.
