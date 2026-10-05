# Parallelism and Reproducibility

This guide explains how to configure PFGAP's worker count and random seed. It also describes the execution settings and artifacts that should be retained when repeating or comparing runs.

## Prerequisites

Before following this guide:

1. Complete [Installation](../getting-started/Installation.md).
2. Read [Configuration](../getting-started/Configuration.md) for interface, path, and option conventions.
3. Read the guide for the task being run, such as [Classification](Classification.md), [Regression](Regression.md), [Imputation](Imputation.md), or [Outlier Scoring](Outlier_Scoring.md).

Use `Application/PFGAP.jar` and `Application/PF_wrapper.py` from the same PFGAP revision.

## Worker configuration

PFGAP uses one application-owned worker budget for supported parallel work within an experiment repetition.

Configure the budget with:

```python
num_workers=4
```

Direct Java form:

```text
-num_workers=4
```

Accepted values are:

- `1` for sequential execution;
- a positive integer greater than `1` for bounded parallel execution; or
- `-1` to use every processor available to the Java Virtual Machine.

The default is one worker.

## Sequential execution

Use:

```python
num_workers=1
```

or:

```text
-num_workers=1
```

Sequential execution creates no PFGAP worker threads. It is useful for first-run testing, debugging, small datasets, and repeatability checks.

Sequential execution does not change the task, distance, reader, or output configuration. It changes only the available PFGAP worker budget.

## Bounded parallel execution

Use a positive worker count greater than one:

```python
num_workers=8
```

or:

```text
-num_workers=8
```

PFGAP creates a private work-stealing pool with that worker count. Worker threads are named `pfgap-worker-1`, `pfgap-worker-2`, and so on.

The private pool is used instead of Java's common fork-join pool. Supported nested PFGAP work shares the same runtime rather than creating unrelated CPU executors.

## Use all available processors

Use:

```python
num_workers=-1
```

or:

```text
-num_workers=-1
```

PFGAP resolves `-1` to:

```text
max(1, Runtime.getRuntime().availableProcessors())
```

The resulting value is the processor count visible to the JVM. Containers, schedulers, and operating-system limits can affect that count.

Use an explicit positive worker count when a fixed budget is required across machines or scheduled jobs.

## Invalid worker counts

`0` and values below `-1` are invalid.

For example, this is rejected:

```text
-num_workers=0
```

Use `-1` or a positive integer.

## One runtime per repetition

Each training or evaluation repetition receives its own parallel runtime. Training, prediction, imputation, proximity calculation, isolation scoring, and supported output preparation within that repetition receive the same runtime.

The runtime is closed after the repetition completes. A parallel runtime waits for its private pool to terminate and interrupts remaining workers if normal shutdown does not complete within the runtime's shutdown period.

This repetition-level ownership prevents parallel task state from carrying into another repetition.

## Repetitions

Configure the number of experiment repetitions with:

```python
repeats=5
```

Direct Java form:

```text
-repeats=5
```

Each repetition performs a separate training or evaluation repetition and records repetition-specific artifacts where applicable.

Repetitions do not replace an explicit random seed. Set both options when a run must be repeatable.

## Random seed

Set the application seed with:

```python
seed=42
```

Direct Java form:

```text
-seed=42
```

PFGAP initializes its application random-number generator from the supplied long integer. If no seed is supplied, the application uses an unseeded `Random` instance.

Always specify a seed for a run that must be repeated or compared.

## Reproducible training example

### Python helper

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    exists_testlabels=True,
    forest_mode="classification",
    repeats=1,
    num_trees=101,
    r=5,
    seed=42,
    num_workers=4,
    shuffle=True,
    bootstrap_trees=True,
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    file_has_header=False,
    target_column="first",
    save_model=True,
    model_name="classification_seed42",
    return_predictions=True,
    output_directory="../output/classification_seed42",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  -test=data/test.csv \
  -exists_testlabels=true \
  -forest_mode=classification \
  -repeats=1 \
  -trees=101 \
  -r=5 \
  -seed=42 \
  -num_workers=4 \
  -shuffle=true \
  -bootstrap_trees=true \
  -is2D=false \
  -isNumeric=true \
  -entry_separator=, \
  -csv_has_header=false \
  -target_column=first \
  -savemodel=true \
  -modelname=classification_seed42 \
  -get_predictions=true \
  -out=output/classification_seed42/
```

Create the output directory before direct Java execution.

## Settings that affect repeated results

A seed is one part of the execution record. Preserve the following settings when reproducing a run.

### Forest construction

Record:

- `forest_mode`;
- `num_trees` or `trees`;
- `r`;
- `bootstrap_trees`;
- `shuffle`;
- `max_depth`;
- purity settings;
- regression or isolation branch settings; and
- dimension-selection settings.

These options affect the fitted forest and its outputs.

### Distances

Record:

- the ordered distance list;
- custom-distance descriptors;
- custom extension files and versions;
- KNN distances used for initial imputation; and
- missing-aware distances used for proximity-first imputation.

Custom Python, Maple, and Java implementations must remain unchanged when reproducing a run.

### Data and readers

Preserve:

- the exact input files;
- observation ordering;
- labels or targets;
- reader type;
- eager or lazy access;
- file patterns;
- separators and headers;
- selected columns;
- numeric storage; and
- one-dimensional or two-dimensional representation.

For per-file datasets, keep the discovered file set and ordering unchanged.

### Preprocessing

Preserve:

- standardization method;
- standardization scope;
- variance convention;
- fitted statistics files;
- missing-value indicators;
- initialization strategy;
- initial imputer;
- update strategy; and
- imputation iteration count.

For reusable standardization scopes, apply the original fitted training statistics during later evaluation.

### Worker count

Record the effective worker count. Do not use `-1` when the same numeric budget must be used on machines that expose different processor counts.

### Application environment

Record:

- the PFGAP revision;
- Java version;
- operating system and architecture;
- JVM heap settings;
- enabled optional runtimes; and
- relevant custom extension versions.

## Shuffle and bootstrap settings

The dataset shuffle setting is:

```python
shuffle=True
```

Direct Java form:

```text
-shuffle=true
```

Tree bootstrap sampling is configured with:

```python
bootstrap_trees=True
```

Direct Java form:

```text
-bootstrap_trees=true
```

Both settings interact with the random seed. Record them explicitly for repeatable experiments.

## Randomized tie handling

PFGAP uses randomized selection for supported equal-choice situations, including majority-vote ties and equal minimum-distance choices. These choices use the application random-number generator.

Supplying `seed` makes this random source repeatable for a compatible run configuration and PFGAP revision.

## Parallel task behavior

PFGAP partitions supported integer ranges into disjoint terminal ranges. The terminal ranges collectively cover the requested work exactly once.

The runtime creates a modest multiple of the worker count as terminal tasks, allowing work stealing while avoiding a large number of tiny tasks.

A task failure is propagated to the calling workflow. Interrupted parallel work restores the thread's interrupted status and aborts the affected operation.

## Parallelism and lazy data

Multiple workers can request lazy observations concurrently. Increasing the worker count can increase:

- simultaneous file reads;
- materialization work;
- cache pressure;
- open format-specific resources; and
- temporary memory use.

Custom readers must declare and implement the thread-safety behavior required by the selected configuration. See [Eager and Lazy Data](Eager_and_Lazy_Data.md) and [Custom Readers](../extensions/Custom_Readers.md).

## Parallelism and imputation

Training and test imputation receive the repetition's parallel runtime. Proximity calculations used during iterative imputation also use that runtime.

Final requested proximity artifacts are computed after iterative imputation from the final imputed data and forest state.

See [Imputation](Imputation.md) for initialization, update, and output configuration.

## Parallelism and scoring

Supported prediction, enhanced evaluation, isolation scoring, OOD scoring, and proximity operations use the repetition's runtime.

Output requests do not create a separate user-configurable worker pool. They share the configured worker budget for the repetition.

## Repeating saved-model evaluation

For repeatable evaluation of a saved model, preserve:

- the saved model;
- evaluation input and ordering;
- reader and representation settings;
- standardization statistics;
- missing-value handling;
- output requests;
- seed;
- worker count; and
- PFGAP revision.

The saved forest topology is reused across evaluation repetitions. Evaluation may update transient test-leaf membership needed by prediction and proximity operations without retraining the forest.

## Compare sequential and parallel runs

To compare execution modes, run the same configuration twice and change only `num_workers`:

```python
# Sequential
num_workers=1

# Parallel
num_workers=4
```

Use separate output directories so artifacts are not overwritten:

```text
output/
├── classification_seed42_workers1/
└── classification_seed42_workers4/
```

Record both worker counts with the outputs.

## Vector API option

The Python helper exposes:

```python
use_vector_api=True
```

Direct Java form:

```text
-use_vector_api=true
```

This option is separate from the worker count. Worker parallelism controls concurrent PFGAP tasks; vectorized execution controls use of supported vectorized numeric operations.

Record the setting when comparing performance or reproducing an execution environment.

## Warmup and garbage collection

The application configuration includes optional Java warmup and repetition-level garbage-collection behavior. These settings affect execution procedure and timing rather than the task definition.

When reporting performance measurements, keep warmup, repetition count, garbage-collection behavior, heap settings, worker count, and Java runtime consistent.

The complete option names and defaults are documented in the [Configuration Reference](../reference/Configuration_Reference.md) and [CLI Reference](../reference/CLI_Reference.md).

## Output isolation

Use a unique output directory for each configuration:

```text
output/
├── run_seed42_workers1/
├── run_seed42_workers4/
└── run_seed43_workers4/
```

This prevents models, predictions, proximities, scores, statistics, and result records from different runs from overwriting one another.

When `repeats` is greater than one, PFGAP uses repetition-aware artifact paths for supported outputs.

## Reproducibility checklist

Before repeating or comparing a run, confirm:

- [ ] The same PFGAP revision is used.
- [ ] The same Java runtime and relevant JVM options are used.
- [ ] The same input files and observation order are used.
- [ ] The same labels or targets are used.
- [ ] The same reader, representation, and numeric storage are used.
- [ ] The same forest mode and structural settings are used.
- [ ] The same distance list and extension implementations are used.
- [ ] The same standardization and imputation settings are used.
- [ ] The same fitted preprocessing artifacts are used.
- [ ] The same seed is used.
- [ ] The same effective worker count is used.
- [ ] The same shuffle and bootstrap settings are used.
- [ ] The same output requests are used.
- [ ] A separate output directory protects the run's artifacts.

## Common problems

### `num_workers` is rejected

Use `-1` or a positive integer. Zero and values below `-1` are invalid.

### `num_workers=-1` gives different budgets on two machines

`-1` uses the processor count visible to each JVM. Use the same explicit positive worker count on both machines.

### Results change when the seed is omitted

Set `seed` explicitly. An omitted seed uses an unseeded application random-number generator.

### Results change after the input directory changes

Restore the same file set and ordering. Per-file dataset ordering affects labels and instance-indexed outputs.

### Parallel execution uses more memory

Concurrent tasks can hold more observations, distance workspaces, lazy materializations, and temporary results at the same time. Reduce `num_workers` or use sequential execution.

### A custom reader fails under parallel access

Use a thread-safe custom reader or configure the extension according to its documented access mode. See [Custom Readers](../extensions/Custom_Readers.md).

### A run cannot be compared with an earlier model

Use the same PFGAP revision, data contract, preprocessing artifacts, custom extensions, seed, and worker configuration. See [Model Persistence](../reference/Model_Persistence.md).

### Output files are overwritten

Use a unique output directory and model name for every seed, worker count, repetition set, or task configuration.

## Next steps

- Read [Configuration Reference](../reference/Configuration_Reference.md) for exact defaults and accepted values.
- Read [CLI Reference](../reference/CLI_Reference.md) for direct Java option names.
- Read [Eager and Lazy Data](Eager_and_Lazy_Data.md) for concurrent lazy materialization considerations.
- Read [Model Persistence](../reference/Model_Persistence.md) for saved-state compatibility.
- Read [Outputs](../reference/Outputs.md) for repetition-aware artifact names and schemas.
