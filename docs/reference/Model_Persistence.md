# Model Persistence

This reference explains how PFGAP saves and loads trained models. A saved model is a Java serialization stream containing the fitted forest, the training dataset, and a snapshot of the model-related application context.

For training and evaluation examples, see [Classification](../guides/Classification.md), [Regression](../guides/Regression.md), [Outlier Scoring](../guides/Outlier_Scoring.md), and [OOD Scoring](../guides/OOD_Scoring.md).

## Save a model

Python:

```python
save_model=True
model_name="classification_model"
```

Direct Java:

```text
-savemodel=true
-modelname=classification_model
```

The model is written during training. Use a distinct model name and output directory for each run that must be retained.

## Load a model

Python:

```python
import PF_wrapper as PF

status = PF.predict(
    model_name="../output/classification_model/classification_model",
    testfile="../data/new_data.csv",
    exists_testlabels=False,
    forest_mode="classification",
    return_predictions=True,
    output_directory="../output/classification_predictions",
)

if status != 0:
    raise SystemExit(status)
```

Direct Java:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=true \
  -train=data/new_data.csv \
  -test=data/new_data.csv \
  -modelname=output/classification_model/classification_model \
  -forest_mode=classification \
  -get_predictions=true \
  -out=output/classification_predictions/
```

The current evaluation path receives the evaluation input through both `-train` and `-test`.

## Serialized object order

PFGAP writes three objects to the model file in this order:

```text
1. ProximityForest
2. ListObjectDataset training data
3. AppContextSnapshot
```

Loading expects the same order and object types.

The file is a Java object-serialization stream rather than JSON, CSV, or a language-neutral model format. Do not edit it as text.

## Fitted forest

The serialized forest contains the trained tree ensemble and the state required by its fitted splitters, including the selected distances, parameters, exemplars, branch structure, leaf state, and retained training-time capabilities.

When split-distance summaries were collected during training, the forest is the authority on whether the loaded model can produce distance-based OOD scores.

## Training dataset

The saved model includes the `ListObjectDataset` used for training. This allows evaluation workflows to use retained training observations for operations such as:

- test-to-training proximities;
- leaf-based prediction behavior;
- applicable imputation workflows;
- supervised training-state calculations; and
- distance comparisons requiring training exemplars.

Model size therefore depends on both the fitted forest and the serialized training dataset. Saving a model does not save only tree topology.

## Application snapshot

The `AppContextSnapshot` preserves model-related configuration needed to reconstruct the fitted workflow.

### Random behavior

The snapshot stores:

```text
rand_seed
config_majority_vote_tie_break_randomly
config_skip_distance_when_exemplar_matches_query
config_use_random_choice_when_min_distance_is_equal
```

When loaded, a saved seed reinitializes the application random-number generator. If the saved seed is null, PFGAP creates an unseeded generator.

### Training identity

The snapshot stores:

```text
training_file
training_labels
datasetName
```

These identify the source training configuration. They do not replace the serialized training dataset.

### Observation representation

The snapshot stores:

```text
is2D
isNumeric
numericStorageType
```

Evaluation data must produce a compatible logical observation representation.

### Forest task and structure

The snapshot stores:

```text
forest_mode
isRegression
purity_measure
purity_threshold
voting
num_trees
num_candidates_per_split
random_dm_per_node
max_depth
```

When `forest_mode` is present, loading derives the regression boolean from it. For an older snapshot without `forest_mode`, PFGAP derives classification or regression mode from the saved legacy regression flag.

### Distances and descriptors

The snapshot stores:

```text
enabled_distance_measures
userdistances
Descriptors
meta_predictions
```

These retain the configured distance identities and supporting descriptor state used by interoperation and meta-distance workflows.

### Missing-value tokens

The snapshot contains the training `MissingStrings` set. The current load path does not restore it into the active invocation. Configure the missing-value tokens needed by the evaluation data when they differ or when evaluation-time parsing requires them.

### Proximity storage

The snapshot stores:

```text
useSparseProximities
```

This restores the saved internal proximity-storage preference.

### Standardization

The snapshot stores:

```text
standardizationConfig
standardizationStats
```

Loading restores both values. If standardization is enabled but fitted statistics are absent, model loading fails. When statistics are present, PFGAP validates them against the restored standardization configuration.

### Class-label mapping

When the training dataset is a `ListObjectDataset`, the snapshot captures its initial class-label mapping when available.

Current public classification input uses integer labels. The retained mapping remains part of the model state used by compatible dataset behavior.

### Lazy reader reconstruction

The snapshot stores serializable lazy-reader specifications selected for the model. The current capture path retains the training reader specification under the model reader registry.

During loading, PFGAP reconstructs the readers from those specifications before replacing the active reader registry. If reconstruction fails, the previous registry remains intact and newly constructed replacement readers are closed.

Runtime-only reader registrations that do not have a serializable reconstruction specification are not part of the saved model snapshot.

## State not restored from the snapshot

Invocation-level choices remain controlled by the current evaluation command. The snapshot does not restore the following as model settings:

```text
testing_file
testing_labels
output_dir
eval
num_repeats
shuffle_dataset
savemodel
getprox
get_training_outlier_scores
get_predictions
modelname
impute_train
impute_test
exists_testlabels
return_enhanced_outputs
return_ood_scores
ood_score_type
```

This separation allows one saved model to be evaluated against different test inputs and to produce different requested artifacts.

The snapshot also does not restore current parser details such as separators, header flags, and embedded-target placement. Evaluation input must be configured so its reader materializes observations compatible with the saved model.

## Training-time capabilities

Some later operations depend on state retained while the forest is trained.

### OOD scoring

To create a model capable of distance-based OOD scoring, train with:

```python
collect_split_distance_summaries=True
save_model=True
```

Direct Java:

```text
-collect_split_distance_summaries=true
-savemodel=true
```

Requesting OOD output during a training invocation also enables the required summary collection for that run.

A later evaluation can choose the supported OOD scorer independently, but it cannot add split-distance summaries that were not retained by the fitted forest.

### Standardization

A model with enabled reusable standardization must contain fitted standardization statistics. Loading validates that the saved method, scope, variance convention, centers, and scales are compatible.

Per-series standardization uses local per-series transformation behavior rather than reusable dataset-level fit statistics.

### Custom distances

A custom or interoperability distance can require external code or runtime resources in addition to the serialized model. Save and retain those resources with the experiment.

## Custom Java distance loading

Before deserializing a model, PFGAP inspects the currently configured distance descriptors for `javadistance` entries.

For each descriptor, it reads the JAR path from:

```text
javadistance:path/to/file[:ClassName]
```

Existing JARs are added to a dedicated `URLClassLoader`. During object deserialization, PFGAP first attempts to resolve each serialized class through that loader, then falls back to normal Java deserialization class resolution.

If a configured JAR path does not exist, PFGAP prints a warning. A model that contains an unavailable custom class can then fail with a class-resolution error.

Configure the custom Java descriptors before loading the model so the necessary JARs are available to the model loader.

## Python and Maple distances

Python and Maple descriptors, scripts, functions, dependencies, and runtime installations are external model dependencies. The model preserves descriptor state but does not embed an external interpreter installation or guarantee that referenced scripts remain available.

Retain:

- the original descriptor;
- referenced script or source file;
- function or method name;
- runtime version;
- package dependencies; and
- any external model files used by the distance.

## Meta distances

Meta distances rely on external pretrained-model or prediction-source behavior. Retain the referenced implementation, prediction source, method selection, and any required artifacts alongside the PFGAP model.

The snapshot can retain meta prediction state, but that does not make absent external implementations portable.

## Reader dependencies

A saved model restores serialized training-reader specifications where available. Evaluation data still require a compatible configured test reader.

Compatibility includes:

- one-dimensional or two-dimensional representation;
- numeric or generic value type;
- primitive numeric storage where required;
- feature or channel ordering;
- time and identifier-column semantics;
- standardization behavior; and
- missing-value handling.

The evaluation physical format may differ from the training format when both readers produce an equivalent supported representation.

## Model filename and repetition naming

The configured `model_name` identifies the model artifact. For multiple repetitions, PFGAP applies its one-based repetition suffix before an extension, or at the end of an extensionless name:

```text
model
model_repeat_2
```

Use the exact artifact path produced by the training run when loading a model. The path is also recorded in `experiment_results.json` when model output is included in the artifact map.

## File-system behavior

`ModelIO.saveModel(...)` writes through `FileOutputStream` and replaces an existing file at the same path. Its immediate parent directory must already exist unless the surrounding artifact workflow creates it.

The model file and any required external dependencies should be moved or archived together.

## Version compatibility

PFGAP model persistence uses Java serialization. Successful loading requires serialized class names and compatible serialized class structures to remain available.

Use the same PFGAP revision to save and load a model whenever possible. A later revision may fail to deserialize an older model or may interpret restored behavior differently even when deserialization succeeds.

For a reproducible archive, retain:

- the PFGAP JAR and revision;
- the Java runtime version;
- the saved model;
- custom Java JARs;
- Python or Maple scripts and runtime details;
- meta-distance artifacts;
- standardization artifacts where separately written;
- the reader configuration;
- the original experiment configuration; and
- the input data version or checksum.

## Java serialization identity

`AppContextSnapshot` declares:

```text
serialVersionUID = 1L
```

Other serialized classes use their own Java serialization identities and class definitions. Snapshot identity alone does not guarantee complete model compatibility across revisions because the serialized stream also contains the forest and training dataset object graphs.

## Evaluation workflow

A saved-model evaluation proceeds conceptually as follows:

1. Parse invocation-level evaluation settings and external descriptors.
2. Load the serialized forest, training dataset, and context snapshot.
3. Apply the saved snapshot to model-related application state.
4. Reconstruct saved training lazy-reader specifications.
5. Restore and validate standardization configuration and statistics.
6. Read the evaluation data with the current evaluation reader configuration.
7. Run prediction, scoring, proximity, imputation, or structured-output operations requested by the current invocation.
8. Write the requested evaluation artifacts.

## Model size

Model size depends on:

- number and depth of trees;
- splitter exemplars and parameter state;
- retained OOD split-distance summaries;
- serialized training-data size and primitive storage;
- distance descriptors and meta state;
- class-label mapping;
- standardization statistics; and
- lazy-reader specifications.

Using lazy training data does not imply that the serialized model is only a lightweight pointer file. The saved stream includes the serialized training dataset object and its references according to that dataset's serializable representation.

## Security

Java deserialization can instantiate classes named in the serialized stream. Load only model files from trusted sources.

Custom Java model loading also adds configured external JARs to a class loader. Use only trusted custom-distance JARs and descriptors.

Do not treat a model file as safe merely because it has a familiar model name or was placed in an expected directory.

## Common problems

### The model file cannot be found

Use the exact path produced by training. Relative paths are resolved from the evaluation process working directory.

### The model parent directory does not exist during saving

Create the output directory before direct Java execution or use the coordinated application output workflow that prepares artifact paths.

### Deserialization reports `ClassNotFoundException`

Use the matching PFGAP JAR and provide any required custom Java JARs through the configured descriptors before loading.

### Deserialization reports an incompatible-class error

Load the model with the PFGAP revision and dependencies used to create it.

### A custom Java JAR warning appears

Correct the JAR path in the active `javadistance` descriptor. Relative descriptor paths resolve from the process working directory.

### Standardization loading fails

The saved model enables standardization but does not contain compatible fitted statistics. Use a model saved with valid training statistics and the matching PFGAP revision.

### Evaluation data have a different shape

Use a reader and preprocessing configuration that produce the same logical representation, dimensionality, feature ordering, and numeric behavior as training.

### OOD output is unavailable

The fitted forest does not contain the required split-distance summaries. Use a model trained with summary collection enabled.

### Output requests appear to have changed after loading

Prediction, proximity, enhanced-output, OOD, and evaluation-path settings are invocation-level choices. Configure them on the current evaluation command.

### A runtime-only custom reader is unavailable

Runtime-only reader objects are not serialized as reconstruction specifications. Use a serializable reader specification or register the required runtime reader in the evaluation process.

### A moved model cannot find external resources

Update descriptor and input paths or preserve the original relative directory structure. External scripts, JARs, data sources, and meta-model artifacts are not embedded automatically.

## Related documentation

- [Configuration Reference](Configuration_Reference.md)
- [CLI Reference](CLI_Reference.md)
- [Outputs](Outputs.md)
- [Standardization](../guides/Standardization.md)
- [OOD Scoring](../guides/OOD_Scoring.md)
- [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md)
- [Custom Distances](../extensions/Custom_Distances.md)
- [Custom Readers](../extensions/Custom_Readers.md)
- [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md)
