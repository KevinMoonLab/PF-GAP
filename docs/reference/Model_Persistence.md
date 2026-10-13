# Model Persistence

This reference explains how PFGAP saves and loads trained models. New model files use a versioned Java-serialization envelope containing the fitted forest, an explicit persistence mode, the original training-set size, an optional training dataset, and a snapshot of model-related application context.

PFGAP currently supports two persistence modes:

- `EMBEDDED`: saves the complete training dataset representation and preserves operations that require arbitrary training observations.
- `PREDICTION_ONLY`: saves an eagerly trained forest without the complete training dataset, substantially reducing model size when eager training data would otherwise be copied into the `.ser` file.

For training and evaluation examples, see [Classification](../guides/Classification.md), [Regression](../guides/Regression.md), [Outlier Scoring](../guides/Outlier_Scoring.md), and [OOD Scoring](../guides/OOD_Scoring.md).

## Save a model

### Python

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/training.csv",
    save_model=True,
    model_name="classification_model",
    model_persistence_mode="embedded",
)
if status != 0:
    raise SystemExit(status)
```

To omit eager training observations from the saved artifact:

```python
status = PF.train(
    train_file="../data/training.csv",
    save_model=True,
    model_name="classification_model_small",
    model_persistence_mode="prediction_only",
)
```

The Python wrapper accepts case-insensitive underscore or hyphen forms, including `prediction_only` and `prediction-only`.

### Direct Java

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -train=data/training.csv \
  -savemodel=true \
  -modelname=classification_model \
  -model_persistence_mode=embedded
```

For prediction-only persistence:

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -train=data/training.csv \
  -savemodel=true \
  -modelname=classification_model_small \
  -model_persistence_mode=prediction_only
```

`EMBEDDED` is the default and preserves the established saving behavior. Use a distinct model name and output directory for each run that must be retained.

## Persistence modes

### `EMBEDDED`

`EMBEDDED` saves:

- the fitted forest;
- the complete `ListObjectDataset` training representation;
- the original training-set size;
- the persistence-mode identifier; and
- the `AppContextSnapshot`.

For eagerly loaded data, the saved model contains the complete training observations. For lazy data, it contains the dataset's lightweight references and saved reader specifications, so the original external data may still be required when those references are resolved.

An embedded model supports the existing post-load workflows, including prediction, supported OOD scoring, train/train and test/train proximities, and applicable imputation operations.

### `PREDICTION_ONLY`

`PREDICTION_ONLY` is a v1 deployment option intended specifically for models trained from eagerly loaded data. It saves:

- the fitted forest and complete tree topology;
- materialized split exemplars already retained by the eager-trained forest;
- leaf prediction state;
- retained split-distance summaries;
- in-bag and out-of-bag placement metadata;
- bootstrap multiplicities;
- the original training-set size; and
- the `AppContextSnapshot`.

It does **not** save the complete training dataset.

A prediction-only model supports:

- ordinary prediction;
- enhanced prediction output;
- supported distance-based OOD scoring when the necessary summaries were collected during training;
- train/train proximities; and
- test/train proximities.

Test/train proximities remain available because test observations can traverse the saved trees using the eagerly materialized split exemplars, while the forest retains the training-instance placement and bootstrap metadata required by the PFGAP proximity calculation.

A prediction-only model does **not** support:

- training-data imputation;
- testing-data imputation;
- imputed-value output; or
- any future operation requiring arbitrary training feature values.

If evaluation requests testing-data imputation from a prediction-only model, PFGAP fails explicitly and instructs the user to save the model in `EMBEDDED` mode.

For v1, attempting to save lazy training data in `PREDICTION_ONLY` mode is rejected. Referenced-data persistence modes, source manifests, path relocation, and eager-source provenance are deferred beyond v1.

## Load a model

### Python

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

### Direct Java

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

## Serialized model format

New model files contain one versioned `PersistedModel` envelope. The envelope records:

1. persistence format version;
2. `ModelPersistenceMode`;
3. fitted `ProximityForest`;
4. optional `ListObjectDataset` training data;
5. original training-set size; and
6. `AppContextSnapshot`.

The envelope validates that:

- `EMBEDDED` models contain training data;
- the embedded dataset size matches the declared training size;
- `PREDICTION_ONLY` models do not contain training data; and
- the declared training size is nonnegative.

The file is a Java object-serialization stream rather than JSON, CSV, or a language-neutral model format. Do not edit it as text.

### Legacy compatibility

PFGAP continues to load legacy model streams that contain three sequential objects:

1. `ProximityForest`;
2. `ListObjectDataset`; and
3. `AppContextSnapshot`.

Legacy models are treated as `EMBEDDED`. Newly saved models use the versioned envelope.

## Fitted forest

The serialized forest contains the trained tree ensemble and fitted splitter state, including selected distances, parameters, exemplars, branch structure, leaf state, training placement metadata, and retained OOD state.

Splitters no longer retain their complete winning child datasets after training. The winning child datasets are returned to the tree-training process as local state, while each fitted splitter retains only post-training routing state. This prevents the forest itself from recursively retaining training subsets that are unnecessary after training.

When split-distance summaries were collected during training, the forest is the authority on whether the loaded model can produce distance-based OOD scores.

## Training dataset and training size

The complete training dataset is present only in `EMBEDDED` models. It is required by workflows that read arbitrary training feature values, most notably imputation.

Every new model also records the original training-set size independently of the dataset object. This permits prediction-only evaluation to:

- reconstruct proximity contexts with the correct matrix dimensions;
- evaluate isolation scores that require the training count; and
- report the correct `trainingInstanceCount` in experiment results.

Training size does not substitute for actual training values in imputation.

## Application snapshot

The `AppContextSnapshot` preserves model-related configuration needed to reconstruct the fitted workflow.

### Random behavior

The snapshot stores:

- `rand_seed`;
- `config_majority_vote_tie_break_randomly`;
- `config_skip_distance_when_exemplar_matches_query`; and
- `config_use_random_choice_when_min_distance_is_equal`.

When loaded, a saved seed reinitializes the application random-number generator. If the saved seed is null, PFGAP creates an unseeded generator.

### Training identity

The snapshot stores:

- `training_file`;
- `training_labels`; and
- `datasetName`.

These identify the source training configuration. They do not reconstruct omitted training observations in a prediction-only model.

### Observation representation

The snapshot stores:

- `is2D`;
- `isNumeric`; and
- `numericStorageType`.

Evaluation data must produce a compatible logical observation representation.

### Forest task and structure

The snapshot stores:

- `forest_mode`;
- `isRegression`;
- `purity_measure`;
- `purity_threshold`;
- `voting`;
- `num_trees`;
- `num_candidates_per_split`;
- `random_dm_per_node`; and
- `max_depth`.

When `forest_mode` is present, loading derives the regression boolean from it. For an older snapshot without `forest_mode`, PFGAP derives classification or regression mode from the saved legacy regression flag.

### Distances and descriptors

The snapshot stores:

- `enabled_distance_measures`;
- `userdistances`;
- `Descriptors`; and
- `meta_predictions`.

These retain configured distance identities and supporting descriptor state used by interoperability and meta-distance workflows.

### Missing-value tokens

The snapshot contains the training `MissingStrings` set. The current load path does not restore it into the active invocation. Configure the missing-value tokens needed by evaluation data when they differ or when evaluation-time parsing requires them.

### Proximity storage

The snapshot stores `useSparseProximities`, which restores the saved internal proximity-storage preference.

### Standardization

The snapshot stores:

- `standardizationConfig`; and
- `standardizationStats`.

Loading restores both values. If standardization is enabled but fitted statistics are absent, model loading fails. When statistics are present, PFGAP validates them against the restored standardization configuration.

### Class-label mapping

When the training dataset is a `ListObjectDataset`, the snapshot captures its initial class-label mapping when available.

During prediction-only classification evaluation, the saved mapping is passed directly to dataset preparation so test labels can be remapped without requiring the complete training dataset. Regression workflows ignore this map.

### Lazy reader reconstruction

The snapshot stores serializable lazy-reader specifications selected for the model. The current capture path retains the training reader specification under the model reader registry.

During loading, PFGAP reconstructs readers from those specifications before replacing the active reader registry. If reconstruction fails, the previous registry remains intact and newly constructed replacement readers are closed.

Runtime-only reader registrations that do not have a serializable reconstruction specification are not part of the saved model snapshot.

## State not restored from the snapshot

Invocation-level choices remain controlled by the current evaluation command. The snapshot does not restore the following as model settings:

- `testing_file`;
- `testing_labels`;
- `output_dir`;
- `eval`;
- `num_repeats`;
- `shuffle_dataset`;
- `savemodel`;
- `modelPersistenceMode`;
- `getprox`;
- `get_training_outlier_scores`;
- `get_predictions`;
- `modelname`;
- `impute_train`;
- `impute_test`;
- `exists_testlabels`;
- `return_enhanced_outputs`;
- `return_ood_scores`; and
- `ood_score_type`.

The persistence mode is stored in the model envelope and reported by the loaded-model representation. The `AppContext.modelPersistenceMode` setting controls how a **new** model is saved; it is not restored as an evaluation-time request.

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

A later evaluation can choose a supported OOD scorer independently, but it cannot add split-distance summaries that were not retained by the fitted forest.

### Proximities

PFGAP proximity contexts are rebuilt from the loaded forest rather than serialized as separate matrices or contexts. Retained state includes tree topology, terminal-node training placement, out-of-bag assignments, and bootstrap multiplicities.

Train/train proximity computation requires the retained training placement and original training size, not materialized training feature values.

Test/train proximity computation additionally requires test observations to traverse the fitted tree topology. Prediction-only models support this because v1 restricts the mode to eager-trained forests whose split exemplars are already materialized.

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

If a configured JAR path does not exist, PFGAP prints a warning. A model containing an unavailable custom class can then fail with a class-resolution error.

Configure custom Java descriptors before loading the model so the necessary JARs are available to the model loader.

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

Meta distances rely on external pretrained-model or prediction-source behavior. Retain the referenced implementation, prediction source, method selection, and required artifacts alongside the PFGAP model.

The snapshot can retain meta prediction state, but that does not make absent external implementations portable.

## Reader dependencies

An embedded model saved from lazy training data restores serialized training-reader specifications where available. Evaluation data still require a compatible configured test reader.

Compatibility includes:

- one-dimensional or two-dimensional representation;
- numeric or generic value type;
- primitive numeric storage where required;
- feature or channel ordering;
- time and identifier-column semantics;
- standardization behavior; and
- missing-value handling.

The evaluation physical format may differ from the training format when both readers produce an equivalent supported representation.

`PREDICTION_ONLY` currently rejects lazy training data rather than persisting unresolved lazy exemplars.

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

PFGAP model persistence uses Java serialization. New model files declare a persistence format version. The current loader rejects an envelope whose format version does not match the supported version.

The loader separately recognizes the legacy three-object format described above. Successful loading still requires serialized class names and compatible serialized class structures to remain available.

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

`AppContextSnapshot` and `PersistedModel` declare their own `serialVersionUID` values. Other serialized classes use their own Java serialization identities and class definitions.

Envelope and snapshot identities do not guarantee complete compatibility across revisions because the stream also contains the forest and, for embedded models, the training dataset object graph.

## Evaluation workflow

A saved-model evaluation proceeds conceptually as follows:

1. Parse invocation-level evaluation settings and external descriptors.
2. Load either the versioned envelope or a legacy model stream.
3. Apply the saved snapshot to model-related application state.
4. Reconstruct saved training lazy-reader specifications when present.
5. Restore and validate standardization configuration and statistics.
6. Read evaluation data with the current evaluation reader configuration.
7. Prepare evaluation state using optional training data, the persisted training size, and the saved label map.
8. Reject imputation early when the loaded model lacks complete training observations.
9. Run the requested prediction, scoring, proximity, or structured-output operations.
10. Write the requested evaluation artifacts and preserve the original training count in result metadata.

## Model size

Model size depends on:

- number and depth of trees;
- splitter exemplars and parameter state;
- retained OOD split-distance summaries;
- retained training placement and bootstrap metadata;
- embedded training-data size and primitive storage in `EMBEDDED` mode;
- distance descriptors and meta state;
- class-label mapping;
- standardization statistics; and
- lazy-reader specifications.

For eager training data, `PREDICTION_ONLY` removes the complete serialized training dataset while retaining the forest state needed for prediction, OOD scoring, and proximity calculations. The artifact is therefore expected to be substantially smaller when the training observations dominate the embedded model size.

The forest still retains selected exemplars and integer training-placement metadata, so prediction-only persistence is not a topology-only or zero-training-state format.

## Security

Java deserialization can instantiate classes named in the serialized stream. Load only model files from trusted sources.

Custom Java model loading also adds configured external JARs to a class loader. Use only trusted custom-distance JARs and descriptors.

Do not treat a model file as safe merely because it has a familiar model name or was placed in an expected directory.

## Common problems

### `PREDICTION_ONLY` rejects the training dataset

The v1 prediction-only implementation supports only eagerly loaded training data. Save a lazy-trained model in `EMBEDDED` mode.

### Imputation is unavailable after loading

The model was saved in `PREDICTION_ONLY` mode and does not contain complete training observations. Retrain or resave the model in `EMBEDDED` mode to retain imputation support.

### The model file cannot be found

Use the exact path produced by training. Relative paths are resolved from the evaluation process working directory.

### The model parent directory does not exist during saving

Create the output directory before direct Java execution or use the coordinated application output workflow that prepares artifact paths.

### Deserialization reports `ClassNotFoundException`

Use the matching PFGAP JAR and provide required custom Java JARs through configured descriptors before loading.

### Deserialization reports an incompatible-class error

Load the model with the PFGAP revision and dependencies used to create it.

### The model format version is unsupported

Use a PFGAP revision supporting the saved envelope version, or use the revision that created the model. Legacy three-object streams remain separately recognized.

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

## Deferred persistence work

The following persistence options are intentionally deferred beyond v1:

- fully referenced training data with referenced split exemplars;
- referenced training data with embedded split exemplars;
- eager-reader provenance capture;
- source manifests and integrity verification;
- data-source relocation and source-root overrides; and
- model-local lazy-reader registries.

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
