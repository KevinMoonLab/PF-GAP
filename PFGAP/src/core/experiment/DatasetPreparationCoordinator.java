package core.experiment;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.DatasetReader;
import datasets.readers.DatasetReaderFactory;
import datasets.readers.ReaderOptions;
import datasets.readers.ReaderType;
import imputation.util.MissingIndicesBuilder;
import preprocessing.standardization.StandardizationConfig;
import preprocessing.standardization.StandardizationPipeline;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Objects;

/**
 * Coordinates dataset reading, preprocessing, compatibility validation, and
 * publication to {@link AppContext} for experiment execution.
 *
 * <p>This class owns dataset preparation only. It does not train forests,
 * evaluate models, compute proximity, write artifacts, or manage parallel
 * runtimes.</p>
 *
 * <p>The preparation order intentionally preserves the established PFGAP
 * workflow:</p>
 *
 * <ol>
 *     <li>Prepare supplied standardization statistics.</li>
 *     <li>Read the original training dataset.</li>
 *     <li>Fit or validate training statistics before reading test data.</li>
 *     <li>Read the optional test dataset.</li>
 *     <li>Validate eager/lazy capability combinations.</li>
 *     <li>Apply prepared statistics.</li>
 *     <li>Reorder classification labels.</li>
 *     <li>Build eager missing-index metadata when requested.</li>
 *     <li>Publish the prepared datasets through {@link AppContext}.</li>
 * </ol>
 */
public final class DatasetPreparationCoordinator {

    /**
     * Reads and prepares training-mode datasets.
     *
     * @return prepared training data, optional testing data, and inferred
     *         dataset name
     */
    public PreparedDatasets prepareTrainingMode()
            throws IOException {
        StandardizationPipeline.prepareSuppliedStatistics();

        ListObjectDataset originalTrainingData =
                readTrainingData();

        StandardizationPipeline.prepareTrainingStatisticsBeforeTestRead(
                originalTrainingData
        );

        ListObjectDataset originalTestingData =
                AppContext.testing_file == null
                        ? null
                        : readTestData();

        validateDatasetCapabilities(
                originalTrainingData,
                originalTestingData
        );

        StandardizationPipeline.applyPreparedStatistics(
                originalTrainingData,
                originalTestingData
        );

        ListObjectDataset preparedTrainingData =
                prepareTrainingData(
                        originalTrainingData
                );

        ListObjectDataset preparedTestingData =
                originalTestingData == null
                        ? null
                        : prepareTestingData(
                                originalTestingData,
                                preparedTrainingData
                                        ._get_initial_class_labels()
                        );

        String datasetName =
                inferDatasetName(
                        AppContext.training_file
                );

        publishPreparedDatasets(
                preparedTrainingData,
                preparedTestingData,
                datasetName
        );

        return new PreparedDatasets(
                preparedTrainingData,
                preparedTestingData,
                datasetName
        );
    }

    /**
     * Reads and prepares evaluation data for a previously loaded model.
     *
     * <p>The supplied training data are retained exactly as loaded from the
     * model. Only the newly read testing data pass through evaluation-time
     * standardization and label remapping.</p>
     *
     * @param loadedTrainingData training data restored with the model
     * @return loaded training data, prepared testing data, and inferred
     *         dataset name
     */
    public PreparedDatasets prepareEvaluationMode(
            ListObjectDataset loadedTrainingData
    ) throws IOException {
        Objects.requireNonNull(
                loadedTrainingData,
                "Loaded training data cannot be null."
        );

        ListObjectDataset originalTestingData =
                readTestData();

        validateDatasetCapabilities(
                loadedTrainingData,
                originalTestingData
        );

        StandardizationPipeline.applyEvaluationStatistics(
                originalTestingData
        );

        ListObjectDataset preparedTestingData =
                prepareTestingData(
                        originalTestingData,
                        loadedTrainingData
                                ._get_initial_class_labels()
                );

        String datasetName =
                inferDatasetName(
                        AppContext.training_file
                );

        publishPreparedDatasets(
                loadedTrainingData,
                preparedTestingData,
                datasetName
        );

        return new PreparedDatasets(
                loadedTrainingData,
                preparedTestingData,
                datasetName
        );
    }

    /**
     * Reads the configured training dataset through the centralized reader
     * factory.
     */
    public ListObjectDataset readTrainingData()
            throws IOException {
        ReaderOptions options =
                buildBaseReaderOptions(
                        AppContext.getTrainingReaderType(),
                        AppContext.getTrainingFilePattern()
                )
                        .setDataPath(AppContext.training_file)
                        .setLabelPath(AppContext.training_labels)
                        .setTest(false);

        DatasetReader reader =
                DatasetReaderFactory.create(
                        options
                );

        return requireReadResult(
                reader.read(),
                "training"
        );
    }

    /**
     * Reads the configured testing dataset through the centralized reader
     * factory.
     */
    public ListObjectDataset readTestData()
            throws IOException {
        if (AppContext.testing_file == null
                || AppContext.testing_file.isBlank()) {
            throw new IllegalArgumentException(
                    "Testing data were requested, but no testing file was "
                            + "configured."
            );
        }

        ReaderOptions options =
                buildBaseReaderOptions(
                        AppContext.getTestingReaderType(),
                        AppContext.getTestingFilePattern()
                )
                        .setDataPath(AppContext.testing_file)
                        .setLabelPath(AppContext.testing_labels)
                        .setTest(true);

        DatasetReader reader =
                DatasetReaderFactory.create(
                        options
                );

        return requireReadResult(
                reader.read(),
                "testing"
        );
    }

    /**
     * Builds common reader configuration from the current application
     * settings.
     */
    public ReaderOptions buildBaseReaderOptions(
            ReaderType readerType,
            String filePattern
    ) {
        if (readerType == null) {
            throw new IllegalArgumentException(
                    "A reader type must be specified."
            );
        }

        return new ReaderOptions()
                .setReaderType(readerType)
                .setEntrySeparator(AppContext.entry_separator)
                .setArraySeparator(AppContext.array_separator)
                .setHasHeader(AppContext.csv_has_header)
                .set2D(AppContext.is2D)
                .setNumeric(AppContext.isNumeric)
                .setHasMissingValues(AppContext.hasMissingValues)
                .setTargetColumnIsFirst(
                        AppContext.target_column_is_first
                )
                .setRegression(AppContext.isRegressionMode())
                .setIdColumn(AppContext.id_column)
                .setTimeColumn(AppContext.time_column)
                .setFeatureColumns(AppContext.feature_columns)
                .setLabelColumns(AppContext.label_columns)
                .setHdf5DatasetPath(AppContext.hdf5_dataset_path)
                .setHdf5LabelDatasetPath(
                        AppContext.hdf5_label_dataset_path
                )
                .setFilePattern(filePattern)
                .setCustomReaderDescriptor(
                        AppContext.customReaderDescriptor
                )
                .setCustomReaderParameters(
                        AppContext.customReaderParameters
                )
                .setCustomReaderThreadSafe(
                        AppContext.customReaderThreadSafe
                )
                .setStandardizationStats(
                        AppContext.standardizationStats
                )
                .setNumericStorageType(
                        AppContext.numericStorageType
                );
    }

    /**
     * Applies classification-label normalization and eager missing-index
     * construction to training data.
     */
    public ListObjectDataset prepareTrainingData(
            ListObjectDataset original
    ) {
        Objects.requireNonNull(
                original,
                "Original training data cannot be null."
        );

        ListObjectDataset prepared =
                AppContext.isClassificationMode()
                        ? original.reorder_class_labels(null)
                        : original;

        attachMissingIndicesWhenRequired(
                prepared
        );

        return prepared;
    }

    /**
     * Applies training label-map normalization and eager missing-index
     * construction to testing data.
     */
    public ListObjectDataset prepareTestingData(
            ListObjectDataset original,
            Map<Object, Integer> labelMap
    ) {
        Objects.requireNonNull(
                original,
                "Original testing data cannot be null."
        );

        if (AppContext.isClassificationMode()
                && labelMap == null) {
            throw new IllegalArgumentException(
                    "Classification testing data require the training "
                            + "label map."
            );
        }

        ListObjectDataset prepared =
                AppContext.isClassificationMode()
                        ? original.reorder_class_labels(labelMap)
                        : original;

        attachMissingIndicesWhenRequired(
                prepared
        );

        return prepared;
    }

    /**
     * Validates workflow capabilities that depend on whether either dataset is
     * lazy.
     */
    public void validateDatasetCapabilities(
            ListObjectDataset trainingData,
            ListObjectDataset testingData
    ) {
        boolean lazyTraining =
                StandardizationPipeline.isLazyDataset(
                        trainingData
                );

        boolean lazyTesting =
                StandardizationPipeline.isLazyDataset(
                        testingData
                );

        if (!lazyTraining && !lazyTesting) {
            return;
        }

        if (AppContext.perform_train_imputation
                || AppContext.perform_test_imputation
                || AppContext.impute_train
                || AppContext.impute_test) {
            throw new UnsupportedOperationException(
                    "Imputation is not currently supported for lazy datasets. "
                            + "Disable perform_train_imputation, "
                            + "perform_test_imputation, impute_train, and "
                            + "impute_test when using a lazy reader."
            );
        }

        StandardizationConfig configuration =
                AppContext.standardizationConfig;

        if (lazyTraining
                && configuration != null
                && configuration.isEnabled()
                && AppContext.standardizationStats == null) {
            throw new UnsupportedOperationException(
                    "Lazy training with standardization requires precomputed "
                            + "statistics supplied through "
                            + "-standardization_stats. Automatic lazy "
                            + "statistics fitting is not implemented."
            );
        }

        if (lazyTraining != lazyTesting
                && trainingData != null
                && testingData != null) {
            System.out.println(
                    "Warning: training and testing datasets use different "
                            + "storage modes. One dataset is lazy and the "
                            + "other is materialized."
            );
        }
    }

    /**
     * Infers the logical dataset name from a configured training path.
     */
    public String inferDatasetName(
            String trainingFilePath
    ) {
        if (trainingFilePath == null
                || trainingFilePath.isBlank()) {
            return "dataset";
        }

        File trainingFile =
                new File(
                        trainingFilePath
                );

        String fileName =
                trainingFile.getName();

        String inferred =
                fileName.replaceFirst(
                        "(?i)_TRAIN\\.(txt|tsv|csv)$",
                        ""
                );

        if (inferred.isBlank()) {
            return "dataset";
        }

        return inferred;
    }

    private static void attachMissingIndicesWhenRequired(
            ListObjectDataset prepared
    ) {
        if (!AppContext.hasMissingValues
                || StandardizationPipeline.isLazyDataset(prepared)) {
            return;
        }

        prepared.setMissingIndices(
                MissingIndicesBuilder.buildFromDataset(
                        prepared.getData()
                )
        );
    }

    private static void publishPreparedDatasets(
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            String datasetName
    ) {
        AppContext.setTraining_data(
                trainingData
        );

        AppContext.setTesting_data(
                testingData
        );

        AppContext.setDatasetName(
                datasetName
        );
    }

    private static ListObjectDataset requireReadResult(
            ListObjectDataset dataset,
            String role
    ) {
        if (dataset == null) {
            throw new IllegalStateException(
                    "The "
                            + role
                            + " dataset reader returned null."
            );
        }

        if (dataset.size() == 0) {
            throw new IllegalArgumentException(
                    "The "
                            + role
                            + " dataset is empty."
            );
        }

        return dataset;
    }

    /**
     * Fully prepared datasets and their logical name.
     *
     * <p>The record does not clone either dataset. Callers receive the exact
     * prepared objects published to {@link AppContext}.</p>
     */
    public record PreparedDatasets(
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            String datasetName
    ) {
        public PreparedDatasets {
            Objects.requireNonNull(
                    trainingData,
                    "Prepared training data cannot be null."
            );

            if (datasetName == null || datasetName.isBlank()) {
                throw new IllegalArgumentException(
                        "Prepared dataset name cannot be null or blank."
                );
            }
        }

        public boolean hasTestingData() {
            return testingData != null;
        }
    }
}
