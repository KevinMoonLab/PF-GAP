package imputation;

import core.AppContext;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import distance.MEASURE;
import imputation.update.DTWPFImpute;
import imputation.update.PFImpute;
import proximity.ProximityMatrixResult;
import trees.ProximityForest;

import java.util.Objects;

/**
 * Orchestrates iterative proximity-based numeric imputation.
 *
 * <p>This class owns imputation strategy sequencing only. Forest training,
 * proximity computation, and value updates are delegated to their respective
 * components, all using the caller-owned {@link ParallelRuntime}.</p>
 *
 * <p>Every imputation iteration explicitly invalidates the previously cached
 * proximity result, computes a fresh typed {@link ProximityMatrixResult}, uses
 * that result for one update, and invalidates it again after the dataset has
 * changed. No proximity matrix is read from global application state.</p>
 */
public final class ProximityImputation {

    public static final String IMPUTE_FIRST =
            "impute_first";

    public static final String PROXIMITY_FIRST =
            "proximity_first";

    public static final String GAP_STANDARD =
            "standard";

    public static final String GAP_DTW_ALIGNMENT =
            "dtw_alignment";

    private static final int DEFAULT_DTW_WINDOW_SIZE =
            -1;

    private ProximityImputation() {
    }

    /** Computes and returns a train/train proximity result. */
    @FunctionalInterface
    public interface TrainProximityComputer {
        ProximityMatrixResult compute(
                ProximityForest forest,
                ListObjectDataset trainingData
        ) throws Exception;
    }

    /** Computes and returns a test/train proximity result. */
    @FunctionalInterface
    public interface TestTrainProximityComputer {
        ProximityMatrixResult compute(
                ProximityForest forest,
                ListObjectDataset testingData,
                ListObjectDataset trainingData
        ) throws Exception;
    }

    /**
     * Performs configured iterative training-data imputation.
     *
     * @param trainingData mutable training dataset
     * @param repetition repetition index used for temporary forests
     * @param proximityComputer typed train/train proximity provider
     * @param proximityInvalidator clears a cached train/train result
     * @param runtime repetition-owned parallel runtime
     */
    public static void imputeTraining(
            ListObjectDataset trainingData,
            int repetition,
            TrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        if (!AppContext.perform_train_imputation) {
            return;
        }

        if (!shouldRunNumericImputation(trainingData)) {
            return;
        }

        Objects.requireNonNull(
                proximityComputer,
                "Train proximity computer cannot be null."
        );
        Objects.requireNonNull(
                proximityInvalidator,
                "Train proximity invalidator cannot be null."
        );
        Objects.requireNonNull(
                runtime,
                "ParallelRuntime cannot be null."
        );

        switch (getInitializationStrategy()) {
            case IMPUTE_FIRST -> imputeTrainingImputeFirst(
                    trainingData,
                    repetition,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
            case PROXIMITY_FIRST -> imputeTrainingProximityFirst(
                    trainingData,
                    repetition,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
            default -> throw new IllegalArgumentException(
                    "Unknown imputation initialization strategy: "
                            + getInitializationStrategy()
            );
        }
    }

    /**
     * Performs configured iterative test-data imputation.
     *
     * @param testingData mutable testing dataset
     * @param trainingData training reference dataset
     * @param trainedForest final trained forest
     * @param proximityComputer typed test/train proximity provider
     * @param proximityInvalidator clears a cached test/train result
     * @param runtime repetition-owned parallel runtime
     */
    public static void imputeTesting(
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            ProximityForest trainedForest,
            TestTrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        if (!AppContext.perform_test_imputation) {
            return;
        }

        if (!shouldRunNumericImputation(testingData)) {
            return;
        }

        Objects.requireNonNull(
                trainingData,
                "Training data cannot be null."
        );
        Objects.requireNonNull(
                trainedForest,
                "Trained forest cannot be null."
        );
        Objects.requireNonNull(
                proximityComputer,
                "Test/train proximity computer cannot be null."
        );
        Objects.requireNonNull(
                proximityInvalidator,
                "Test/train proximity invalidator cannot be null."
        );
        Objects.requireNonNull(
                runtime,
                "ParallelRuntime cannot be null."
        );

        switch (getInitializationStrategy()) {
            case IMPUTE_FIRST -> imputeTestingImputeFirst(
                    testingData,
                    trainingData,
                    trainedForest,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
            case PROXIMITY_FIRST -> imputeTestingProximityFirst(
                    testingData,
                    trainingData,
                    trainedForest,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
            default -> throw new IllegalArgumentException(
                    "Unknown imputation initialization strategy: "
                            + getInitializationStrategy()
            );
        }
    }

    private static void imputeTrainingImputeFirst(
            ListObjectDataset trainingData,
            int repetition,
            TrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        log("Imputing the training set using impute-first strategy...");
        log("Performing initial imputation...");

        AppContext.initial_imputer.Impute(
                trainingData
        );

        for (int iteration = 0;
             iteration < AppContext.numImputes;
             iteration++) {

            logIteration(
                    "Training",
                    iteration,
                    "using normal model distances..."
            );

            ProximityForest forest = new ProximityForest(
                    repetition,
                    AppContext.userdistances
            );

            forest.train(
                    trainingData,
                    runtime
            );

            updateTrainingIteration(
                    trainingData,
                    forest,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
        }

        log("Done imputing the training set.");
    }

    private static void imputeTrainingProximityFirst(
            ListObjectDataset trainingData,
            int repetition,
            TrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        requirePositiveIterationsForProximityFirst();
        log("Imputing the training set using proximity-first strategy...");

        for (int iteration = 0;
             iteration < AppContext.numImputes;
             iteration++) {

            boolean firstPass = iteration == 0;
            MEASURE[] distances = firstPass
                    ? getMissingProximityDistances()
                    : AppContext.userdistances;

            logIteration(
                    "Training",
                    iteration,
                    firstPass
                            ? "using missing-compatible proximity distances..."
                            : "using normal model distances..."
            );

            ProximityForest forest = new ProximityForest(
                    repetition,
                    distances
            );

            forest.train(
                    trainingData,
                    runtime
            );

            updateTrainingIteration(
                    trainingData,
                    forest,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
        }

        log("Done imputing the training set.");
    }

    private static void imputeTestingImputeFirst(
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            ProximityForest trainedForest,
            TestTrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        log("Imputing the testing set using impute-first strategy...");
        log("Performing initial imputation...");

        AppContext.initial_imputer.Impute(
                testingData
        );

        for (int iteration = 0;
             iteration < AppContext.numImputes;
             iteration++) {

            logIteration(
                    "Testing",
                    iteration,
                    "using normal trained forest..."
            );

            updateTestingIteration(
                    testingData,
                    trainingData,
                    trainedForest,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
        }

        log("Done imputing the testing set.");
    }

    private static void imputeTestingProximityFirst(
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            ProximityForest trainedForest,
            TestTrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        requirePositiveIterationsForProximityFirst();
        log("Imputing the testing set using proximity-first strategy...");

        for (int iteration = 0;
             iteration < AppContext.numImputes;
             iteration++) {

            boolean firstPass = iteration == 0;
            ProximityForest forestForProximities;

            if (firstPass) {
                logIteration(
                        "Testing",
                        iteration,
                        "using missing-compatible proximity distances..."
                );

                forestForProximities = new ProximityForest(
                        0,
                        getMissingProximityDistances()
                );

                forestForProximities.train(
                        trainingData,
                        runtime
                );
            } else {
                logIteration(
                        "Testing",
                        iteration,
                        "using normal trained forest..."
                );

                forestForProximities = trainedForest;
            }

            updateTestingIteration(
                    testingData,
                    trainingData,
                    forestForProximities,
                    proximityComputer,
                    proximityInvalidator,
                    runtime
            );
        }

        log("Done imputing the testing set.");
    }

    private static void updateTrainingIteration(
            ListObjectDataset trainingData,
            ProximityForest forest,
            TrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        proximityInvalidator.run();

        ProximityMatrixResult proximities =
                proximityComputer.compute(
                        forest,
                        trainingData
                );

        try {
            if (usesDTWAlignmentUpdate()) {
                DTWPFImpute.trainNumericImpute(
                        trainingData,
                        proximities,
                        runtime,
                        DEFAULT_DTW_WINDOW_SIZE
                );
            } else {
                PFImpute.trainNumericImpute(
                        trainingData,
                        proximities,
                        runtime
                );
            }
        } finally {
            /*
             * The dataset may now differ from the one represented by this
             * matrix, so the cached result must never escape the iteration.
             */
            proximityInvalidator.run();
        }
    }

    private static void updateTestingIteration(
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            ProximityForest forest,
            TestTrainProximityComputer proximityComputer,
            Runnable proximityInvalidator,
            ParallelRuntime runtime
    ) throws Exception {
        proximityInvalidator.run();

        ProximityMatrixResult proximities =
                proximityComputer.compute(
                        forest,
                        testingData,
                        trainingData
                );

        try {
            if (usesDTWAlignmentUpdate()) {
                DTWPFImpute.testNumericImpute(
                        testingData,
                        trainingData,
                        proximities,
                        runtime,
                        DEFAULT_DTW_WINDOW_SIZE
                );
            } else {
                PFImpute.testNumericImpute(
                        testingData,
                        trainingData,
                        proximities,
                        runtime
                );
            }
        } finally {
            proximityInvalidator.run();
        }
    }

    private static boolean shouldRunNumericImputation(
            ListObjectDataset data
    ) {
        return data != null
                && AppContext.hasMissingValues
                && AppContext.isNumeric
                && data.getMissingIndices() != null;
    }

    private static String getInitializationStrategy() {
        String strategy =
                AppContext.imputation_initialization_strategy;

        if (strategy == null || strategy.isBlank()) {
            return IMPUTE_FIRST;
        }

        return strategy.trim().toLowerCase();
    }

    private static boolean usesDTWAlignmentUpdate() {
        String strategy =
                AppContext.gap_update_strategy;

        if (strategy == null || strategy.isBlank()) {
            return AppContext.DTWImpute;
        }

        return GAP_DTW_ALIGNMENT.equals(
                strategy.trim().toLowerCase()
        );
    }

    private static MEASURE[] getMissingProximityDistances() {
        if (AppContext.missing_proximity_distances != null
                && AppContext.missing_proximity_distances.length > 0) {
            return AppContext.missing_proximity_distances;
        }

        return AppContext.is2D
                ? new MEASURE[]{MEASURE.nan_euclidean_i}
                : new MEASURE[]{MEASURE.nan_euclidean};
    }

    private static void requirePositiveIterationsForProximityFirst() {
        if (AppContext.numImputes <= 0) {
            throw new IllegalArgumentException(
                    "proximity_first imputation requires "
                            + "AppContext.numImputes > 0."
            );
        }
    }

    private static void logIteration(
            String domain,
            int iteration,
            String detail
    ) {
        log(
                domain
                        + " imputation iteration "
                        + (iteration + 1)
                        + " of "
                        + AppContext.numImputes
                        + " "
                        + detail
        );
    }

    private static void log(
            String message
    ) {
        if (AppContext.verbosity > 0) {
            System.out.println(message);
        }
    }

    public static String missingProximityDistancesToString() {
        MEASURE[] distances =
                getMissingProximityDistances();

        StringBuilder builder =
                new StringBuilder();

        for (int index = 0;
             index < distances.length;
             index++) {
            if (index > 0) {
                builder.append(',');
            }

            builder.append(
                    distances[index]
            );
        }

        return builder.toString();
    }
}
