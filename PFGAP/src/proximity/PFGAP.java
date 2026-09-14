package proximity;

import core.AppContext;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import trees.ProximityForest;

import java.util.Objects;

/**
 * Public entry point for configured proximity computation.
 *
 * <p>This class resolves the configured proximity formula and delegates
 * context preparation, matrix construction, and scheduling to
 * {@link ProximityMatrixComputer}. It does not create executors, retain matrix
 * state, publish results globally, or convert compressed sparse matrices to
 * boxed collections.</p>
 *
 * <p>Matrix computation requires a caller-owned {@link ParallelRuntime} and
 * returns a typed {@link ProximityMatrixResult}. Callers should preserve the
 * selected dense or CSR representation through scoring, imputation, and
 * artifact generation.</p>
 */
public final class PFGAP {

    /**
     * Default absolute retention threshold for sparse matrix construction.
     *
     * <p>A final normalized proximity is retained when its absolute value is
     * strictly greater than this threshold. Zero therefore preserves every
     * mathematically nonzero proximity while omitting exact zeros.</p>
     */
    public static final double DEFAULT_SPARSE_RETENTION_THRESHOLD =
            0.0;

    private PFGAP() {
    }

    /** Resolves and validates the currently configured proximity measure. */
    public static ProximityMeasure configuredMeasure() {
        ProximityMeasure measure =
                ProximityMeasureResolver.resolve(
                        AppContext.proximityType
                );

        measure.validateContract();
        return measure;
    }

    /** Computes one directed train/train proximity from a prepared context. */
    public static double computeProximity(
            int sourceTrainIndex,
            int targetTrainIndex,
            ProximityContext context
    ) {
        Objects.requireNonNull(
                context,
                "ProximityContext cannot be null."
        );

        return configuredMeasure().computeTrainTrain(
                sourceTrainIndex,
                targetTrainIndex,
                context
        );
    }

    /** Computes one directed test/train proximity from a prepared context. */
    public static double computeTestTrainProximity(
            int testIndex,
            int trainIndex,
            ProximityContext context
    ) {
        Objects.requireNonNull(
                context,
                "ProximityContext cannot be null."
        );

        return configuredMeasure().computeTestTrain(
                testIndex,
                trainIndex,
                context
        );
    }

    /**
     * Computes the configured train/train proximity matrix.
     *
     * @return typed dense or compressed-sparse result
     */
    public static ProximityMatrixResult computeTrainProximities(
            ProximityForest forest,
            ListObjectDataset trainingData,
            ParallelRuntime runtime
    ) throws Exception {
        validateForestAndDataset(
                forest,
                trainingData,
                "trainingData"
        );

        Objects.requireNonNull(
                runtime,
                "ParallelRuntime cannot be null."
        );

        ProximityMeasure measure =
                configuredMeasure();

        if (AppContext.useSparseProximities) {
            CompressedSparseProximityMatrix sparse =
                    ProximityMatrixComputer.computeTrainSparse(
                            forest,
                            trainingData.size(),
                            measure,
                            DEFAULT_SPARSE_RETENTION_THRESHOLD,
                            runtime
                    );

            return new ProximityMatrixResult.Sparse(
                    sparse
            );
        }

        double[][] dense =
                ProximityMatrixComputer.computeTrainDense(
                        forest,
                        trainingData.size(),
                        measure,
                        runtime
                );

        return new ProximityMatrixResult.Dense(
                dense
        );
    }

    /**
     * Computes the configured test/train proximity matrix.
     *
     * @return typed dense or compressed-sparse result
     */
    public static ProximityMatrixResult computeTestTrainProximities(
            ProximityForest forest,
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            ParallelRuntime runtime
    ) throws Exception {
        validateForestAndDataset(
                forest,
                trainingData,
                "trainingData"
        );

        validateDataset(
                testingData,
                "testingData"
        );

        Objects.requireNonNull(
                runtime,
                "ParallelRuntime cannot be null."
        );

        ProximityMeasure measure =
                configuredMeasure();

        if (AppContext.useSparseProximities) {
            CompressedSparseProximityMatrix sparse =
                    ProximityMatrixComputer.computeTestTrainSparse(
                            forest,
                            testingData.size(),
                            trainingData.size(),
                            measure,
                            DEFAULT_SPARSE_RETENTION_THRESHOLD,
                            runtime
                    );

            return new ProximityMatrixResult.Sparse(
                    sparse
            );
        }

        double[][] dense =
                ProximityMatrixComputer.computeTestTrainDense(
                        forest,
                        testingData.size(),
                        trainingData.size(),
                        measure,
                        runtime
                );

        return new ProximityMatrixResult.Dense(
                dense
        );
    }

    private static void validateForestAndDataset(
            ProximityForest forest,
            ListObjectDataset dataset,
            String datasetName
    ) {
        Objects.requireNonNull(
                forest,
                "ProximityForest cannot be null."
        );

        if (forest.getTrees() == null) {
            throw new IllegalStateException(
                    "ProximityForest has no tree array."
            );
        }

        validateDataset(
                dataset,
                datasetName
        );
    }

    private static void validateDataset(
            ListObjectDataset dataset,
            String datasetName
    ) {
        if (dataset == null) {
            throw new IllegalArgumentException(
                    datasetName
                            + " cannot be null."
            );
        }
    }
}
