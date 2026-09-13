package core.experiment;

import core.AppContext;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import output.ProximityWriter;
import proximity.CompressedSparseProximityMatrix;
import proximity.PFGAP;
import proximity.ProximityMatrixResult;
import trees.ProximityForest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Coordinates proximity computation, typed in-memory result ownership,
 * artifact output, and cleanup for one experiment repetition.
 *
 * <p>Each matrix domain is stored as one {@link ProximityMatrixResult}. Dense
 * results retain their {@code double[][]}; sparse results retain their
 * {@link CompressedSparseProximityMatrix}. Sparse results are never expanded
 * into boxed maps.</p>
 *
 * <p>All proximity work uses the repetition-owned {@link ParallelRuntime}.
 * This class creates no executors and owns neither proximity formulas nor
 * matrix-level scheduling.</p>
 */
public final class ExperimentProximityCoordinator {

    private final ExperimentArtifactPaths artifactPaths;
    private final ParallelRuntime parallelRuntime;

    private ProximityMatrixResult trainingProximities;
    private ProximityMatrixResult testTrainProximities;

    public ExperimentProximityCoordinator(
            ExperimentArtifactPaths artifactPaths,
            ParallelRuntime parallelRuntime
    ) {
        this.artifactPaths = Objects.requireNonNull(
                artifactPaths,
                "ExperimentArtifactPaths cannot be null."
        );

        this.parallelRuntime = Objects.requireNonNull(
                parallelRuntime,
                "ParallelRuntime cannot be null."
        );
    }

    /** Returns whether a training proximity result is available. */
    public boolean trainingProximitiesAvailable() {
        return trainingProximities != null;
    }

    /** Returns whether a test/train proximity result is available. */
    public boolean testTrainProximitiesAvailable() {
        return testTrainProximities != null;
    }

    /** Returns the typed training result or throws if it is unavailable. */
    public ProximityMatrixResult requireTrainingProximities() {
        if (trainingProximities == null) {
            throw new IllegalStateException(
                    "Training proximities are not available."
            );
        }

        return trainingProximities;
    }

    /** Returns the typed test/train result or throws if it is unavailable. */
    public ProximityMatrixResult requireTestTrainProximities() {
        if (testTrainProximities == null) {
            throw new IllegalStateException(
                    "Test/train proximities are not available."
            );
        }

        return testTrainProximities;
    }

    /** Returns the dense training matrix or throws if unavailable or sparse. */
    public double[][] requireTrainingDense() {
        return requireTrainingProximities()
                .requireDense()
                .values();
    }

    /** Returns the sparse training matrix or throws if unavailable or dense. */
    public CompressedSparseProximityMatrix requireTrainingSparse() {
        return requireTrainingProximities()
                .requireSparse()
                .values();
    }

    /** Returns the dense test/train matrix or throws if unavailable or sparse. */
    public double[][] requireTestTrainDense() {
        return requireTestTrainProximities()
                .requireDense()
                .values();
    }

    /** Returns the sparse test/train matrix or throws if unavailable or dense. */
    public CompressedSparseProximityMatrix requireTestTrainSparse() {
        return requireTestTrainProximities()
                .requireSparse()
                .values();
    }

    /**
     * Computes training proximities if absent and returns the typed result.
     */
    public ProximityMatrixResult ensureTrainingProximities(
            ProximityForest forest,
            ListObjectDataset trainingData
    ) throws Exception {
        Objects.requireNonNull(
                forest,
                "ProximityForest cannot be null."
        );
        Objects.requireNonNull(
                trainingData,
                "Training data cannot be null."
        );

        if (trainingProximities != null) {
            validateConfiguredRepresentation(
                    trainingProximities,
                    "training"
            );
            return trainingProximities;
        }

        logStart("Training");
        long startNanoseconds = System.nanoTime();

        ProximityMatrixResult result =
                PFGAP.computeTrainProximities(
                        forest,
                        trainingData,
                        parallelRuntime
                );

        validateConfiguredRepresentation(
                result,
                "training"
        );

        trainingProximities = result;

        logFinished(
                "Training",
                startNanoseconds
        );

        return result;
    }

    /**
     * Computes test/train proximities if absent and returns the typed result.
     */
    public ProximityMatrixResult ensureTestTrainProximities(
            ProximityForest forest,
            ListObjectDataset testingData,
            ListObjectDataset trainingData
    ) throws Exception {
        Objects.requireNonNull(
                forest,
                "ProximityForest cannot be null."
        );
        Objects.requireNonNull(
                testingData,
                "Testing data cannot be null."
        );
        Objects.requireNonNull(
                trainingData,
                "Training data cannot be null."
        );

        if (testTrainProximities != null) {
            validateConfiguredRepresentation(
                    testTrainProximities,
                    "test/train"
            );
            return testTrainProximities;
        }

        logStart("Test/Train");
        long startNanoseconds = System.nanoTime();

        ProximityMatrixResult result =
                PFGAP.computeTestTrainProximities(
                        forest,
                        testingData,
                        trainingData,
                        parallelRuntime
                );

        validateConfiguredRepresentation(
                result,
                "test/train"
        );

        testTrainProximities = result;

        logFinished(
                "Test/Train",
                startNanoseconds
        );

        return result;
    }

    /** Computes and writes all requested training-related proximity artifacts. */
    public Map<String, Path> computeRequestedTrainingArtifacts(
            ProximityForest forest,
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            int repetition
    ) throws Exception {
        Map<String, Path> artifacts =
                new LinkedHashMap<>();

        if (!AppContext.getprox) {
            return artifacts;
        }

        ensureTrainingProximities(
                forest,
                trainingData
        );

        artifacts.put(
                "trainingProximities",
                writeTrainingProximities(
                        repetition
                )
        );

        if (testingData != null) {
            artifacts.put(
                    "testTrainProximities",
                    computeAndWriteTestTrainProximities(
                            forest,
                            testingData,
                            trainingData,
                            repetition
                    )
            );
        }

        return artifacts;
    }

    /** Computes and writes one test/train proximity artifact. */
    public Path computeAndWriteTestTrainProximities(
            ProximityForest forest,
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            int repetition
    ) throws Exception {
        ensureTestTrainProximities(
                forest,
                testingData,
                trainingData
        );

        return writeTestTrainProximities(
                repetition
        );
    }

    /** Records relative artifact paths in one repetition context. */
    public void recordArtifacts(
            Map<String, Path> artifacts,
            ExperimentRepetitionContext context
    ) {
        Objects.requireNonNull(
                context,
                "ExperimentRepetitionContext cannot be null."
        );

        if (artifacts == null || artifacts.isEmpty()) {
            return;
        }

        for (Map.Entry<String, Path> entry : artifacts.entrySet()) {
            context.addArtifact(
                    entry.getKey(),
                    artifactPaths.relativeArtifactPath(
                            entry.getValue()
                    )
            );
        }
    }

    /** Clears both training and test/train results. */
    public void clearAllResults() {
        trainingProximities = null;
        testTrainProximities = null;
    }

    /** Clears only the training result. */
    public void clearTrainingResults() {
        trainingProximities = null;
    }

    /** Clears only the test/train result. */
    public void clearTestTrainResults() {
        testTrainProximities = null;
    }

    private Path writeTrainingProximities(
            int repetition
    ) throws IOException {
        ProximityMatrixResult result =
                requireTrainingProximities();

        if (result instanceof ProximityMatrixResult.Sparse sparse) {
            return ProximityWriter.writeSparseMatrix(
                    artifactPaths.trainingProximitiesSparse(
                            repetition
                    ),
                    sparse.values()
            );
        }

        ProximityMatrixResult.Dense dense =
                result.requireDense();

        return ProximityWriter.writeDenseCsv(
                artifactPaths.trainingProximitiesDense(
                        repetition
                ),
                dense.values()
        );
    }

    private Path writeTestTrainProximities(
            int repetition
    ) throws IOException {
        ProximityMatrixResult result =
                requireTestTrainProximities();

        if (result instanceof ProximityMatrixResult.Sparse sparse) {
            return ProximityWriter.writeSparseMatrix(
                    artifactPaths.testTrainProximitiesSparse(
                            repetition
                    ),
                    sparse.values()
            );
        }

        ProximityMatrixResult.Dense dense =
                result.requireDense();

        return ProximityWriter.writeDenseCsv(
                artifactPaths.testTrainProximitiesDense(
                        repetition
                ),
                dense.values()
        );
    }

    private static void validateConfiguredRepresentation(
            ProximityMatrixResult result,
            String matrixName
    ) {
        Objects.requireNonNull(
                result,
                matrixName
                        + " proximity result cannot be null."
        );

        boolean expectedSparse =
                AppContext.useSparseProximities;

        if (expectedSparse == result.isSparse()) {
            return;
        }

        throw new IllegalStateException(
                "Configured "
                        + matrixName
                        + " proximity representation requires "
                        + (expectedSparse ? "sparse CSR" : "dense array")
                        + ", but computation returned "
                        + (result.isSparse() ? "sparse CSR" : "dense array")
                        + "."
        );
    }

    private static void logStart(
            String matrixName
    ) {
        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Computing "
                            + matrixName
                            + " Proximities..."
            );
        }
    }

    private static void logFinished(
            String matrixName,
            long startNanoseconds
    ) {
        if (AppContext.verbosity > 0) {
            long elapsedMilliseconds =
                    (System.nanoTime() - startNanoseconds)
                            / 1_000_000L;

            System.out.println(
                    "Done Computing "
                            + matrixName
                            + " Proximities. Computation time: "
                            + elapsedMilliseconds
                            + "ms"
            );
        }
    }
}
