package core.experiment;

import core.AppContext;
import core.ProximityForestResult;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import imputation.ProximityImputation;
import output.ExperimentResultRecord;
import output.ExperimentResultWriter;
import trees.ProximityForest;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Executes one complete evaluation-mode experiment repetition using a
 * previously trained forest.
 *
 * <p>The runner coordinates optional test imputation, classification or
 * regression prediction, isolation scoring, requested test/train proximity
 * output, and result-record assembly. Model loading and dataset preparation
 * remain responsibilities of the top-level experiment workflow.</p>
 *
 * <p>Each invocation receives an {@link ExperimentRepetitionContext} that owns
 * exactly one repetition's parallel runtime and artifact metadata. Proximity
 * and scoring coordinators are created inside {@link #run} so their runtime
 * and mutable result state cannot leak across evaluation repetitions.</p>
 *
 * <p>The supplied trained forest may be reused across evaluation repetitions.
 * Its trained topology is not modified here, although prediction and
 * proximity preparation may update or inspect transient test-leaf membership
 * according to the forest evaluation contract.</p>
 */
public final class EvaluationRepetitionRunner {

    private final ExperimentArtifactPaths artifactPaths;
    private final ExperimentResultAssembler resultAssembler;
    private final ExperimentOutputCoordinator outputCoordinator;

    public EvaluationRepetitionRunner(
            ExperimentArtifactPaths artifactPaths,
            ExperimentResultAssembler resultAssembler,
            ExperimentOutputCoordinator outputCoordinator
    ) {
        this.artifactPaths = Objects.requireNonNull(
                artifactPaths,
                "ExperimentArtifactPaths cannot be null."
        );

        this.resultAssembler = Objects.requireNonNull(
                resultAssembler,
                "ExperimentResultAssembler cannot be null."
        );

        this.outputCoordinator = Objects.requireNonNull(
                outputCoordinator,
                "ExperimentOutputCoordinator cannot be null."
        );
    }

    /** Runs one evaluation repetition and appends its immutable result record. */
    public void run(
            ProximityForest forest,
            DatasetPreparationCoordinator.PreparedDatasets datasets,
            String datasetName,
            ExperimentRepetitionContext context,
            ExperimentResultWriter resultWriter
    ) throws Exception {
        Objects.requireNonNull(
                forest,
                "ProximityForest cannot be null."
        );
        Objects.requireNonNull(
                datasets,
                "PreparedDatasets cannot be null."
        );
        Objects.requireNonNull(
                context,
                "ExperimentRepetitionContext cannot be null."
        );
        Objects.requireNonNull(
                resultWriter,
                "ExperimentResultWriter cannot be null."
        );

        ExperimentProximityCoordinator proximityCoordinator =
                new ExperimentProximityCoordinator(
                        artifactPaths,
                        context.getParallelRuntime()
                );

        ExperimentScoringCoordinator scoringCoordinator =
                new ExperimentScoringCoordinator(
                        artifactPaths,
                        context.getParallelRuntime()
                );

        ListObjectDataset trainingData = Objects.requireNonNull(
                datasets.trainingData(),
                "Evaluation mode requires loaded training data."
        );

        ListObjectDataset testingData = Objects.requireNonNull(
                datasets.testingData(),
                "Evaluation mode requires prepared testing data."
        );

        String effectiveDatasetName = normalizeDatasetName(
                datasetName,
                datasets.datasetName()
        );

        int repetition = context.getRepetition();

        performTestingImputationWhenRequested(
                testingData,
                trainingData,
                forest,
                proximityCoordinator,
                context.getParallelRuntime()
        );

        /*
         * Test/train proximities produced during iterative imputation are
         * intermediate. A final requested artifact must be computed from the
         * final imputed testing data.
         */
        proximityCoordinator.clearTestTrainResults();

        ProximityForestResult result;

        if (AppContext.isIsolationMode()) {
            result = runIsolationEvaluation(
                    forest,
                    trainingData,
                    testingData,
                    repetition,
                    context,
                    scoringCoordinator
            );
        } else {
            result = runPredictiveEvaluation(
                    forest,
                    trainingData,
                    testingData,
                    effectiveDatasetName,
                    repetition,
                    context,
                    proximityCoordinator
            );
        }

        ExperimentResultRecord record = resultAssembler.assemble(
                result,
                effectiveDatasetName,
                trainingData,
                testingData,
                context
        );

        resultWriter.add(record);
    }

    private ProximityForestResult runIsolationEvaluation(
            ProximityForest forest,
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            int repetition,
            ExperimentRepetitionContext context,
            ExperimentScoringCoordinator scoringCoordinator
    ) throws Exception {
        ExperimentScoringCoordinator.ScoreArtifact scoreArtifact =
                scoringCoordinator.computeTestIsolationScores(
                        forest,
                        testingData,
                        trainingData.size(),
                        repetition
                );

        scoringCoordinator.recordArtifact(
                "testOutlierScores",
                scoreArtifact,
                "testIsolationScore",
                "testIsolationScoringMilliseconds",
                context
        );

        return forest.getResultSet();
    }

    private ProximityForestResult runPredictiveEvaluation(
            ProximityForest forest,
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            String datasetName,
            int repetition,
            ExperimentRepetitionContext context,
            ExperimentProximityCoordinator proximityCoordinator
    ) throws Exception {
        ProximityForestResult result = forest.test(
                testingData,
                context.getParallelRuntime()
        );

        if (!AppContext.perform_test_imputation) {
            result.printResults(
                    datasetName,
                    repetition,
                    ""
            );
        }

        outputCoordinator.writeTestingDataWhenRequested(
                testingData
        );

        outputCoordinator.writeTestPredictionsWhenRequested(
                result,
                testingData,
                context
        );

        writeTestTrainProximitiesWhenRequested(
                forest,
                testingData,
                trainingData,
                repetition,
                context,
                proximityCoordinator
        );

        return result;
    }

    private void performTestingImputationWhenRequested(
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            ProximityForest forest,
            ExperimentProximityCoordinator proximityCoordinator,
            ParallelRuntime parallelRuntime
    ) throws Exception {
        if (!AppContext.perform_test_imputation) {
            return;
        }

        ProximityImputation.imputeTesting(
                testingData,
                trainingData,
                forest,
                proximityCoordinator::ensureTestTrainProximities,
                proximityCoordinator::clearTestTrainResults,
                parallelRuntime
        );
    }

    /** Computes and writes a requested test/train proximity artifact. */
    private void writeTestTrainProximitiesWhenRequested(
            ProximityForest forest,
            ListObjectDataset testingData,
            ListObjectDataset trainingData,
            int repetition,
            ExperimentRepetitionContext context,
            ExperimentProximityCoordinator proximityCoordinator
    ) throws Exception {
        if (!AppContext.getprox) {
            return;
        }

        Path proximityPath =
                proximityCoordinator.computeAndWriteTestTrainProximities(
                        forest,
                        testingData,
                        trainingData,
                        repetition
                );

        context.addArtifact(
                "testTrainProximities",
                artifactPaths.relativeArtifactPath(
                        proximityPath
                )
        );
    }

    private static String normalizeDatasetName(
            String suppliedName,
            String preparedName
    ) {
        if (suppliedName != null && !suppliedName.isBlank()) {
            return suppliedName.trim();
        }

        if (preparedName != null && !preparedName.isBlank()) {
            return preparedName.trim();
        }

        return "dataset";
    }
}
