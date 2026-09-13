package core.experiment;

import core.AppContext;
import core.AppContextSnapshot;
import core.AppContextUtils;
import core.ModelIO;
import core.ProximityForestResult;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import imputation.ProximityImputation;
import output.ExperimentResultRecord;
import output.ExperimentResultWriter;
import trees.ProximityForest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Executes one complete training-mode experiment repetition.
 *
 * <p>The runner coordinates training, optional validation, model persistence,
 * requested score and proximity artifacts, and result-record assembly. Dataset
 * reading and preparation remain the responsibility of
 * {@link DatasetPreparationCoordinator}.</p>
 *
 * <p>Each invocation receives an {@link ExperimentRepetitionContext} that owns
 * exactly one repetition's parallel runtime and accumulated artifact metadata.
 * Proximity and scoring coordinators are created inside {@link #run} so their
 * runtime and mutable matrix state cannot leak across repetitions.</p>
 */
public final class TrainingRepetitionRunner {

    private final ExperimentArtifactPaths artifactPaths;
    private final ExperimentResultAssembler resultAssembler;
    private final ExperimentOutputCoordinator outputCoordinator;

    public TrainingRepetitionRunner(
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

    /** Runs one training repetition and appends its immutable result record. */
    public void run(
            DatasetPreparationCoordinator.PreparedDatasets datasets,
            String datasetName,
            ExperimentRepetitionContext context,
            ExperimentResultWriter resultWriter
    ) throws Exception {
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

        ListObjectDataset trainingData = datasets.trainingData();
        ListObjectDataset testingData = datasets.testingData();

        String effectiveDatasetName = normalizeDatasetName(
                datasetName,
                datasets.datasetName()
        );

        int repetition = context.getRepetition();

        performTrainingImputationWhenRequested(
                trainingData,
                repetition,
                proximityCoordinator,
                context.getParallelRuntime()
        );

        /*
         * Proximities computed during iterative imputation are intermediate.
         * Final requested matrices must be computed from the final forest.
         */
        proximityCoordinator.clearAllResults();

        ProximityForest forest = new ProximityForest(
                repetition,
                AppContext.userdistances
        );

        forest.train(
                trainingData,
                context.getParallelRuntime()
        );

        saveModelWhenRequested(
                forest,
                trainingData,
                repetition,
                context
        );

        outputCoordinator.writeTrainingDataWhenRequested(
                trainingData
        );

        ProximityForestResult result = testingData == null
                ? null
                : runValidation(
                        forest,
                        trainingData,
                        testingData,
                        effectiveDatasetName,
                        context,
                        scoringCoordinator,
                        proximityCoordinator
                );

        handleTrainingScores(
                forest,
                trainingData,
                context,
                scoringCoordinator,
                proximityCoordinator
        );

        Map<String, Path> proximityArtifacts =
                proximityCoordinator.computeRequestedTrainingArtifacts(
                        forest,
                        trainingData,
                        testingData,
                        repetition
                );

        proximityCoordinator.recordArtifacts(
                proximityArtifacts,
                context
        );

        if (result == null) {
            result = forest.getResultSet();
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

    private void performTrainingImputationWhenRequested(
            ListObjectDataset trainingData,
            int repetition,
            ExperimentProximityCoordinator proximityCoordinator,
            ParallelRuntime parallelRuntime
    ) throws Exception {
        if (!AppContext.perform_train_imputation) {
            return;
        }

        ProximityImputation.imputeTraining(
                trainingData,
                repetition,
                proximityCoordinator::ensureTrainingProximities,
                proximityCoordinator::clearTrainingResults,
                parallelRuntime
        );
    }

    private ProximityForestResult runValidation(
            ProximityForest forest,
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            String datasetName,
            ExperimentRepetitionContext context,
            ExperimentScoringCoordinator scoringCoordinator,
            ExperimentProximityCoordinator proximityCoordinator
    ) throws Exception {
        performTestingImputationWhenRequested(
                testingData,
                trainingData,
                forest,
                proximityCoordinator,
                context.getParallelRuntime()
        );

        /*
         * Test/train proximities produced by imputation are intermediate and
         * must not satisfy a later final-artifact request accidentally.
         */
        proximityCoordinator.clearTestTrainResults();

        int repetition = context.getRepetition();

        if (AppContext.isIsolationMode()) {
            ExperimentScoringCoordinator.ScoreArtifact scoreArtifact =
                    scoringCoordinator.computeValidationIsolationScores(
                            forest,
                            testingData,
                            trainingData.size(),
                            repetition
                    );

            scoringCoordinator.recordArtifact(
                    "validationOutlierScores",
                    scoreArtifact,
                    "validationIsolationScore",
                    "validationIsolationScoringMilliseconds",
                    context
            );

            return forest.getResultSet();
        }

        outputCoordinator.writeTestingDataWhenRequested(
                testingData
        );

        ProximityForestResult result = forest.test(
                testingData,
                context.getParallelRuntime()
        );

        outputCoordinator.writeValidationPredictionsWhenRequested(
                result,
                testingData,
                context
        );

        result.printResults(
                datasetName,
                repetition,
                ""
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

    private void handleTrainingScores(
            ProximityForest forest,
            ListObjectDataset trainingData,
            ExperimentRepetitionContext context,
            ExperimentScoringCoordinator scoringCoordinator,
            ExperimentProximityCoordinator proximityCoordinator
    ) throws Exception {
        ExperimentScoringCoordinator.ScoreArtifact scoreArtifact =
                scoringCoordinator.computeTrainingScores(
                        forest,
                        trainingData,
                        context.getRepetition(),
                        proximityCoordinator
                );

        if (scoreArtifact == null) {
            return;
        }

        boolean isolation = AppContext.isIsolationMode();

        scoringCoordinator.recordArtifact(
                "trainingOutlierScores",
                scoreArtifact,
                isolation
                        ? "trainingIsolationScore"
                        : "trainingProximityOutlierScore",
                isolation
                        ? "trainingIsolationScoringMilliseconds"
                        : "trainingProximityOutlierScoringMilliseconds",
                context
        );
    }

    private void saveModelWhenRequested(
            ProximityForest forest,
            ListObjectDataset trainingData,
            int repetition,
            ExperimentRepetitionContext context
    ) throws IOException {
        if (!AppContext.savemodel) {
            return;
        }

        AppContextSnapshot snapshot = AppContextUtils.captureSnapshot();

        Path modelPath = artifactPaths.resolveRepeated(
                AppContext.modelname + ".ser",
                repetition
        );

        ModelIO.saveModel(
                modelPath.toString(),
                forest,
                trainingData,
                snapshot
        );

        context.addArtifact(
                "model",
                artifactPaths.relativeArtifactPath(modelPath)
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
