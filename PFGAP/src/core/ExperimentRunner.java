package core;

import core.experiment.DatasetPreparationCoordinator;
import core.experiment.EvaluationRepetitionRunner;
import core.experiment.ExperimentArtifactPaths;
import core.experiment.ExperimentOutputCoordinator;
import core.experiment.ExperimentRepetitionContext;
import core.experiment.ExperimentResultAssembler;
import core.experiment.TrainingRepetitionRunner;
import output.ExperimentResultWriter;
import util.PrintUtilities;

/**
 * Top-level coordinator for PFGAP training and saved-model evaluation.
 *
 * <p>Dataset preparation, repetition execution, artifact paths, output, and
 * result assembly are delegated to application-scoped collaborators. Each
 * repetition owns an independent parallel runtime through
 * {@link ExperimentRepetitionContext}. Repetition-scoped scoring and proximity
 * coordinators are created by the training and evaluation repetition runners.</p>
 */
public final class ExperimentRunner {

    public ExperimentRunner() {
    }

    public void run(
            boolean evaluationMode
    ) throws Exception {
        ExperimentServices services =
                createServices();

        if (evaluationMode) {
            runEvaluationMode(
                    services
            );
        } else {
            runTrainingMode(
                    services
            );
        }
    }

    private void runTrainingMode(
            ExperimentServices services
    ) throws Exception {
        DatasetPreparationCoordinator.PreparedDatasets datasets =
                services.datasetPreparation().prepareTrainingMode();

        System.gc();

        if (AppContext.shuffle_dataset) {
            System.out.println(
                    "Shuffling the training set..."
            );

            datasets.trainingData().shuffle();
        }

        ExperimentResultWriter experimentResults =
                new ExperimentResultWriter();

        for (int repetition = 0;
             repetition < AppContext.num_repeats;
             repetition++) {

            printRepetitionHeader(
                    repetition,
                    datasets.datasetName()
            );

            try (ExperimentRepetitionContext context =
                         new ExperimentRepetitionContext(
                                 repetition,
                                 AppContext.num_workers
                         )) {

                services.trainingRunner().run(
                        datasets,
                        datasets.datasetName(),
                        context,
                        experimentResults
                );
            }

            requestGarbageCollectionWhenConfigured();
        }

        services.outputCoordinator().writeExperimentResults(
                experimentResults
        );
    }

    private void runEvaluationMode(
            ExperimentServices services
    ) throws Exception {
        ModelIO.LoadedModel loaded =
                ModelIO.loadModel(
                        AppContext.modelname
                                + ".ser"
                );

        ModelIO.applySnapshot(
                loaded.snapshot
        );

        DatasetPreparationCoordinator.PreparedDatasets datasets =
                services.datasetPreparation().prepareEvaluationMode(
                        loaded.trainData
                );

        System.gc();

        ExperimentResultWriter experimentResults =
                new ExperimentResultWriter();

        /*
         * A loaded model contains one already trained forest. Evaluation
         * repetitions reuse that forest, but each repetition receives an
         * independent worker-pool lifecycle, proximity state, scoring state,
         * and result metadata.
         */
        for (int repetition = 0;
             repetition < AppContext.num_repeats;
             repetition++) {

            printRepetitionHeader(
                    repetition,
                    datasets.datasetName()
            );

            try (ExperimentRepetitionContext context =
                         new ExperimentRepetitionContext(
                                 repetition,
                                 AppContext.num_workers
                         )) {

                services.evaluationRunner().run(
                        loaded.forest,
                        datasets,
                        datasets.datasetName(),
                        context,
                        experimentResults
                );
            }

            requestGarbageCollectionWhenConfigured();
        }

        services.outputCoordinator().writeExperimentResults(
                experimentResults
        );
    }

    private static ExperimentServices createServices() {
        ExperimentArtifactPaths artifactPaths =
                new ExperimentArtifactPaths(
                        AppContext.output_dir,
                        AppContext.num_repeats
                );

        DatasetPreparationCoordinator datasetPreparation =
                new DatasetPreparationCoordinator();

        ExperimentResultAssembler resultAssembler =
                new ExperimentResultAssembler();

        ExperimentOutputCoordinator outputCoordinator =
                new ExperimentOutputCoordinator(
                        artifactPaths
                );

        TrainingRepetitionRunner trainingRunner =
                new TrainingRepetitionRunner(
                        artifactPaths,
                        resultAssembler,
                        outputCoordinator
                );

        EvaluationRepetitionRunner evaluationRunner =
                new EvaluationRepetitionRunner(
                        artifactPaths,
                        resultAssembler,
                        outputCoordinator
                );

        return new ExperimentServices(
                datasetPreparation,
                outputCoordinator,
                trainingRunner,
                evaluationRunner
        );
    }

    private static void requestGarbageCollectionWhenConfigured() {
        if (AppContext.garbage_collect_after_each_repetition) {
            System.gc();
        }
    }

    private static void printRepetitionHeader(
            int repetition,
            String datasetName
    ) {
        if (AppContext.verbosity > 0) {
            System.out.println(
                    "-----------------Repetition No: "
                            + (repetition + 1)
                            + " ("
                            + datasetName
                            + ")   -----------------"
            );

            PrintUtilities.printMemoryUsage();
            return;
        }

        if (AppContext.verbosity == 0 && repetition == 0) {
            System.out.println(
                    "Repetition, Dataset, Score, TrainingTime(ms), "
                            + "TestingTime(ms), MeanDepthPerTree"
            );
        }
    }

    private record ExperimentServices(
            DatasetPreparationCoordinator datasetPreparation,
            ExperimentOutputCoordinator outputCoordinator,
            TrainingRepetitionRunner trainingRunner,
            EvaluationRepetitionRunner evaluationRunner
    ) {
    }
}
