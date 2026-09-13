package core.experiment;

import core.AppContext;
import core.ProximityForestResult;
import datasets.ListObjectDataset;
import output.ExperimentResultWriter;
import output.PredictionWriter;
import util.GeneralUtilities;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Coordinates experiment output that is shared by training and evaluation
 * repetition workflows.
 *
 * <p>This class owns prediction decoding and writing, optional imputed-data
 * output, and aggregate experiment-result output. It does not train forests,
 * evaluate models, compute scores or proximity, or manage worker runtimes.</p>
 */
public final class ExperimentOutputCoordinator {

    private final ExperimentArtifactPaths artifactPaths;

    public ExperimentOutputCoordinator(
            ExperimentArtifactPaths artifactPaths
    ) {
        this.artifactPaths = Objects.requireNonNull(
                artifactPaths,
                "ExperimentArtifactPaths cannot be null."
        );
    }

    /**
     * Writes validation predictions when requested and records the artifact in
     * the repetition context.
     */
    public void writeValidationPredictionsWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context
    ) throws IOException {
        if (!AppContext.get_predictions) {
            return;
        }

        Path writtenPath = writePredictions(
                result,
                data,
                artifactPaths.validationPredictions(
                        context.getRepetition()
                )
        );

        context.addArtifact(
                "validationPredictions",
                artifactPaths.relativeArtifactPath(writtenPath)
        );
    }

    /**
     * Writes test predictions when requested and records the artifact in the
     * repetition context.
     */
    public void writeTestPredictionsWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context
    ) throws IOException {
        if (!AppContext.get_predictions) {
            return;
        }

        Path writtenPath = writePredictions(
                result,
                data,
                artifactPaths.testPredictions(
                        context.getRepetition()
                )
        );

        context.addArtifact(
                "testPredictions",
                artifactPaths.relativeArtifactPath(writtenPath)
        );
    }

    /**
     * Writes prepared training data when imputed training output was requested.
     *
     * <p>The destination behavior intentionally preserves the established
     * PFGAP output contract.</p>
     */
    public void writeTrainingDataWhenRequested(
            ListObjectDataset trainingData
    ) throws IOException {
        Objects.requireNonNull(
                trainingData,
                "Training data cannot be null."
        );

        if (!AppContext.impute_train) {
            return;
        }

        GeneralUtilities.writeDelimitedData(
                trainingData.getData(),
                AppContext.output_dir + AppContext.training_file,
                AppContext.array_separator,
                AppContext.entry_separator
        );
    }

    /**
     * Writes prepared testing data when imputed testing output was requested.
     *
     * <p>The destination behavior intentionally preserves the established
     * PFGAP output contract.</p>
     */
    public void writeTestingDataWhenRequested(
            ListObjectDataset testingData
    ) throws IOException {
        Objects.requireNonNull(
                testingData,
                "Testing data cannot be null."
        );

        if (!AppContext.impute_test) {
            return;
        }

        GeneralUtilities.writeDelimitedData(
                testingData.getData(),
                AppContext.output_dir + AppContext.testing_file,
                AppContext.array_separator,
                AppContext.entry_separator
        );
    }

    /**
     * Writes all accumulated experiment-result records when export is enabled.
     */
    public Path writeExperimentResults(
            ExperimentResultWriter resultWriter
    ) throws IOException {
        Objects.requireNonNull(
                resultWriter,
                "ExperimentResultWriter cannot be null."
        );

        if (AppContext.export_level < 1 || resultWriter.isEmpty()) {
            return null;
        }

        Path outputPath = resultWriter.writeToDirectory(
                artifactPaths.getOutputDirectory()
        );

        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Wrote experiment results to: " + outputPath
            );
        }

        return outputPath;
    }

    /**
     * Writes classification or regression predictions to an explicit path.
     */
    public Path writePredictions(
            ProximityForestResult result,
            ListObjectDataset data,
            Path outputPath
    ) throws IOException {
        Objects.requireNonNull(
                result,
                "ProximityForestResult cannot be null."
        );
        Objects.requireNonNull(
                data,
                "Prediction dataset cannot be null."
        );
        Objects.requireNonNull(
                outputPath,
                "Prediction output path cannot be null."
        );

        if (AppContext.isIsolationMode()) {
            throw new IllegalStateException(
                    "Prediction output is unavailable in isolation mode."
            );
        }

        if (result.Predictions == null) {
            throw new IllegalStateException(
                    "Prediction output was requested, but the forest result "
                            + "contains no predictions."
            );
        }

        if (AppContext.isRegressionMode()) {
            return PredictionWriter.writeRegression(
                    outputPath,
                    result.Predictions
            );
        }

        Map<Integer, Object> newToOriginal = data.invertLabelMap(
                data._get_initial_class_labels()
        );

        List<Object> originalPredictions = result.Predictions.stream()
                .map(newToOriginal::get)
                .toList();

        return PredictionWriter.writeClassification(
                outputPath,
                originalPredictions
        );
    }
}
