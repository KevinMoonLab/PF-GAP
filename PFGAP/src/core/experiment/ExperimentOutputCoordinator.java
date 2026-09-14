package core.experiment;

import core.AppContext;
import core.ForestPredictionResult;
import core.ProximityForestResult;
import datasets.ListObjectDataset;
import output.ExperimentResultWriter;
import output.PredictionWriter;
import util.GeneralUtilities;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;

/**
 * Coordinates experiment output shared by training and evaluation repetitions.
 *
 * <p>Legacy prediction artifacts and structured per-instance artifacts are
 * independent. A repetition may write ordinary predictions, enhanced prediction
 * details, OOD scores, or enhanced predictions and OOD scores together.</p>
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

    /** Writes requested validation prediction and structured artifacts. */
    public void writeValidationPredictionsWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context
    ) throws IOException {
        Objects.requireNonNull(context, "ExperimentRepetitionContext cannot be null.");
        Path basePath = artifactPaths.validationPredictions(
                context.getRepetition()
        );
        writeRequestedEvaluationArtifacts(
                result,
                data,
                context,
                basePath,
                artifactPaths.validationEnhancedOutput(context.getRepetition()),
                "validationPredictions",
                "validationEnhancedOutput"
        );
    }

    /** Writes requested test prediction and structured artifacts. */
    public void writeTestPredictionsWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context
    ) throws IOException {
        Objects.requireNonNull(context, "ExperimentRepetitionContext cannot be null.");
        Path basePath = artifactPaths.testPredictions(
                context.getRepetition()
        );
        writeRequestedEvaluationArtifacts(
                result,
                data,
                context,
                basePath,
                artifactPaths.testEnhancedOutput(context.getRepetition()),
                "testPredictions",
                "testEnhancedOutput"
        );
    }

    /**
     * Writes structured validation output independently of legacy predictions.
     *
     * <p>This method is required for OOD-only validation, where no prediction
     * artifact exists.</p>
     */
    public void writeValidationStructuredOutputWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context
    ) throws IOException {
        writeStructuredArtifactWhenRequested(
                result,
                data,
                context,
                artifactPaths.validationEnhancedOutput(
                        context.getRepetition()
                ),
                "validationEnhancedOutput"
        );
    }

    /** Writes structured test output independently of legacy predictions. */
    public void writeTestStructuredOutputWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context
    ) throws IOException {
        writeStructuredArtifactWhenRequested(
                result,
                data,
                context,
                artifactPaths.testEnhancedOutput(
                        context.getRepetition()
                ),
                "testEnhancedOutput"
        );
    }

    private void writeRequestedEvaluationArtifacts(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context,
            Path predictionPath,
            Path structuredOutputPath,
            String predictionArtifactName,
            String structuredArtifactName
    ) throws IOException {
        if (AppContext.get_predictions) {
            Path writtenPath = writePredictions(result, data, predictionPath);
            context.addArtifact(
                    predictionArtifactName,
                    artifactPaths.relativeArtifactPath(writtenPath)
            );
        }

        writeStructuredArtifactWhenRequested(
                result,
                data,
                context,
                structuredOutputPath,
                structuredArtifactName
        );
    }

    private void writeStructuredArtifactWhenRequested(
            ProximityForestResult result,
            ListObjectDataset data,
            ExperimentRepetitionContext context,
            Path outputPath,
            String artifactName
    ) throws IOException {
        if (!AppContext.shouldUseStructuredEvaluation()) {
            return;
        }
        Path writtenPath = writeStructuredResults(result, data, outputPath);
        context.addArtifact(
                artifactName,
                artifactPaths.relativeArtifactPath(writtenPath)
        );
    }

    /** Writes prepared training data when imputed output was requested. */
    public void writeTrainingDataWhenRequested(
            ListObjectDataset trainingData
    ) throws IOException {
        Objects.requireNonNull(trainingData, "Training data cannot be null.");
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

    /** Writes prepared testing data when imputed output was requested. */
    public void writeTestingDataWhenRequested(
            ListObjectDataset testingData
    ) throws IOException {
        Objects.requireNonNull(testingData, "Testing data cannot be null.");
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

    /** Writes all accumulated experiment records when export is enabled. */
    public Path writeExperimentResults(
            ExperimentResultWriter resultWriter
    ) throws IOException {
        Objects.requireNonNull(resultWriter, "ExperimentResultWriter cannot be null.");
        if (AppContext.export_level < 1 || resultWriter.isEmpty()) {
            return null;
        }
        Path outputPath = resultWriter.writeToDirectory(
                artifactPaths.getOutputDirectory()
        );
        if (AppContext.verbosity > 0) {
            System.out.println("Wrote experiment results to: " + outputPath);
        }
        return outputPath;
    }

    /** Writes classification or regression aggregate predictions. */
    public Path writePredictions(
            ProximityForestResult result,
            ListObjectDataset data,
            Path outputPath
    ) throws IOException {
        Objects.requireNonNull(result, "ProximityForestResult cannot be null.");
        Objects.requireNonNull(data, "Prediction dataset cannot be null.");
        Objects.requireNonNull(outputPath, "Prediction output path cannot be null.");

        if (AppContext.isIsolationMode()) {
            throw new IllegalStateException(
                    "Legacy prediction output is unavailable in isolation mode."
            );
        }
        if (result.Predictions == null || result.Predictions.isEmpty()) {
            throw new IllegalStateException(
                    "Prediction output was requested, but the result contains no predictions."
            );
        }
        if (result.Predictions.size() != data.size()) {
            throw new IllegalStateException(
                    "Prediction count " + result.Predictions.size()
                            + " does not match dataset size " + data.size() + "."
            );
        }

        if (AppContext.isRegressionMode()) {
            return PredictionWriter.writeRegression(outputPath, result.Predictions);
        }

        Map<Integer, Object> newToOriginal = data.invertLabelMap(
                data._get_initial_class_labels()
        );
        List<Object> originalPredictions = result.Predictions.stream()
                .map(newToOriginal::get)
                .toList();
        return PredictionWriter.writeClassification(outputPath, originalPredictions);
    }

    /**
     * Writes one CSV row per evaluated instance.
     *
     * <p>The stable schema supports prediction-only, OOD-only, and combined
     * output. Inapplicable values are written as empty fields. Classification
     * vote proportions are written as a deterministic semicolon-delimited map
     * inside one CSV field, avoiding a dataset-dependent column schema.</p>
     */
    public Path writeStructuredResults(
            ProximityForestResult result,
            ListObjectDataset data,
            Path outputPath
    ) throws IOException {
        Objects.requireNonNull(result, "ProximityForestResult cannot be null.");
        Objects.requireNonNull(data, "Evaluation dataset cannot be null.");
        Objects.requireNonNull(outputPath, "Structured output path cannot be null.");

        if (result.PredictionResults == null
                || result.PredictionResults.isEmpty()) {
            throw new IllegalStateException(
                    "Structured output was requested, but no structured results are available."
            );
        }
        if (result.PredictionResults.size() != data.size()) {
            throw new IllegalStateException(
                    "Structured result count " + result.PredictionResults.size()
                            + " does not match dataset size " + data.size() + "."
            );
        }

        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (BufferedWriter writer = Files.newBufferedWriter(
                outputPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE
        )) {
            writer.write(
                    "instance_index,prediction_kind,prediction,"
                            + "prediction_tree_count,prediction_mean,"
                            + "prediction_standard_deviation,class_vote_probabilities,"
                            + "ood_score_type,ood_mean,ood_standard_deviation,"
                            + "ood_available_tree_count,ood_total_tree_count"
            );
            writer.newLine();

            for (int index = 0; index < result.PredictionResults.size(); index++) {
                writeStructuredRow(
                        writer,
                        index,
                        result.PredictionResults.get(index),
                        data
                );
            }
        }
        return outputPath;
    }

    private static void writeStructuredRow(
            BufferedWriter writer,
            int instanceIndex,
            ForestPredictionResult output,
            ListObjectDataset data
    ) throws IOException {
        Objects.requireNonNull(
                output,
                "Structured result cannot be null at index " + instanceIndex + "."
        );

        Object displayedPrediction = output.hasPrediction()
                ? decodePrediction(output.prediction(), output.predictionKind(), data)
                : null;

        writer.write(Integer.toString(instanceIndex));
        writeField(writer, output.predictionKind().name());
        writeField(writer, displayedPrediction);
        writeField(writer, output.hasPrediction() ? output.predictionTreeCount() : null);
        writeFiniteField(writer, output.predictionMean());
        writeFiniteField(writer, output.predictionStandardDeviation());
        writeField(writer, encodeProbabilities(output.classVoteProbabilities(), data));
        writeField(writer, output.wasOODRequested() ? output.oodScoreType().configValue() : null);
        writeFiniteField(writer, output.oodMean());
        writeFiniteField(writer, output.oodStandardDeviation());
        writeField(writer, output.wasOODRequested() ? output.oodAvailableTreeCount() : null);
        writeField(writer, output.wasOODRequested() ? output.oodTotalTreeCount() : null);
        writer.newLine();
    }

    private static Object decodePrediction(
            Object prediction,
            ForestPredictionResult.PredictionKind kind,
            ListObjectDataset data
    ) {
        if (kind != ForestPredictionResult.PredictionKind.CLASSIFICATION) {
            return prediction;
        }
        Map<Integer, Object> labels = data.invertLabelMap(
                data._get_initial_class_labels()
        );
        return labels.getOrDefault(prediction, prediction);
    }

    private static String encodeProbabilities(
            Map<Object, Double> probabilities,
            ListObjectDataset data
    ) {
        if (probabilities == null || probabilities.isEmpty()) {
            return null;
        }
        Map<Integer, Object> labels = data.invertLabelMap(
                data._get_initial_class_labels()
        );
        StringJoiner encoded = new StringJoiner(";");
        for (Map.Entry<Object, Double> entry : probabilities.entrySet()) {
            Object label = labels.getOrDefault(entry.getKey(), entry.getKey());
            encoded.add(String.valueOf(label) + "=" + entry.getValue());
        }
        return encoded.toString();
    }

    private static void writeFiniteField(
            BufferedWriter writer,
            double value
    ) throws IOException {
        writeField(writer, Double.isFinite(value) ? value : null);
    }

    private static void writeField(
            BufferedWriter writer,
            Object value
    ) throws IOException {
        writer.write(',');
        if (value == null) {
            return;
        }
        String text = String.valueOf(value);
        boolean quote = text.indexOf(',') >= 0
                || text.indexOf('"') >= 0
                || text.indexOf('\n') >= 0
                || text.indexOf('\r') >= 0;
        if (!quote) {
            writer.write(text);
            return;
        }
        writer.write('"');
        writer.write(text.replace("\"", "\"\""));
        writer.write('"');
    }
}
