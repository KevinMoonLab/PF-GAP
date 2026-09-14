package core.experiment;

import core.AppContext;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import outlier.IsolationDepthScorer;
import outlier.OutlierScorer;
import output.OutlierScoreWriter;
import trees.ProximityForest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Coordinates experiment outlier-score computation, output, and summary
 * metadata.
 *
 * <p>This class keeps two mathematically distinct scoring workflows separate:</p>
 *
 * <ul>
 *     <li>Isolation-mode scores use terminal-node depth and are delegated to
 *         {@link IsolationDepthScorer}.</li>
 *     <li>Classification proximity outlier scores use a dense or CSR training
 *         proximity matrix and are delegated to {@link OutlierScorer}.</li>
 * </ul>
 *
 * <p>The coordinator does not compute proximity matrices or create executors.
 * Proximities are owned by {@link ExperimentProximityCoordinator}, and all
 * parallel scoring uses the repetition-owned {@link ParallelRuntime}.</p>
 */
public final class ExperimentScoringCoordinator {

    private final ExperimentArtifactPaths artifactPaths;
    private final ParallelRuntime parallelRuntime;

    public ExperimentScoringCoordinator(
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

    /**
     * Computes and writes requested training outlier scores.
     *
     * <p>Isolation mode never requests a proximity matrix. Classification mode
     * ensures that the configured training proximity representation exists and
     * passes it directly to the corresponding dense or CSR scorer. Regression
     * currently has no proximity-based training outlier score.</p>
     *
     * @param forest trained forest
     * @param trainingData prepared training data
     * @param repetition zero-based repetition index
     * @param proximityCoordinator owner and provider of training proximities
     * @return score artifact, or null when scores are not requested or not
     *         defined for the configured task
     */
    public ScoreArtifact computeTrainingScores(
            ProximityForest forest,
            ListObjectDataset trainingData,
            int repetition,
            ExperimentProximityCoordinator proximityCoordinator
    ) throws Exception {
        Objects.requireNonNull(
                forest,
                "ProximityForest cannot be null."
        );

        Objects.requireNonNull(
                trainingData,
                "Training data cannot be null."
        );

        if (!AppContext.get_training_outlier_scores) {
            return null;
        }

        if (AppContext.isIsolationMode()) {
            return computeTrainingIsolationScores(
                    forest,
                    trainingData,
                    repetition
            );
        }

        if (AppContext.isRegressionMode()) {
            return null;
        }

        Objects.requireNonNull(
                proximityCoordinator,
                "ExperimentProximityCoordinator cannot be null for "
                        + "proximity-based classification outlier scoring."
        );

        proximityCoordinator.ensureTrainingProximities(
                forest,
                trainingData
        );

        return computeTrainingProximityOutlierScores(
                trainingData,
                repetition,
                proximityCoordinator
        );
    }

    /** Computes and writes validation isolation scores. */
    public ScoreArtifact computeValidationIsolationScores(
            ProximityForest forest,
            ListObjectDataset data,
            int normalizationSampleSize,
            int repetition
    ) throws Exception {
        return computeInferenceIsolationScores(
                forest,
                data,
                normalizationSampleSize,
                artifactPaths.validationOutlierScores(repetition)
        );
    }

    /** Computes and writes test isolation scores. */
    public ScoreArtifact computeTestIsolationScores(
            ProximityForest forest,
            ListObjectDataset data,
            int normalizationSampleSize,
            int repetition
    ) throws Exception {
        return computeInferenceIsolationScores(
                forest,
                data,
                normalizationSampleSize,
                artifactPaths.testOutlierScores(repetition)
        );
    }

    /** Adds one score artifact's metrics and timing to a repetition context. */
    public void addArtifactResults(
            ScoreArtifact scoreArtifact,
            String metricPrefix,
            String timingName,
            ExperimentRepetitionContext context
    ) {
        if (scoreArtifact == null) {
            return;
        }

        Objects.requireNonNull(
                context,
                "ExperimentRepetitionContext cannot be null."
        );

        requireName(metricPrefix, "Metric prefix");
        requireName(timingName, "Timing name");

        NumericScoreSummary summary = scoreArtifact.summary();

        context.addCount(metricPrefix + "Count", summary.count());
        context.addCount(
                metricPrefix + "NonfiniteCount",
                summary.nonfiniteCount()
        );
        context.addTimingMilliseconds(
                timingName,
                scoreArtifact.computationMilliseconds()
        );

        if (summary.count() == 0L) {
            return;
        }

        context.addMetric(metricPrefix + "Mean", summary.mean());
        context.addMetric(
                metricPrefix + "PopulationStandardDeviation",
                summary.populationStandardDeviation()
        );
        context.addMetric(metricPrefix + "Minimum", summary.minimum());
        context.addMetric(metricPrefix + "Maximum", summary.maximum());
    }

    /** Adds an artifact path and its numerical summary to one repetition. */
    public void recordArtifact(
            String artifactName,
            ScoreArtifact scoreArtifact,
            String metricPrefix,
            String timingName,
            ExperimentRepetitionContext context
    ) {
        if (scoreArtifact == null) {
            return;
        }

        Objects.requireNonNull(
                context,
                "ExperimentRepetitionContext cannot be null."
        );

        context.addArtifact(
                artifactName,
                artifactPaths.relativeArtifactPath(scoreArtifact.path())
        );

        addArtifactResults(
                scoreArtifact,
                metricPrefix,
                timingName,
                context
        );
    }

    private ScoreArtifact computeTrainingProximityOutlierScores(
            ListObjectDataset trainingData,
            int repetition,
            ExperimentProximityCoordinator proximityCoordinator
    ) throws Exception {
        log("Computing Training Proximity Outlier Scores...");
        long start = System.nanoTime();

        Object[] labels = trainingData._internal_class_array();
        double[] scores;

        if (AppContext.useSparseProximities) {
            scores = OutlierScorer.scoreSparse(
                    labels,
                    proximityCoordinator.requireTrainingSparse(),
                    false,
                    parallelRuntime
            );
        } else {
            scores = OutlierScorer.scoreDense(
                    labels,
                    proximityCoordinator.requireTrainingDense(),
                    false,
                    parallelRuntime
            );
        }

        long end = System.nanoTime();

        return writeScoreArtifact(
                scores,
                start,
                end,
                artifactPaths.trainingOutlierScores(repetition)
        );
    }

    /**
     * Computes training isolation scores from stored tree topology. This path
     * does not request or consume a proximity matrix.
     */
    private ScoreArtifact computeTrainingIsolationScores(
            ProximityForest forest,
            ListObjectDataset trainingData,
            int repetition
    ) throws IOException {
        log(
                "Computing Training Isolation Scores from stored tree topology..."
        );

        long start = System.nanoTime();
        double[] scores = IsolationDepthScorer.scoreTraining(
                forest,
                trainingData
        );
        long end = System.nanoTime();

        return writeScoreArtifact(
                scores,
                start,
                end,
                artifactPaths.trainingOutlierScores(repetition)
        );
    }

    /**
     * Computes isolation scores for unseen data by routing each instance
     * through every tree. This path remains distinct from proximity scoring.
     */
    private ScoreArtifact computeInferenceIsolationScores(
            ProximityForest forest,
            ListObjectDataset data,
            int normalizationSampleSize,
            Path outputPath
    ) throws Exception {
        Objects.requireNonNull(forest, "ProximityForest cannot be null.");
        Objects.requireNonNull(data, "Inference data cannot be null.");
        Objects.requireNonNull(
                outputPath,
                "Isolation-score output path cannot be null."
        );

        if (normalizationSampleSize < 1) {
            throw new IllegalArgumentException(
                    "Isolation-score normalization sample size must be "
                            + "positive. Received: "
                            + normalizationSampleSize
                            + "."
            );
        }

        log(
                "Computing Inference Isolation Scores by routing instances "
                        + "through the forest..."
        );

        long start = System.nanoTime();
        double[] scores = IsolationDepthScorer.scoreInference(
                forest,
                data,
                normalizationSampleSize
        );
        long end = System.nanoTime();

        return writeScoreArtifact(
                scores,
                start,
                end,
                outputPath
        );
    }

    private static ScoreArtifact writeScoreArtifact(
            double[] scores,
            long computationStart,
            long computationEnd,
            Path outputPath
    ) throws IOException {
        Objects.requireNonNull(scores, "Score array cannot be null.");
        Objects.requireNonNull(outputPath, "Score output path cannot be null.");

        if (computationEnd < computationStart) {
            throw new IllegalArgumentException(
                    "Computation end time cannot precede its start time."
            );
        }

        Path writtenPath = OutlierScoreWriter.write(
                outputPath,
                scores
        );

        return new ScoreArtifact(
                writtenPath,
                summarizeScores(scores),
                (computationEnd - computationStart) / 1_000_000.0
        );
    }

    /** Computes stable summary statistics while excluding nonfinite values. */
    public static NumericScoreSummary summarizeScores(
            double[] scores
    ) {
        if (scores == null || scores.length == 0) {
            return emptySummary(0L);
        }

        long count = 0L;
        long nonfiniteCount = 0L;
        double mean = 0.0;
        double m2 = 0.0;
        double minimum = Double.POSITIVE_INFINITY;
        double maximum = Double.NEGATIVE_INFINITY;

        for (double score : scores) {
            if (!Double.isFinite(score)) {
                nonfiniteCount++;
                continue;
            }

            count++;
            double delta = score - mean;
            mean += delta / count;
            double updatedDelta = score - mean;
            m2 += delta * updatedDelta;
            minimum = Math.min(minimum, score);
            maximum = Math.max(maximum, score);
        }

        if (count == 0L) {
            return emptySummary(nonfiniteCount);
        }

        double populationVariance = Math.max(0.0, m2) / count;

        return new NumericScoreSummary(
                count,
                nonfiniteCount,
                mean,
                Math.sqrt(populationVariance),
                minimum,
                maximum
        );
    }

    private static NumericScoreSummary emptySummary(
            long nonfiniteCount
    ) {
        return new NumericScoreSummary(
                0L,
                nonfiniteCount,
                Double.NaN,
                Double.NaN,
                Double.NaN,
                Double.NaN
        );
    }

    private static void requireName(
            String value,
            String description
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    description + " cannot be null or blank."
            );
        }
    }

    private static void log(
            String message
    ) {
        if (AppContext.verbosity > 0) {
            System.out.println(message);
        }
    }

    /** Numerical summary of finite score values. */
    public record NumericScoreSummary(
            long count,
            long nonfiniteCount,
            double mean,
            double populationStandardDeviation,
            double minimum,
            double maximum
    ) {
        public NumericScoreSummary {
            if (count < 0L) {
                throw new IllegalArgumentException(
                        "Score count cannot be negative."
                );
            }
            if (nonfiniteCount < 0L) {
                throw new IllegalArgumentException(
                        "Nonfinite score count cannot be negative."
                );
            }
        }
    }

    /** Written score artifact and its computation metadata. */
    public record ScoreArtifact(
            Path path,
            NumericScoreSummary summary,
            double computationMilliseconds
    ) {
        public ScoreArtifact {
            Objects.requireNonNull(
                    path,
                    "Score artifact path cannot be null."
            );
            Objects.requireNonNull(
                    summary,
                    "Score artifact summary cannot be null."
            );
            if (!Double.isFinite(computationMilliseconds)
                    || computationMilliseconds < 0.0) {

                throw new IllegalArgumentException(
                        "Score computation time must be finite and "
                                + "nonnegative. Received: "
                                + computationMilliseconds
                                + "."
                );
            }
        }
    }
}
