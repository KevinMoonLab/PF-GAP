package core;

import ood.OODScoreType;

import java.io.Serial;
import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable per-instance output from proximity-forest evaluation.
 *
 * <p>Prediction and out-of-distribution output are independently optional. A
 * caller may request prediction only, OOD scoring only, or both in one forest
 * traversal. The explicit availability methods distinguish an omitted output
 * from a legitimate null prediction or a valid numeric value of zero.</p>
 *
 * <p>Classification probabilities are forest vote proportions rather than
 * calibrated posterior probabilities. Regression mean and standard deviation
 * summarize the finite numeric predictions contributed by individual trees;
 * the standard deviation is the population standard deviation. OOD mean and
 * standard deviation likewise summarize available tree-level OOD scores.</p>
 */
public final class ForestPredictionResult implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Describes the kind of prediction details stored in this result. */
    public enum PredictionKind {
        NONE,
        CLASSIFICATION,
        REGRESSION,
        ISOLATION
    }

    private final PredictionKind predictionKind;
    private final boolean predictionAvailable;
    private final Object prediction;
    private final int predictionTreeCount;

    private final double predictionMean;
    private final double predictionStandardDeviation;
    private final Map<Object, Double> classVoteProbabilities;

    private final boolean oodAvailable;
    private final OODScoreType oodScoreType;
    private final double oodMean;
    private final double oodStandardDeviation;
    private final int oodAvailableTreeCount;
    private final int oodTotalTreeCount;

    /** Creates a prediction-only classification result. */
    public static ForestPredictionResult classification(
            Object prediction,
            Map<Object, Double> classVoteProbabilities,
            int predictionTreeCount
    ) {
        return classificationWithOOD(
                prediction,
                classVoteProbabilities,
                predictionTreeCount,
                null,
                Double.NaN,
                Double.NaN,
                0,
                0
        );
    }

    /** Creates a classification result with optional forest-level OOD output. */
    public static ForestPredictionResult classificationWithOOD(
            Object prediction,
            Map<Object, Double> classVoteProbabilities,
            int predictionTreeCount,
            OODScoreType oodScoreType,
            double oodMean,
            double oodStandardDeviation,
            int oodAvailableTreeCount,
            int oodTotalTreeCount
    ) {
        return new ForestPredictionResult(
                PredictionKind.CLASSIFICATION,
                true,
                prediction,
                predictionTreeCount,
                Double.NaN,
                Double.NaN,
                classVoteProbabilities,
                oodScoreType,
                oodMean,
                oodStandardDeviation,
                oodAvailableTreeCount,
                oodTotalTreeCount
        );
    }

    /** Creates a prediction-only regression result. */
    public static ForestPredictionResult regression(
            Object prediction,
            double predictionMean,
            double predictionStandardDeviation,
            int predictionTreeCount
    ) {
        return regressionWithOOD(
                prediction,
                predictionMean,
                predictionStandardDeviation,
                predictionTreeCount,
                null,
                Double.NaN,
                Double.NaN,
                0,
                0
        );
    }

    /** Creates a regression result with optional forest-level OOD output. */
    public static ForestPredictionResult regressionWithOOD(
            Object prediction,
            double predictionMean,
            double predictionStandardDeviation,
            int predictionTreeCount,
            OODScoreType oodScoreType,
            double oodMean,
            double oodStandardDeviation,
            int oodAvailableTreeCount,
            int oodTotalTreeCount
    ) {
        return new ForestPredictionResult(
                PredictionKind.REGRESSION,
                true,
                prediction,
                predictionTreeCount,
                predictionMean,
                predictionStandardDeviation,
                Map.of(),
                oodScoreType,
                oodMean,
                oodStandardDeviation,
                oodAvailableTreeCount,
                oodTotalTreeCount
        );
    }

    /** Creates a prediction-only isolation result. */
    public static ForestPredictionResult isolation(
            Object prediction,
            int predictionTreeCount
    ) {
        return isolationWithOOD(
                prediction,
                predictionTreeCount,
                null,
                Double.NaN,
                Double.NaN,
                0,
                0
        );
    }

    /** Creates an isolation result with optional forest-level OOD output. */
    public static ForestPredictionResult isolationWithOOD(
            Object prediction,
            int predictionTreeCount,
            OODScoreType oodScoreType,
            double oodMean,
            double oodStandardDeviation,
            int oodAvailableTreeCount,
            int oodTotalTreeCount
    ) {
        return new ForestPredictionResult(
                PredictionKind.ISOLATION,
                true,
                prediction,
                predictionTreeCount,
                Double.NaN,
                Double.NaN,
                Map.of(),
                oodScoreType,
                oodMean,
                oodStandardDeviation,
                oodAvailableTreeCount,
                oodTotalTreeCount
        );
    }

    /** Creates an OOD-only result with no prediction output. */
    public static ForestPredictionResult oodOnly(
            OODScoreType oodScoreType,
            double oodMean,
            double oodStandardDeviation,
            int oodAvailableTreeCount,
            int oodTotalTreeCount
    ) {
        return new ForestPredictionResult(
                PredictionKind.NONE,
                false,
                null,
                0,
                Double.NaN,
                Double.NaN,
                Map.of(),
                oodScoreType,
                oodMean,
                oodStandardDeviation,
                oodAvailableTreeCount,
                oodTotalTreeCount
        );
    }

    private ForestPredictionResult(
            PredictionKind predictionKind,
            boolean predictionAvailable,
            Object prediction,
            int predictionTreeCount,
            double predictionMean,
            double predictionStandardDeviation,
            Map<Object, Double> classVoteProbabilities,
            OODScoreType oodScoreType,
            double oodMean,
            double oodStandardDeviation,
            int oodAvailableTreeCount,
            int oodTotalTreeCount
    ) {
        this.predictionKind = Objects.requireNonNull(
                predictionKind,
                "Prediction kind cannot be null."
        );
        validatePrediction(
                predictionKind,
                predictionAvailable,
                predictionTreeCount,
                predictionMean,
                predictionStandardDeviation,
                classVoteProbabilities
        );
        validateOOD(
                oodScoreType,
                oodMean,
                oodStandardDeviation,
                oodAvailableTreeCount,
                oodTotalTreeCount
        );

        this.predictionAvailable = predictionAvailable;
        this.prediction = prediction;
        this.predictionTreeCount = predictionTreeCount;
        this.predictionMean = canonicalizeZero(predictionMean);
        this.predictionStandardDeviation =
                canonicalizeZero(predictionStandardDeviation);
        this.classVoteProbabilities = immutableProbabilities(
                classVoteProbabilities
        );

        this.oodAvailable = oodAvailableTreeCount > 0;
        this.oodScoreType = oodScoreType;
        this.oodMean = canonicalizeZero(oodMean);
        this.oodStandardDeviation = canonicalizeZero(oodStandardDeviation);
        this.oodAvailableTreeCount = oodAvailableTreeCount;
        this.oodTotalTreeCount = oodTotalTreeCount;
    }

    public PredictionKind predictionKind() {
        return predictionKind;
    }

    public boolean hasPrediction() {
        return predictionAvailable;
    }

    /**
     * Returns the aggregate forest prediction. This may legitimately be null,
     * particularly for an isolation-mode result; use {@link #hasPrediction()}
     * to determine whether prediction output was requested.
     */
    public Object prediction() {
        return prediction;
    }

    public int predictionTreeCount() {
        return predictionTreeCount;
    }

    /** Returns the regression tree-prediction mean, or NaN when inapplicable. */
    public double predictionMean() {
        return predictionMean;
    }

    /**
     * Returns the population standard deviation of regression tree predictions,
     * or NaN when inapplicable.
     */
    public double predictionStandardDeviation() {
        return predictionStandardDeviation;
    }

    /**
     * Returns immutable classification vote proportions in deterministic
     * insertion order, or an empty map for nonclassification results.
     */
    public Map<Object, Double> classVoteProbabilities() {
        return classVoteProbabilities;
    }

    public boolean hasOODScore() {
        return oodAvailable;
    }

    /** Returns the requested OOD method, or null when OOD was not requested. */
    public OODScoreType oodScoreType() {
        return oodScoreType;
    }

    /** Returns the forest OOD mean, or NaN when unavailable or not requested. */
    public double oodMean() {
        return oodMean;
    }

    /**
     * Returns the population standard deviation of available tree OOD scores,
     * or NaN when unavailable or not requested.
     */
    public double oodStandardDeviation() {
        return oodStandardDeviation;
    }

    public int oodAvailableTreeCount() {
        return oodAvailableTreeCount;
    }

    public int oodTotalTreeCount() {
        return oodTotalTreeCount;
    }

    /** Returns whether OOD evaluation was requested, even if unavailable. */
    public boolean wasOODRequested() {
        return oodScoreType != null;
    }

    private static void validatePrediction(
            PredictionKind kind,
            boolean available,
            int treeCount,
            double mean,
            double standardDeviation,
            Map<Object, Double> probabilities
    ) {
        Objects.requireNonNull(
                probabilities,
                "Class vote probabilities cannot be null."
        );
        if (treeCount < 0) {
            throw new IllegalArgumentException(
                    "Prediction tree count cannot be negative."
            );
        }
        if (!available) {
            if (kind != PredictionKind.NONE || treeCount != 0
                    || !Double.isNaN(mean)
                    || !Double.isNaN(standardDeviation)
                    || !probabilities.isEmpty()) {
                throw new IllegalArgumentException(
                        "An omitted prediction must use kind NONE, zero trees, "
                                + "NaN numeric statistics, and no probabilities."
                );
            }
            return;
        }
        if (kind == PredictionKind.NONE || treeCount == 0) {
            throw new IllegalArgumentException(
                    "An available prediction requires a prediction kind and at least one tree."
            );
        }

        if (kind == PredictionKind.REGRESSION) {
            if (!Double.isFinite(mean)
                    || !Double.isFinite(standardDeviation)
                    || standardDeviation < 0.0
                    || !probabilities.isEmpty()) {
                throw new IllegalArgumentException(
                        "Regression output requires finite mean and nonnegative "
                                + "standard deviation, with no class probabilities."
                );
            }
            return;
        }

        if (!Double.isNaN(mean) || !Double.isNaN(standardDeviation)) {
            throw new IllegalArgumentException(
                    "Nonregression output must use NaN numeric prediction statistics."
            );
        }
        if (kind == PredictionKind.CLASSIFICATION) {
            validateProbabilities(probabilities);
        } else if (!probabilities.isEmpty()) {
            throw new IllegalArgumentException(
                    "Only classification output may contain class vote probabilities."
            );
        }
    }

    private static void validateOOD(
            OODScoreType scoreType,
            double mean,
            double standardDeviation,
            int availableTreeCount,
            int totalTreeCount
    ) {
        if (availableTreeCount < 0 || totalTreeCount < 0
                || availableTreeCount > totalTreeCount) {
            throw new IllegalArgumentException(
                    "Invalid available and total OOD tree counts."
            );
        }
        if (scoreType == null) {
            if (availableTreeCount != 0 || totalTreeCount != 0
                    || !Double.isNaN(mean)
                    || !Double.isNaN(standardDeviation)) {
                throw new IllegalArgumentException(
                        "OOD output that was not requested must use zero counts and NaN statistics."
                );
            }
            return;
        }
        if (availableTreeCount == 0) {
            if (!Double.isNaN(mean) || !Double.isNaN(standardDeviation)) {
                throw new IllegalArgumentException(
                        "Unavailable requested OOD output must use NaN statistics."
                );
            }
            return;
        }
        if (!Double.isFinite(mean) || mean < 0.0
                || !Double.isFinite(standardDeviation)
                || standardDeviation < 0.0) {
            throw new IllegalArgumentException(
                    "Available OOD mean and standard deviation must be finite and nonnegative."
            );
        }
    }

    private static void validateProbabilities(Map<Object, Double> probabilities) {
        if (probabilities.isEmpty()) {
            throw new IllegalArgumentException(
                    "Classification output requires at least one class vote probability."
            );
        }
        double sum = 0.0;
        for (Map.Entry<Object, Double> entry : probabilities.entrySet()) {
            if (entry.getValue() == null
                    || !Double.isFinite(entry.getValue())
                    || entry.getValue() < 0.0
                    || entry.getValue() > 1.0) {
                throw new IllegalArgumentException(
                        "Class vote probabilities must be finite values within [0, 1]."
                );
            }
            sum += entry.getValue();
        }
        double tolerance = Math.max(1.0e-12, probabilities.size() * 8.0 * Math.ulp(1.0));
        if (Math.abs(sum - 1.0) > tolerance) {
            throw new IllegalArgumentException(
                    "Class vote probabilities must sum to 1.0, but summed to "
                            + sum + "."
            );
        }
    }

    private static Map<Object, Double> immutableProbabilities(
            Map<Object, Double> probabilities
    ) {
        if (probabilities.isEmpty()) {
            return Map.of();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(probabilities));
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    @Override
    public String toString() {
        return "ForestPredictionResult{"
                + "predictionKind=" + predictionKind
                + ", predictionAvailable=" + predictionAvailable
                + ", prediction=" + prediction
                + ", predictionTreeCount=" + predictionTreeCount
                + ", predictionMean=" + predictionMean
                + ", predictionStandardDeviation="
                + predictionStandardDeviation
                + ", classVoteProbabilities=" + classVoteProbabilities
                + ", oodScoreType=" + oodScoreType
                + ", oodMean=" + oodMean
                + ", oodStandardDeviation=" + oodStandardDeviation
                + ", oodAvailableTreeCount=" + oodAvailableTreeCount
                + ", oodTotalTreeCount=" + oodTotalTreeCount
                + '}';
    }
}
