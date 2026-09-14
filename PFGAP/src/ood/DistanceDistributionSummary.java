package ood;

import java.io.Serial;
import java.io.Serializable;

/**
 * Immutable compact summary of winning distances observed during training.
 *
 * <p>The serialized size is constant with respect to the number of training
 * observations. The summary stores no raw distances, quantiles, variance
 * accumulator, or per-observation state.</p>
 *
 * <p>Counts distinguish arithmetic distance evaluations from bootstrap
 * multiplicity and also distinguish finite distances from positive-infinite
 * distances:</p>
 *
 * <ul>
 *     <li>{@code distinctCount} counts all distinct observations summarized.</li>
 *     <li>{@code weightedCount} counts all bootstrap occurrences represented.</li>
 *     <li>{@code positiveInfinityDistinctCount} counts distinct observations
 *         whose winning distance was positive infinity.</li>
 *     <li>{@code positiveInfinityWeightedCount} counts the corresponding
 *         bootstrap occurrences.</li>
 * </ul>
 *
 * <p>The mean, population standard deviation, and minimum summarize only the
 * finite distances. The maximum is the empirical upper support boundary and is
 * positive infinity whenever any summarized distance is positive infinity.
 * Consequently, a summary with an infinite maximum does not establish a finite
 * upper support boundary for OOD scoring.</p>
 *
 * <p>An empty summary has zero counts and {@link Double#NaN} for every
 * floating-point statistic. A nonempty all-infinite summary has finite counts,
 * positive-infinity counts equal to its total counts, NaN finite moments and
 * minimum, and a positive-infinite maximum.</p>
 */
public final class DistanceDistributionSummary implements Serializable {

    @Serial
    private static final long serialVersionUID = 2L;

    private static final DistanceDistributionSummary EMPTY =
            new DistanceDistributionSummary(
                    0L,
                    0L,
                    0L,
                    0L,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    false
            );

    private final long distinctCount;
    private final long weightedCount;
    private final long positiveInfinityDistinctCount;
    private final long positiveInfinityWeightedCount;
    private final double mean;
    private final double standardDeviation;
    private final double minimum;
    private final double maximum;

    /**
     * Creates a summary containing only finite distances.
     */
    public DistanceDistributionSummary(
            long distinctCount,
            long weightedCount,
            double mean,
            double standardDeviation,
            double minimum,
            double maximum
    ) {
        this(
                distinctCount,
                weightedCount,
                0L,
                0L,
                mean,
                standardDeviation,
                minimum,
                maximum
        );
    }

    /**
     * Creates a summary that may include positive-infinite distances.
     *
     * @param distinctCount all distinct observations summarized
     * @param weightedCount all bootstrap occurrences represented
     * @param positiveInfinityDistinctCount distinct positive-infinite distances
     * @param positiveInfinityWeightedCount weighted positive-infinite distances
     * @param mean weighted mean of finite distances, or NaN if none are finite
     * @param standardDeviation weighted population standard deviation of finite
     *                          distances, or NaN if none are finite
     * @param minimum minimum finite distance, or NaN if none are finite
     * @param maximum largest finite distance when all distances are finite;
     *                positive infinity when any distance is positive infinity
     */
    public DistanceDistributionSummary(
            long distinctCount,
            long weightedCount,
            long positiveInfinityDistinctCount,
            long positiveInfinityWeightedCount,
            double mean,
            double standardDeviation,
            double minimum,
            double maximum
    ) {
        this(
                distinctCount,
                weightedCount,
                positiveInfinityDistinctCount,
                positiveInfinityWeightedCount,
                mean,
                standardDeviation,
                minimum,
                maximum,
                true
        );
    }

    private DistanceDistributionSummary(
            long distinctCount,
            long weightedCount,
            long positiveInfinityDistinctCount,
            long positiveInfinityWeightedCount,
            double mean,
            double standardDeviation,
            double minimum,
            double maximum,
            boolean validate
    ) {
        if (validate) {
            validate(
                    distinctCount,
                    weightedCount,
                    positiveInfinityDistinctCount,
                    positiveInfinityWeightedCount,
                    mean,
                    standardDeviation,
                    minimum,
                    maximum
            );
        }
        this.distinctCount = distinctCount;
        this.weightedCount = weightedCount;
        this.positiveInfinityDistinctCount = positiveInfinityDistinctCount;
        this.positiveInfinityWeightedCount = positiveInfinityWeightedCount;
        this.mean = canonicalizeZero(mean);
        this.standardDeviation = canonicalizeZero(standardDeviation);
        this.minimum = canonicalizeZero(minimum);
        this.maximum = canonicalizeZero(maximum);
    }

    public static DistanceDistributionSummary empty() {
        return EMPTY;
    }

    public boolean isEmpty() {
        return distinctCount == 0L;
    }

    public long distinctCount() {
        return distinctCount;
    }

    public long weightedCount() {
        return weightedCount;
    }

    public long positiveInfinityDistinctCount() {
        return positiveInfinityDistinctCount;
    }

    public long positiveInfinityWeightedCount() {
        return positiveInfinityWeightedCount;
    }

    public long finiteDistinctCount() {
        return distinctCount - positiveInfinityDistinctCount;
    }

    public long finiteWeightedCount() {
        return weightedCount - positiveInfinityWeightedCount;
    }

    public boolean hasFiniteDistances() {
        return finiteDistinctCount() > 0L;
    }

    public boolean hasPositiveInfinity() {
        return positiveInfinityDistinctCount > 0L;
    }

    public boolean hasFiniteUpperSupportBoundary() {
        return !isEmpty() && Double.isFinite(maximum);
    }

    public double mean() {
        return mean;
    }

    public double standardDeviation() {
        return standardDeviation;
    }

    public double minimum() {
        return minimum;
    }

    /**
     * Returns the empirical upper support boundary. Positive infinity means the
     * training observations did not establish a finite upper boundary.
     */
    public double maximum() {
        return maximum;
    }

    private static void validate(
            long distinctCount,
            long weightedCount,
            long infinityDistinctCount,
            long infinityWeightedCount,
            double mean,
            double standardDeviation,
            double minimum,
            double maximum
    ) {
        validateCounts(
                distinctCount,
                weightedCount,
                infinityDistinctCount,
                infinityWeightedCount
        );

        if (distinctCount == 0L) {
            requireNaN(mean, "mean", "an empty summary");
            requireNaN(standardDeviation, "standardDeviation", "an empty summary");
            requireNaN(minimum, "minimum", "an empty summary");
            requireNaN(maximum, "maximum", "an empty summary");
            return;
        }

        long finiteDistinctCount = distinctCount - infinityDistinctCount;
        if (finiteDistinctCount == 0L) {
            requireNaN(mean, "mean", "an all-infinite summary");
            requireNaN(
                    standardDeviation,
                    "standardDeviation",
                    "an all-infinite summary"
            );
            requireNaN(minimum, "minimum", "an all-infinite summary");
            if (maximum != Double.POSITIVE_INFINITY) {
                throw new IllegalArgumentException(
                        "An all-infinite summary requires maximum=+Infinity."
                );
            }
            return;
        }

        requireFinite(mean, "mean");
        requireFinite(standardDeviation, "standardDeviation");
        requireFinite(minimum, "minimum");
        if (standardDeviation < 0.0) {
            throw new IllegalArgumentException(
                    "standardDeviation cannot be negative, but received "
                            + standardDeviation + "."
            );
        }
        if (Double.compare(mean, minimum) < 0) {
            throw new IllegalArgumentException(
                    "Finite mean cannot be less than finite minimum."
            );
        }

        if (infinityDistinctCount == 0L) {
            requireFinite(maximum, "maximum");
            if (Double.compare(minimum, maximum) > 0
                    || Double.compare(mean, maximum) > 0) {
                throw new IllegalArgumentException(
                        "Finite statistics must satisfy minimum <= mean <= maximum."
                );
            }
        } else if (maximum != Double.POSITIVE_INFINITY) {
            throw new IllegalArgumentException(
                    "A summary containing positive infinity requires "
                            + "maximum=+Infinity."
            );
        }
    }

    private static void validateCounts(
            long distinctCount,
            long weightedCount,
            long infinityDistinctCount,
            long infinityWeightedCount
    ) {
        if (distinctCount < 0L || weightedCount < 0L
                || infinityDistinctCount < 0L
                || infinityWeightedCount < 0L) {
            throw new IllegalArgumentException("Summary counts cannot be negative.");
        }
        if (weightedCount < distinctCount) {
            throw new IllegalArgumentException(
                    "weightedCount must be at least distinctCount."
            );
        }
        if (infinityDistinctCount > distinctCount
                || infinityWeightedCount > weightedCount) {
            throw new IllegalArgumentException(
                    "Positive-infinity counts cannot exceed total counts."
            );
        }
        if (infinityWeightedCount < infinityDistinctCount) {
            throw new IllegalArgumentException(
                    "positiveInfinityWeightedCount must be at least "
                            + "positiveInfinityDistinctCount."
            );
        }
        if ((infinityDistinctCount == 0L) != (infinityWeightedCount == 0L)) {
            throw new IllegalArgumentException(
                    "Positive-infinity distinct and weighted counts must both "
                            + "be zero or both be positive."
            );
        }
        if (distinctCount == 0L && weightedCount != 0L) {
            throw new IllegalArgumentException(
                    "An empty distinct population must have weightedCount=0."
            );
        }
        if (infinityDistinctCount == distinctCount
                && infinityWeightedCount != weightedCount) {
            throw new IllegalArgumentException(
                    "If every distinct distance is infinite, every represented "
                            + "weighted distance must also be infinite."
            );
        }
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    name + " must be finite, but received " + value + "."
            );
        }
    }

    private static void requireNaN(
            double value,
            String name,
            String summaryDescription
    ) {
        if (!Double.isNaN(value)) {
            throw new IllegalArgumentException(
                    name + " must be NaN for " + summaryDescription
                            + ", but received " + value + "."
            );
        }
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    @Serial
    private Object readResolve() {
        return isEmpty() ? EMPTY : this;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof DistanceDistributionSummary summary)) {
            return false;
        }
        return distinctCount == summary.distinctCount
                && weightedCount == summary.weightedCount
                && positiveInfinityDistinctCount
                == summary.positiveInfinityDistinctCount
                && positiveInfinityWeightedCount
                == summary.positiveInfinityWeightedCount
                && Double.compare(mean, summary.mean) == 0
                && Double.compare(standardDeviation, summary.standardDeviation) == 0
                && Double.compare(minimum, summary.minimum) == 0
                && Double.compare(maximum, summary.maximum) == 0;
    }

    @Override
    public int hashCode() {
        int result = Long.hashCode(distinctCount);
        result = 31 * result + Long.hashCode(weightedCount);
        result = 31 * result + Long.hashCode(positiveInfinityDistinctCount);
        result = 31 * result + Long.hashCode(positiveInfinityWeightedCount);
        result = 31 * result + Double.hashCode(mean);
        result = 31 * result + Double.hashCode(standardDeviation);
        result = 31 * result + Double.hashCode(minimum);
        result = 31 * result + Double.hashCode(maximum);
        return result;
    }

    @Override
    public String toString() {
        if (isEmpty()) {
            return "DistanceDistributionSummary{empty}";
        }
        return "DistanceDistributionSummary{"
                + "distinctCount=" + distinctCount
                + ", weightedCount=" + weightedCount
                + ", positiveInfinityDistinctCount="
                + positiveInfinityDistinctCount
                + ", positiveInfinityWeightedCount="
                + positiveInfinityWeightedCount
                + ", finiteDistinctCount=" + finiteDistinctCount()
                + ", finiteWeightedCount=" + finiteWeightedCount()
                + ", mean=" + mean
                + ", standardDeviation=" + standardDeviation
                + ", minimum=" + minimum
                + ", maximum=" + maximum
                + '}';
    }
}
