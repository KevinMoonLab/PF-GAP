package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;

/**
 * Numerically stable online accumulator for the mean and variance of a
 * sequence of finite numeric observations.
 *
 * This class uses Welford's online algorithm and stores:
 *
 *     count
 *         Number of accepted observations.
 *
 *     mean
 *         Running arithmetic mean.
 *
 *     m2
 *         Running sum of squared deviations from the mean.
 *
 * The accumulated state can be converted to either population or sample
 * variance through {@link VarianceConvention}.
 *
 * OnlineMoments can also merge another accumulator without revisiting the
 * original observations. This is useful for:
 *
 *     parallel standardization fitting
 *     per-file accumulation
 *     combining independently processed dataset partitions
 *
 * Missing-value handling is intentionally not built into this class.
 * Callers must decide whether null, NaN, or infinite values should be
 * skipped or rejected before calling add(...).
 *
 * This class is mutable and is not thread-safe. Parallel code should use
 * independent accumulators and merge them after local accumulation.
 */
public final class OnlineMoments implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private long count;
    private double mean;
    private double m2;

    /**
     * Creates an empty accumulator.
     */
    public OnlineMoments() {
        this.count = 0L;
        this.mean = 0.0;
        this.m2 = 0.0;
    }

    /**
     * Creates an accumulator from an existing valid state.
     *
     * This constructor is primarily useful when reconstructing statistics
     * from another representation.
     *
     * @param count number of accumulated observations
     * @param mean accumulated arithmetic mean
     * @param m2 accumulated sum of squared deviations from the mean
     */
    public OnlineMoments(
            long count,
            double mean,
            double m2
    ) {
        validateState(
                count,
                mean,
                m2
        );

        this.count = count;
        this.mean = count == 0L
                ? 0.0
                : mean;
        this.m2 = count == 0L
                ? 0.0
                : normalizeM2(m2);
    }

    /**
     * Adds one finite observation to this accumulator.
     *
     * @param value finite numeric observation
     * @throws IllegalArgumentException if value is NaN or infinite
     * @throws ArithmeticException if the observation count overflows
     */
    public void add(
            double value
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "OnlineMoments accepts only finite observations, "
                            + "but received: "
                            + value
            );
        }

        if (count == Long.MAX_VALUE) {
            throw new ArithmeticException(
                    "OnlineMoments observation count overflow."
            );
        }

        long updatedCount =
                count + 1L;

        double delta =
                value - mean;

        double updatedMean =
                mean + delta / updatedCount;

        double deltaFromUpdatedMean =
                value - updatedMean;

        double updatedM2 =
                m2 + delta * deltaFromUpdatedMean;

        count =
                updatedCount;

        mean =
                updatedMean;

        m2 =
                normalizeM2(updatedM2);
    }

    /**
     * Adds a boxed observation when it is non-null.
     *
     * A null value is ignored and returns false. A non-null NaN or infinite
     * value is rejected in the same way as {@link #add(double)}.
     *
     * @param value boxed observation, possibly null
     * @return true if an observation was added, or false if value was null
     */
    public boolean addIfPresent(
            Double value
    ) {
        if (value == null) {
            return false;
        }

        add(value);

        return true;
    }

    /**
     * Adds an observation only when it is finite.
     *
     * NaN and positive or negative infinity are ignored rather than rejected.
     * This method is useful when the caller deliberately treats non-finite
     * numeric values as missing observations.
     *
     * @param value numeric observation
     * @return true if the observation was added
     */
    public boolean addIfFinite(
            double value
    ) {
        if (!Double.isFinite(value)) {
            return false;
        }

        add(value);

        return true;
    }

    /**
     * Adds a boxed observation only when it is non-null and finite.
     *
     * @param value boxed numeric observation
     * @return true if the observation was added
     */
    public boolean addIfFinite(
            Double value
    ) {
        if (value == null
                || !Double.isFinite(value)) {

            return false;
        }

        add(value);

        return true;
    }

    /**
     * Merges another independent accumulator into this accumulator.
     *
     * The operation uses the parallel form of the online variance algorithm,
     * allowing independently accumulated partitions to be combined without
     * revisiting their observations.
     *
     * Merging an empty accumulator has no effect. If this accumulator is
     * empty, it adopts the complete state of the other accumulator.
     *
     * @param other accumulator to merge
     * @throws IllegalArgumentException if other is null
     * @throws ArithmeticException if the combined count overflows
     */
    public void merge(
            OnlineMoments other
    ) {
        if (other == null) {
            throw new IllegalArgumentException(
                    "Cannot merge a null OnlineMoments accumulator."
            );
        }

        if (other.count == 0L) {
            return;
        }

        if (this.count == 0L) {
            this.count =
                    other.count;

            this.mean =
                    other.mean;

            this.m2 =
                    other.m2;

            return;
        }

        if (Long.MAX_VALUE - this.count < other.count) {
            throw new ArithmeticException(
                    "OnlineMoments observation count overflow while "
                            + "merging accumulators."
            );
        }

        long combinedCount =
                this.count + other.count;

        double delta =
                other.mean - this.mean;

        double firstWeight =
                (double) this.count
                        / combinedCount;

        double secondWeight =
                (double) other.count
                        / combinedCount;

        double combinedMean =
                this.mean * firstWeight
                        + other.mean * secondWeight;

        double crossTerm =
                delta
                        * delta
                        * ((double) this.count * other.count)
                        / combinedCount;

        double combinedM2 =
                this.m2
                        + other.m2
                        + crossTerm;

        this.count =
                combinedCount;

        this.mean =
                combinedMean;

        this.m2 =
                normalizeM2(combinedM2);
    }

    /**
     * Returns an independent copy of this accumulator.
     *
     * @return copied accumulator
     */
    public OnlineMoments copy() {
        return new OnlineMoments(
                count,
                mean,
                m2
        );
    }

    /**
     * Removes all accumulated observations.
     */
    public void reset() {
        count = 0L;
        mean = 0.0;
        m2 = 0.0;
    }

    /**
     * Returns whether no observations have been accumulated.
     *
     * @return true if the count is zero
     */
    public boolean isEmpty() {
        return count == 0L;
    }

    /**
     * Returns whether at least one observation has been accumulated.
     *
     * @return true if the count is positive
     */
    public boolean hasObservations() {
        return count > 0L;
    }

    /**
     * Returns the number of accumulated observations.
     *
     * @return observation count
     */
    public long getCount() {
        return count;
    }

    /**
     * Returns the accumulated arithmetic mean.
     *
     * @return arithmetic mean
     * @throws IllegalStateException if no observations have been accumulated
     */
    public double getMean() {
        requireObservations();

        return mean;
    }

    /**
     * Returns the mean, or the supplied fallback when this accumulator is
     * empty.
     *
     * @param fallback value returned for an empty accumulator
     * @return accumulated mean or fallback
     */
    public double getMeanOrDefault(
            double fallback
    ) {
        return count == 0L
                ? fallback
                : mean;
    }

    /**
     * Returns the accumulated sum of squared deviations from the mean.
     *
     * For an empty accumulator, this value is zero.
     *
     * @return accumulated M2 value
     */
    public double getM2() {
        return m2;
    }

    /**
     * Calculates variance using the requested denominator convention.
     *
     * @param convention population or sample variance convention
     * @return fitted variance
     */
    public double getVariance(
            VarianceConvention convention
    ) {
        requireConvention(convention);

        return convention.variance(
                count,
                m2
        );
    }

    /**
     * Calculates standard deviation using the requested denominator
     * convention.
     *
     * @param convention population or sample variance convention
     * @return fitted standard deviation
     */
    public double getStandardDeviation(
            VarianceConvention convention
    ) {
        requireConvention(convention);

        return convention.standardDeviation(
                count,
                m2
        );
    }

    /**
     * Returns whether this accumulator contains enough observations to
     * calculate variance under the requested convention.
     *
     * Population variance requires at least one observation.
     * Sample variance requires at least two observations.
     *
     * @param convention population or sample variance convention
     * @return true if variance can be calculated
     */
    public boolean canCalculateVariance(
            VarianceConvention convention
    ) {
        requireConvention(convention);

        return count
                > convention.degreesOfFreedom();
    }

    /**
     * Calculates variance when possible, otherwise returns a caller-supplied
     * fallback.
     *
     * @param convention population or sample variance convention
     * @param fallback value returned when observations are insufficient
     * @return variance or fallback
     */
    public double getVarianceOrDefault(
            VarianceConvention convention,
            double fallback
    ) {
        if (!canCalculateVariance(convention)) {
            return fallback;
        }

        return getVariance(convention);
    }

    /**
     * Calculates standard deviation when possible, otherwise returns a
     * caller-supplied fallback.
     *
     * @param convention population or sample variance convention
     * @param fallback value returned when observations are insufficient
     * @return standard deviation or fallback
     */
    public double getStandardDeviationOrDefault(
            VarianceConvention convention,
            double fallback
    ) {
        if (!canCalculateVariance(convention)) {
            return fallback;
        }

        return getStandardDeviation(convention);
    }

    private void requireObservations() {
        if (count == 0L) {
            throw new IllegalStateException(
                    "OnlineMoments contains no observations."
            );
        }
    }

    private static void requireConvention(
            VarianceConvention convention
    ) {
        if (convention == null) {
            throw new IllegalArgumentException(
                    "VarianceConvention cannot be null."
            );
        }
    }

    private static void validateState(
            long count,
            double mean,
            double m2
    ) {
        if (count < 0L) {
            throw new IllegalArgumentException(
                    "Observation count cannot be negative: "
                            + count
            );
        }

        if (!Double.isFinite(mean)) {
            throw new IllegalArgumentException(
                    "Mean must be finite: "
                            + mean
            );
        }

        if (!Double.isFinite(m2)) {
            throw new IllegalArgumentException(
                    "Accumulated squared deviation must be finite: "
                            + m2
            );
        }

        if (count == 0L) {
            if (mean != 0.0 || m2 != 0.0) {
                throw new IllegalArgumentException(
                        "An empty OnlineMoments accumulator must have "
                                + "mean=0.0 and m2=0.0."
                );
            }

            return;
        }

        normalizeM2(m2);
    }

    /**
     * Clamps very small negative M2 values caused by floating-point
     * roundoff, while rejecting meaningfully negative values.
     */
    private static double normalizeM2(
            double value
    ) {
        if (value >= 0.0) {
            return value;
        }

        /*
         * Scale the tolerance to the magnitude of the accumulated value.
         * The minimum scale of 1.0 keeps the comparison meaningful when M2
         * is close to zero.
         */
        double tolerance =
                32.0
                        * Math.ulp(
                        Math.max(
                                1.0,
                                Math.abs(value)
                        )
                );

        if (value >= -tolerance) {
            return 0.0;
        }

        throw new IllegalArgumentException(
                "Accumulated squared deviation cannot be negative: "
                        + value
        );
    }

    @Override
    public String toString() {
        return "OnlineMoments{"
                + "count=" + count
                + ", mean=" + mean
                + ", m2=" + m2
                + '}';
    }
}