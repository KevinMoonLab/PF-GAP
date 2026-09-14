package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * Mutable Welford accumulator for online mean and variance fitting.
 *
 * <p>The hot {@link #add(double)} path trusts its caller to supply finite
 * observations. Missing-value filtering belongs to the reader or fitter.
 * This avoids repeating finite-value validation for every observation in
 * large numeric datasets. Use {@link #addChecked(double)} when accepting
 * values from an untrusted source.</p>
 *
 * <p>This class is not thread-safe. Parallel fitting should use independent
 * accumulators and combine them with {@link #merge(OnlineMoments)}.</p>
 */
public final class OnlineMoments implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private long count;
    private double mean;
    private double m2;

    public OnlineMoments() {
        // Java's zero initialization is exactly the empty Welford state.
    }

    /**
     * Reconstructs a validated accumulator state.
     */
    public OnlineMoments(
            long count,
            double mean,
            double m2
    ) {
        validateState(count, mean, m2);
        this.count = count;
        this.mean = count == 0L ? 0.0 : mean;
        this.m2 = count == 0L ? 0.0 : normalizeM2(m2);
    }

    /**
     * Adds one trusted finite observation using Welford's recurrence.
     *
     * <p>No NaN or infinity check is performed. Supplying a nonfinite value
     * corrupts the accumulator and is a caller configuration error.</p>
     *
     * @param value trusted finite observation
     * @throws ArithmeticException if the observation count overflows
     */
    public void add(
            double value
    ) {
        if (count == Long.MAX_VALUE) {
            throw new ArithmeticException(
                    "OnlineMoments observation count overflow."
            );
        }

        long updatedCount = count + 1L;
        double delta = value - mean;
        double updatedMean = mean + delta / updatedCount;

        m2 += delta * (value - updatedMean);
        mean = updatedMean;
        count = updatedCount;
    }

    /**
     * Validates and adds one finite observation.
     */
    public void addChecked(
            double value
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "OnlineMoments requires a finite observation: " + value
            );
        }

        add(value);
    }

    /**
     * Adds a non-null boxed value through the trusted hot path.
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
     * Adds a primitive value only when it is finite.
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
     * Adds a boxed value only when it is non-null and finite.
     */
    public boolean addIfFinite(
            Double value
    ) {
        if (value == null || !Double.isFinite(value)) {
            return false;
        }

        add(value);
        return true;
    }

    /**
     * Merges another Welford state without revisiting observations.
     */
    public void merge(
            OnlineMoments other
    ) {
        Objects.requireNonNull(other, "OnlineMoments to merge cannot be null.");

        if (other.count == 0L) {
            return;
        }

        if (count == 0L) {
            count = other.count;
            mean = other.mean;
            m2 = other.m2;
            return;
        }

        if (Long.MAX_VALUE - count < other.count) {
            throw new ArithmeticException(
                    "OnlineMoments observation count overflow during merge."
            );
        }

        long combinedCount = count + other.count;
        double delta = other.mean - mean;
        double firstWeight = (double) count;
        double secondWeight = (double) other.count;

        mean += delta * secondWeight / combinedCount;
        m2 += other.m2
                + delta * delta
                * firstWeight * secondWeight
                / combinedCount;
        count = combinedCount;

        m2 = normalizeM2(m2);

        if (!Double.isFinite(mean) || !Double.isFinite(m2)) {
            throw new ArithmeticException(
                    "Merging OnlineMoments produced nonfinite state."
            );
        }
    }

    public OnlineMoments copy() {
        return new OnlineMoments(count, mean, m2);
    }

    public void reset() {
        count = 0L;
        mean = 0.0;
        m2 = 0.0;
    }

    public boolean isEmpty() {
        return count == 0L;
    }

    public boolean hasObservations() {
        return count > 0L;
    }

    public long getCount() {
        return count;
    }

    public double getMean() {
        requireObservations();
        return mean;
    }

    public double getMeanOrDefault(
            double fallback
    ) {
        return count == 0L ? fallback : mean;
    }

    public double getM2() {
        return m2;
    }

    public double getVariance(
            VarianceConvention convention
    ) {
        requireConvention(convention);
        return convention.variance(count, normalizedM2());
    }

    public double getStandardDeviation(
            VarianceConvention convention
    ) {
        requireConvention(convention);
        return convention.standardDeviation(count, normalizedM2());
    }

    public boolean canCalculateVariance(
            VarianceConvention convention
    ) {
        requireConvention(convention);
        return count > convention.degreesOfFreedom();
    }

    public double getVarianceOrDefault(
            VarianceConvention convention,
            double fallback
    ) {
        return canCalculateVariance(convention)
                ? getVariance(convention)
                : fallback;
    }

    public double getStandardDeviationOrDefault(
            VarianceConvention convention,
            double fallback
    ) {
        return canCalculateVariance(convention)
                ? getStandardDeviation(convention)
                : fallback;
    }

    private double normalizedM2() {
        m2 = normalizeM2(m2);
        return m2;
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
        Objects.requireNonNull(
                convention,
                "VarianceConvention cannot be null."
        );
    }

    private static void validateState(
            long count,
            double mean,
            double m2
    ) {
        if (count < 0L) {
            throw new IllegalArgumentException(
                    "Observation count cannot be negative: " + count
            );
        }

        if (!Double.isFinite(mean) || !Double.isFinite(m2)) {
            throw new IllegalArgumentException(
                    "OnlineMoments state must be finite."
            );
        }

        if (count == 0L && (mean != 0.0 || m2 != 0.0)) {
            throw new IllegalArgumentException(
                    "An empty accumulator requires mean=0.0 and m2=0.0."
            );
        }

        normalizeM2(m2);
    }

    /**
     * Clamps tiny negative M2 values attributable to floating-point roundoff.
     */
    private static double normalizeM2(
            double value
    ) {
        if (value >= 0.0) {
            return value;
        }

        double tolerance =
                32.0 * Math.ulp(Math.max(1.0, Math.abs(value)));

        if (value >= -tolerance) {
            return 0.0;
        }

        throw new IllegalArgumentException(
                "Accumulated squared deviation cannot be negative: " + value
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