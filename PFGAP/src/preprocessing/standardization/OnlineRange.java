package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * Mutable online accumulator for minimum, maximum, and observation count.
 *
 * <p>The hot {@link #add(double)} path trusts its caller to supply finite
 * observations. Missing-value filtering belongs to the reader or fitter. This
 * avoids repeating finite-value validation for every observation in large
 * numeric datasets. Use {@link #addChecked(double)} when accepting values from
 * an untrusted source.</p>
 *
 * <p>This class is not thread-safe. Parallel fitting should use independent
 * accumulators and combine them with {@link #merge(OnlineRange)}.</p>
 *
 * <p>The accumulator is intended primarily for min-max standardization. For a
 * nonconstant completed group:</p>
 *
 * <pre>
 * center = minimum
 * scale  = maximum - minimum
 * </pre>
 *
 * <p>Constant-group scale normalization remains the responsibility of the
 * fitter. The standard policy is to replace a zero range with scale
 * {@code 1.0}, causing every value in the constant group to transform to
 * zero.</p>
 */
public final class OnlineRange implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private long count;
    private double minimum;
    private double maximum;

    /**
     * Constructs an empty accumulator.
     */
    public OnlineRange() {
        reset();
    }

    /**
     * Reconstructs a validated accumulator state.
     *
     * <p>An empty state requires {@code count=0},
     * {@code minimum=+infinity}, and {@code maximum=-infinity}. A nonempty
     * state requires finite bounds with {@code minimum <= maximum}.</p>
     *
     * @param count number of accepted observations
     * @param minimum accumulated minimum
     * @param maximum accumulated maximum
     */
    public OnlineRange(
            long count,
            double minimum,
            double maximum
    ) {
        validateState(
                count,
                minimum,
                maximum
        );
        this.count = count;
        this.minimum = minimum;
        this.maximum = maximum;
    }

    /**
     * Adds one trusted finite observation.
     *
     * <p>No NaN or infinity check is performed. Supplying a nonfinite value is
     * a caller contract violation and may corrupt the accumulator.</p>
     *
     * @param value trusted finite observation
     * @throws ArithmeticException if the observation count overflows
     */
    public void add(
            double value
    ) {
        if (count == Long.MAX_VALUE) {
            throw new ArithmeticException(
                    "OnlineRange observation count overflow."
            );
        }

        if (count == 0L) {
            minimum = value;
            maximum = value;
            count = 1L;
            return;
        }

        if (value < minimum) {
            minimum = value;
        }
        if (value > maximum) {
            maximum = value;
        }
        count++;
    }

    /**
     * Validates and adds one finite observation.
     *
     * @param value observation to validate and add
     */
    public void addChecked(
            double value
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "OnlineRange requires a finite observation: "
                            + value
            );
        }
        add(value);
    }

    /**
     * Adds a primitive value only when it is finite.
     *
     * @param value candidate observation
     * @return true when the value was finite and added
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
     *
     * @param value candidate observation
     * @return true when the value was present, finite, and added
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
     * Adds a non-null boxed value through the trusted hot path.
     *
     * @param value candidate observation
     * @return true when the value was non-null and added
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
     * Merges another range state without revisiting its observations.
     *
     * @param other accumulator to merge
     * @throws ArithmeticException if the combined count overflows
     */
    public void merge(
            OnlineRange other
    ) {
        Objects.requireNonNull(
                other,
                "OnlineRange to merge cannot be null."
        );

        if (other.count == 0L) {
            return;
        }
        if (count == 0L) {
            count = other.count;
            minimum = other.minimum;
            maximum = other.maximum;
            return;
        }
        if (Long.MAX_VALUE - count < other.count) {
            throw new ArithmeticException(
                    "OnlineRange observation count overflow during merge."
            );
        }

        count += other.count;
        if (other.minimum < minimum) {
            minimum = other.minimum;
        }
        if (other.maximum > maximum) {
            maximum = other.maximum;
        }
    }

    /**
     * Returns an independent copy of this accumulator.
     *
     * @return copied accumulator
     */
    public OnlineRange copy() {
        return new OnlineRange(
                count,
                minimum,
                maximum
        );
    }

    /**
     * Restores the canonical empty state.
     */
    public void reset() {
        count = 0L;
        minimum = Double.POSITIVE_INFINITY;
        maximum = Double.NEGATIVE_INFINITY;
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

    public double getMinimum() {
        requireObservations();
        return minimum;
    }

    public double getMaximum() {
        requireObservations();
        return maximum;
    }

    /**
     * Returns {@code maximum - minimum}.
     *
     * <p>The result may be positive infinity if two finite bounds at opposite
     * extremes overflow during subtraction. In that case the fitter should
     * reject the range rather than constructing nonfinite standardization
     * statistics.</p>
     *
     * @return observed range
     */
    public double getRange() {
        requireObservations();
        return maximum - minimum;
    }

    /**
     * Returns whether every accepted observation has the same value.
     *
     * @return true for a nonempty zero-range state
     */
    public boolean isConstant() {
        return count > 0L
                && Double.compare(minimum, maximum) == 0;
    }

    public double getMinimumOrDefault(
            double fallback
    ) {
        return count == 0L
                ? fallback
                : minimum;
    }

    public double getMaximumOrDefault(
            double fallback
    ) {
        return count == 0L
                ? fallback
                : maximum;
    }

    public double getRangeOrDefault(
            double fallback
    ) {
        return count == 0L
                ? fallback
                : maximum - minimum;
    }

    private void requireObservations() {
        if (count == 0L) {
            throw new IllegalStateException(
                    "OnlineRange contains no observations."
            );
        }
    }

    private static void validateState(
            long count,
            double minimum,
            double maximum
    ) {
        if (count < 0L) {
            throw new IllegalArgumentException(
                    "Observation count cannot be negative: "
                            + count
            );
        }

        if (count == 0L) {
            if (minimum != Double.POSITIVE_INFINITY
                    || maximum != Double.NEGATIVE_INFINITY) {
                throw new IllegalArgumentException(
                        "An empty OnlineRange requires minimum=+Infinity "
                                + "and maximum=-Infinity."
                );
            }
            return;
        }

        if (!Double.isFinite(minimum)
                || !Double.isFinite(maximum)) {
            throw new IllegalArgumentException(
                    "A nonempty OnlineRange requires finite bounds."
            );
        }
        if (minimum > maximum) {
            throw new IllegalArgumentException(
                    "OnlineRange minimum cannot exceed maximum. Received "
                            + "minimum="
                            + minimum
                            + " and maximum="
                            + maximum
                            + "."
            );
        }
    }

    @Override
    public String toString() {
        return "OnlineRange{"
                + "count=" + count
                + ", minimum=" + minimum
                + ", maximum=" + maximum
                + '}';
    }
}
