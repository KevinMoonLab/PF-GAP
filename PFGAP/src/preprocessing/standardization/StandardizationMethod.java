package preprocessing.standardization;

import java.util.Locale;

/**
 * Identifies the affine standardization or scaling method applied to numeric
 * dataset values.
 *
 * <p>The method defines the transformation formula, while
 * {@link StandardizationScope} defines the values over which the required
 * parameters are fitted or calculated.</p>
 *
 * <p>The currently implemented methods are:</p>
 *
 * <ul>
 *     <li>{@link #NONE}</li>
 *     <li>{@link #Z_SCORE}</li>
 *     <li>{@link #MEAN_CENTER}</li>
 *     <li>{@link #MIN_MAX}</li>
 * </ul>
 *
 * <p>{@link #ROBUST} remains a recognized extension point. Its exact quantile
 * convention and fitting strategy must be defined before implementation.</p>
 */
public enum StandardizationMethod {
    /**
     * Do not fit or apply any standardization.
     */
    NONE,

    /**
     * Center values and divide them by a standard deviation:
     *
     * <pre>
     * z = (x - mean) / standardDeviation
     * </pre>
     */
    Z_SCORE,

    /**
     * Subtract a fitted or per-series mean without rescaling:
     *
     * <pre>
     * centered = x - mean
     * </pre>
     *
     * <p>This method uses a stored scale of {@code 1.0}, allowing it to share
     * the common affine forward and inverse transformation kernels.</p>
     */
    MEAN_CENTER,

    /**
     * Shift and scale values using a fitted or per-series minimum and maximum:
     *
     * <pre>
     * scaled = (x - minimum) / (maximum - minimum)
     * </pre>
     *
     * <p>A constant statistic group uses scale {@code 1.0}, causing its values
     * to transform to zero without a special case in the transform kernel.</p>
     */
    MIN_MAX,

    /**
     * Center values using a median and scale them using an interquartile range
     * or another explicitly configured robust scale statistic.
     *
     * <p>This method is recognized for forward compatibility but is not yet
     * implemented because the exact quantile convention and fitting strategy
     * remain to be defined.</p>
     */
    ROBUST;

    /**
     * Parses a user-facing standardization method name.
     *
     * <p>Accepted values are case-insensitive. Hyphens and spaces are treated
     * as underscores. A null, blank, or {@code "none"} value resolves to
     * {@link #NONE}.</p>
     *
     * @param value user-supplied method name
     * @return parsed standardization method
     * @throws IllegalArgumentException if the value is not recognized
     */
    public static StandardizationMethod fromString(
            String value
    ) {
        if (value == null || value.isBlank()) {
            return NONE;
        }

        String normalized =
                value.trim()
                        .toUpperCase(Locale.ROOT)
                        .replace('-', '_')
                        .replace(' ', '_');

        return switch (normalized) {
            case "NONE",
                    "NO",
                    "FALSE",
                    "OFF" ->
                    NONE;

            case "Z_SCORE",
                    "ZSCORE",
                    "STANDARD",
                    "STANDARDIZE",
                    "STANDARDIZATION",
                    "Z_NORMALIZE",
                    "ZNORMALIZE",
                    "Z_NORMALIZATION",
                    "ZNORMALIZATION" ->
                    Z_SCORE;

            case "MEAN_CENTER",
                    "MEAN_CENTERING",
                    "MEANCENTER",
                    "MEANCENTERING",
                    "CENTER",
                    "CENTERING",
                    "CENTRE",
                    "CENTRING" ->
                    MEAN_CENTER;

            case "MIN_MAX",
                    "MINMAX",
                    "RESCALE",
                    "RESCALING",
                    "RANGE",
                    "RANGE_SCALE",
                    "RANGE_SCALING" ->
                    MIN_MAX;

            case "ROBUST",
                    "ROBUST_SCALE",
                    "ROBUST_SCALING" ->
                    ROBUST;

            default ->
                    throw new IllegalArgumentException(
                            "Unknown standardization method: "
                                    + value
                                    + ". Supported methods are: NONE, "
                                    + "Z_SCORE, MEAN_CENTER, MIN_MAX, and "
                                    + "ROBUST. ROBUST is recognized but is "
                                    + "not yet implemented."
                    );
        };
    }

    /**
     * Returns whether this method requires calculated transformation
     * parameters.
     *
     * <p>For {@link StandardizationScope#GLOBAL} and
     * {@link StandardizationScope#PER_DIMENSION}, these parameters are fitted
     * from training data and reused. For per-series scopes, they are calculated
     * from each realized series at transformation time.</p>
     *
     * @return true when center/scale parameters are required
     */
    public boolean requiresFittedStatistics() {
        return this != NONE;
    }

    /**
     * Returns whether this method uses a variance convention.
     *
     * @return true only for z-score standardization
     */
    public boolean usesVarianceConvention() {
        return this == Z_SCORE;
    }

    /**
     * Returns whether this method is represented by an affine center-and-scale
     * transformation and can consequently use the shared forward and inverse
     * kernels.
     *
     * @return true for the currently planned affine methods
     */
    public boolean isAffine() {
        return this == Z_SCORE
                || this == MEAN_CENTER
                || this == MIN_MAX
                || this == ROBUST;
    }

    /**
     * Returns whether this method is currently executable.
     *
     * @return true for NONE, Z_SCORE, MEAN_CENTER, and MIN_MAX
     */
    public boolean isImplemented() {
        return this == NONE
                || this == Z_SCORE
                || this == MEAN_CENTER
                || this == MIN_MAX;
    }

    /**
     * Throws an informative exception if this method has not yet been
     * implemented.
     */
    public void requireImplemented() {
        if (!isImplemented()) {
            throw new UnsupportedOperationException(
                    "Standardization method "
                            + this
                            + " is recognized but is not yet implemented. "
                            + "The current implementation supports NONE, "
                            + "Z_SCORE, MEAN_CENTER, and MIN_MAX."
            );
        }
    }
}
