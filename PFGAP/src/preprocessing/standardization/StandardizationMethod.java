package preprocessing.standardization;

import java.util.Locale;

/**
 * Identifies the mathematical standardization or scaling method applied
 * to numeric dataset values.
 *
 * The method defines the transformation formula, while
 * {@link StandardizationScope} defines the values over which the required
 * statistics are fitted.
 *
 * For example:
 *
 *     method = Z_SCORE
 *     scope  = PER_DIMENSION
 *
 * means that each dimension is centered and scaled using the mean and
 * standard deviation fitted from the training values in that dimension.
 *
 * Initial implementation support:
 *
 *     NONE
 *     Z_SCORE
 *
 * MIN_MAX and ROBUST are included for forward compatibility. Their fitting
 * and transformation implementations may be added after the initial
 * z-score standardization pipeline is complete.
 */
public enum StandardizationMethod {

    /**
     * Do not fit or apply any standardization.
     */
    NONE,

    /**
     * Center values and divide them by a standard deviation:
     *
     *     z = (x - mean) / standardDeviation
     */
    Z_SCORE,

    /**
     * Shift and scale values using a fitted minimum and maximum:
     *
     *     scaled = (x - minimum) / (maximum - minimum)
     *
     * This method is reserved for a later implementation phase.
     */
    MIN_MAX,

    /**
     * Center values using a median and scale them using an interquartile
     * range or another configured robust scale statistic.
     *
     * This method is reserved for a later implementation phase.
     */
    ROBUST;

    /**
     * Parses a user-facing standardization method name.
     *
     * Accepted values are case-insensitive. Hyphens and spaces are treated
     * as underscores, allowing forms such as:
     *
     *     z_score
     *     Z-SCORE
     *     z score
     *     min_max
     *     min-max
     *
     * A null, blank, or "none" value resolves to {@link #NONE}.
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
                    "STANDARDIZATION" ->
                    Z_SCORE;

            case "MIN_MAX",
                    "MINMAX",
                    "RESCALE" ->
                    MIN_MAX;

            case "ROBUST",
                    "ROBUST_SCALE",
                    "ROBUST_SCALING" ->
                    ROBUST;

            default ->
                    throw new IllegalArgumentException(
                            "Unknown standardization method: "
                                    + value
                                    + ". Supported methods are: "
                                    + "NONE, Z_SCORE, MIN_MAX, and ROBUST."
                    );
        };
    }

    /**
     * Returns whether this method requires statistics fitted from data or
     * supplied externally.
     *
     * @return true when fitted statistics are required
     */
    public boolean requiresFittedStatistics() {
        return this != NONE;
    }

    /**
     * Returns whether this method is implemented by the initial
     * standardization pipeline.
     *
     * This allows command-line validation to distinguish methods that are
     * valid design options from methods whose fit and transform operations
     * have not yet been implemented.
     *
     * @return true if the method is currently implemented
     */
    public boolean isImplemented() {
        return this == NONE
                || this == Z_SCORE;
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
                            + " is recognized but is not yet implemented."
            );
        }
    }
}