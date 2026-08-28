package preprocessing.standardization;

import java.util.Locale;

/**
 * Defines the denominator convention used when converting accumulated
 * squared deviations into a variance.
 *
 * For a collection of n observed values with accumulated squared deviation
 * M2:
 *
 *     POPULATION:
 *
 *         variance = M2 / n
 *
 *     SAMPLE:
 *
 *         variance = M2 / (n - 1)
 *
 * Population variance is the default for machine-learning standardization,
 * where the available training data is treated as the fitted population.
 */
public enum VarianceConvention {

    /**
     * Divide the accumulated squared deviation by n.
     *
     * This convention is appropriate when the observed training values are
     * treated as the complete population used to define the transformation.
     */
    POPULATION,

    /**
     * Divide the accumulated squared deviation by n - 1.
     *
     * This applies Bessel's correction and is appropriate when estimating
     * population variance from a sample.
     */
    SAMPLE;

    /**
     * Parses a user-facing variance-convention name.
     *
     * Parsing is case-insensitive. Hyphens and spaces are converted to
     * underscores.
     *
     * Accepted population aliases include:
     *
     *     population
     *     population_variance
     *     n
     *     ddof_0
     *     ddof0
     *
     * Accepted sample aliases include:
     *
     *     sample
     *     sample_variance
     *     n_minus_1
     *     ddof_1
     *     ddof1
     *
     * A null or blank value defaults to {@link #POPULATION}.
     *
     * @param value user-supplied variance convention
     * @return parsed variance convention
     * @throws IllegalArgumentException if the value is not recognized
     */
    public static VarianceConvention fromString(
            String value
    ) {
        if (value == null || value.isBlank()) {
            return POPULATION;
        }

        String normalized =
                value.trim()
                        .toUpperCase(Locale.ROOT)
                        .replace('-', '_')
                        .replace(' ', '_');

        return switch (normalized) {
            case "POPULATION",
                    "POPULATION_VARIANCE",
                    "N",
                    "DDOF_0",
                    "DDOF0" ->
                    POPULATION;

            case "SAMPLE",
                    "SAMPLE_VARIANCE",
                    "N_MINUS_1",
                    "N_1",
                    "BESSEL",
                    "BESSELS_CORRECTION",
                    "DDOF_1",
                    "DDOF1" ->
                    SAMPLE;

            default ->
                    throw new IllegalArgumentException(
                            "Unknown variance convention: "
                                    + value
                                    + ". Supported conventions are "
                                    + "POPULATION and SAMPLE."
                    );
        };
    }

    /**
     * Returns the degrees-of-freedom adjustment associated with this
     * convention.
     *
     * The variance denominator is:
     *
     *     count - degreesOfFreedom()
     *
     * @return 0 for population variance or 1 for sample variance
     */
    public int degreesOfFreedom() {
        return switch (this) {
            case POPULATION -> 0;
            case SAMPLE -> 1;
        };
    }

    /**
     * Returns the denominator used to calculate variance for the supplied
     * observation count.
     *
     * @param count number of observations
     * @return variance denominator
     * @throws IllegalArgumentException if count is negative or insufficient
     *                                  for this convention
     */
    public long denominator(
            long count
    ) {
        if (count < 0) {
            throw new IllegalArgumentException(
                    "Observation count cannot be negative: "
                            + count
            );
        }

        long denominator =
                count - degreesOfFreedom();

        if (denominator <= 0) {
            throw new IllegalArgumentException(
                    "Variance convention "
                            + this
                            + " requires more than "
                            + degreesOfFreedom()
                            + " observation(s), but received "
                            + count
                            + "."
            );
        }

        return denominator;
    }

    /**
     * Calculates variance from an observation count and an accumulated
     * sum of squared deviations from the mean.
     *
     * Small negative M2 values caused by floating-point roundoff are clamped
     * to zero. Meaningfully negative values are rejected.
     *
     * @param count number of observations
     * @param m2 accumulated squared deviation from the mean
     * @return variance according to this convention
     */
    public double variance(
            long count,
            double m2
    ) {
        if (!Double.isFinite(m2)) {
            throw new IllegalArgumentException(
                    "Accumulated squared deviation must be finite: "
                            + m2
            );
        }

        double adjustedM2 =
                normalizeM2(m2);

        return adjustedM2
                / denominator(count);
    }

    /**
     * Calculates standard deviation from an observation count and an
     * accumulated sum of squared deviations from the mean.
     *
     * @param count number of observations
     * @param m2 accumulated squared deviation from the mean
     * @return standard deviation according to this convention
     */
    public double standardDeviation(
            long count,
            double m2
    ) {
        return Math.sqrt(
                variance(
                        count,
                        m2
                )
        );
    }

    /**
     * Handles tiny negative M2 values caused by floating-point roundoff.
     */
    private static double normalizeM2(
            double m2
    ) {
        if (m2 >= 0.0) {
            return m2;
        }

        /*
         * Welford accumulation should not ordinarily produce a negative M2.
         * A value within a small multiple of machine precision can, however,
         * result from floating-point roundoff.
         */
        double tolerance =
                16.0 * Math.ulp(1.0);

        if (m2 >= -tolerance) {
            return 0.0;
        }

        throw new IllegalArgumentException(
                "Accumulated squared deviation cannot be negative: "
                        + m2
        );
    }
}