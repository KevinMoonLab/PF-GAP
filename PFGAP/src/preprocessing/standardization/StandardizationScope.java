package preprocessing.standardization;

import java.util.Locale;

/**
 * Defines the group of numeric values over which standardization statistics
 * are fitted and applied.
 *
 * StandardizationScope is independent of {@link StandardizationMethod}.
 * For example:
 *
 *     method = Z_SCORE
 *     scope  = PER_DIMENSION
 *
 * means that each dimension receives its own mean and standard deviation.
 *
 * Initial implementation support:
 *
 *     GLOBAL
 *     PER_DIMENSION
 *
 * PER_SERIES and PER_SERIES_PER_DIMENSION are included as planned extension
 * points. They do not use training-set-level fitted statistics in the same
 * way as GLOBAL and PER_DIMENSION, so they will be implemented separately.
 */
public enum StandardizationScope {

    /**
     * Fit one center and one scale from every numeric value in the training
     * dataset, across all instances, dimensions, and time points.
     *
     * For z-score standardization:
     *
     *     z = (x - globalMean) / globalStandardDeviation
     */
    GLOBAL,

    /**
     * Fit one center and one scale per dimension using all training values
     * belonging to that dimension across all instances and time points.
     *
     * For dimension d:
     *
     *     z[d][t] =
     *         (x[d][t] - mean[d]) / standardDeviation[d]
     *
     * This is the recommended initial scope for multivariate time-series
     * datasets whose dimensions may have different physical units or scales.
     */
    PER_DIMENSION,

    /**
     * Fit one center and one scale independently for each complete series.
     *
     * For multivariate data, all dimensions and time points belonging to one
     * instance contribute to the same per-series statistics.
     *
     * This scope is recognized for future implementation but is not included
     * in the initial fitted training-statistics pipeline.
     */
    PER_SERIES,

    /**
     * Fit one center and one scale independently for each dimension within
     * each series.
     *
     * This is the usual per-instance, per-channel z-normalization behavior
     * for multivariate time series. It removes the offset and amplitude of
     * every dimension separately within every instance.
     *
     * This scope is recognized for future implementation but is not included
     * in the initial fitted training-statistics pipeline.
     */
    PER_SERIES_PER_DIMENSION;

    /**
     * Parses a user-facing standardization scope name.
     *
     * Parsing is case-insensitive. Hyphens and spaces are converted to
     * underscores, allowing forms such as:
     *
     *     global
     *     per_dimension
     *     per-dimension
     *     per dimension
     *     per_series
     *     per-series-per-dimension
     *
     * A null or blank value defaults to {@link #PER_DIMENSION}. This default
     * is generally appropriate for multivariate data because it prevents
     * dimensions with larger numerical scales from disproportionately
     * influencing distance calculations.
     *
     * @param value user-supplied scope name
     * @return parsed standardization scope
     * @throws IllegalArgumentException if the value is not recognized
     */
    public static StandardizationScope fromString(
            String value
    ) {
        if (value == null || value.isBlank()) {
            return PER_DIMENSION;
        }

        String normalized =
                value.trim()
                        .toUpperCase(Locale.ROOT)
                        .replace('-', '_')
                        .replace(' ', '_');

        return switch (normalized) {
            case "GLOBAL",
                    "DATASET",
                    "WHOLE_DATASET" ->
                    GLOBAL;

            case "PER_DIMENSION",
                    "DIMENSION",
                    "DIMENSIONS",
                    "PER_FEATURE",
                    "FEATURE",
                    "FEATURES",
                    "PER_CHANNEL",
                    "CHANNEL",
                    "CHANNELS" ->
                    PER_DIMENSION;

            case "PER_SERIES",
                    "SERIES",
                    "PER_INSTANCE",
                    "INSTANCE" ->
                    PER_SERIES;

            case "PER_SERIES_PER_DIMENSION",
                    "PER_SERIES_DIMENSION",
                    "SERIES_PER_DIMENSION",
                    "PER_INSTANCE_PER_DIMENSION",
                    "PER_INSTANCE_DIMENSION",
                    "PER_SERIES_PER_FEATURE",
                    "PER_INSTANCE_PER_FEATURE",
                    "PER_SERIES_PER_CHANNEL" ->
                    PER_SERIES_PER_DIMENSION;

            default ->
                    throw new IllegalArgumentException(
                            "Unknown standardization scope: "
                                    + value
                                    + ". Supported scopes are: "
                                    + "GLOBAL, PER_DIMENSION, PER_SERIES, "
                                    + "and PER_SERIES_PER_DIMENSION."
                    );
        };
    }

    /**
     * Returns whether this scope uses statistics fitted across the training
     * dataset and then reused for testing data.
     *
     * GLOBAL and PER_DIMENSION require fitted training statistics.
     * PER_SERIES and PER_SERIES_PER_DIMENSION calculate statistics from each
     * individual series at transformation time.
     *
     * @return true if the scope uses reusable training-set statistics
     */
    public boolean usesTrainingStatistics() {
        return this == GLOBAL
                || this == PER_DIMENSION;
    }

    /**
     * Returns whether this scope calculates statistics separately for each
     * series during transformation.
     *
     * @return true for either per-series scope
     */
    public boolean usesPerSeriesStatistics() {
        return this == PER_SERIES
                || this == PER_SERIES_PER_DIMENSION;
    }

    /**
     * Returns whether statistics are maintained separately per dimension.
     *
     * For PER_DIMENSION, the resulting statistics are fitted from the whole
     * training dataset and reused.
     *
     * For PER_SERIES_PER_DIMENSION, statistics are calculated independently
     * for each dimension of each individual series.
     *
     * @return true when dimensions have independent statistics
     */
    public boolean isDimensionWise() {
        return this == PER_DIMENSION
                || this == PER_SERIES_PER_DIMENSION;
    }

    /**
     * Returns whether this scope is supported by the initial Phase 1
     * standardization implementation.
     *
     * @return true for GLOBAL and PER_DIMENSION
     */
    public boolean isImplemented() {
        return this == GLOBAL
                || this == PER_DIMENSION;
    }

    /**
     * Throws an informative exception if this scope has not yet been
     * implemented.
     */
    public void requireImplemented() {
        if (!isImplemented()) {
            throw new UnsupportedOperationException(
                    "Standardization scope "
                            + this
                            + " is recognized but is not yet implemented. "
                            + "The initial implementation supports GLOBAL "
                            + "and PER_DIMENSION."
            );
        }
    }

    /**
     * Returns the number of independently fitted statistic groups needed
     * for this scope.
     *
     * GLOBAL always requires one group. PER_DIMENSION requires one group for
     * each dimension.
     *
     * Per-series scopes are rejected because their number of statistic
     * groups depends on each individual series and they are not represented
     * by reusable training statistics.
     *
     * @param dimensionCount number of dimensions in the numeric data
     * @return number of independently fitted statistic groups
     */
    public int statisticGroupCount(
            int dimensionCount
    ) {
        if (dimensionCount <= 0) {
            throw new IllegalArgumentException(
                    "dimensionCount must be positive, but received: "
                            + dimensionCount
            );
        }

        return switch (this) {
            case GLOBAL ->
                    1;

            case PER_DIMENSION ->
                    dimensionCount;

            case PER_SERIES,
                    PER_SERIES_PER_DIMENSION ->
                    throw new UnsupportedOperationException(
                            "Scope "
                                    + this
                                    + " does not use a fixed set of reusable "
                                    + "training-statistic groups."
                    );
        };
    }
}