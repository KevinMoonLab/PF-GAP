package preprocessing.standardization;

import java.util.Locale;

/**
 * Defines the group of numeric values over which standardization parameters
 * are fitted or calculated.
 *
 * <p>{@code GLOBAL} and {@code PER_DIMENSION} use reusable parameters fitted
 * from training data. {@code PER_SERIES} and
 * {@code PER_SERIES_PER_DIMENSION} calculate local parameters from each
 * realized series during transformation.</p>
 */
public enum StandardizationScope {
    /**
     * Fit one center and one scale from every accepted numeric value in the
     * training dataset, across all instances, dimensions, and time points.
     */
    GLOBAL,

    /**
     * Fit one reusable center and scale per realized dimension.
     *
     * <p>For one-dimensional tabular rows, each array position is one feature
     * and receives its own training-set statistic group. For dimension-major
     * multivariate series, each outer-array channel receives its own group
     * fitted across training instances and time points.</p>
     */
    PER_DIMENSION,

    /**
     * Calculate one center and scale independently for each complete realized
     * series.
     *
     * <p>For a univariate series, all observed time points contribute to the
     * local parameters. For a multivariate series, all observed dimensions and
     * time points in that instance contribute to the same local parameters.</p>
     */
    PER_SERIES,

    /**
     * Calculate one center and scale independently for each dimension within
     * each realized series.
     *
     * <p>This is the usual per-instance, per-channel normalization behavior for
     * multivariate time series. For univariate input, it is equivalent to
     * {@link #PER_SERIES}.</p>
     */
    PER_SERIES_PER_DIMENSION;

    /**
     * Parses a user-facing standardization scope name.
     *
     * <p>Parsing is case-insensitive. Hyphens and spaces are converted to
     * underscores. A null or blank value defaults to
     * {@link #PER_DIMENSION}.</p>
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
                    "PER_SERIES_PER_CHANNEL",
                    "PER_INSTANCE_PER_CHANNEL" ->
                    PER_SERIES_PER_DIMENSION;

            default ->
                    throw new IllegalArgumentException(
                            "Unknown standardization scope: "
                                    + value
                                    + ". Supported scopes are: GLOBAL, "
                                    + "PER_DIMENSION, PER_SERIES, and "
                                    + "PER_SERIES_PER_DIMENSION."
                    );
        };
    }

    /**
     * Returns whether this scope uses parameters fitted across the training
     * dataset and subsequently reused for testing or evaluation data.
     *
     * @return true for GLOBAL and PER_DIMENSION
     */
    public boolean usesTrainingStatistics() {
        return this == GLOBAL
                || this == PER_DIMENSION;
    }

    /**
     * Returns whether this scope calculates parameters independently from each
     * realized series during transformation.
     *
     * @return true for either per-series scope
     */
    public boolean usesPerSeriesStatistics() {
        return this == PER_SERIES
                || this == PER_SERIES_PER_DIMENSION;
    }

    /**
     * Returns whether independent parameters are maintained per dimension.
     *
     * @return true for either dimension-wise scope
     */
    public boolean isDimensionWise() {
        return this == PER_DIMENSION
                || this == PER_SERIES_PER_DIMENSION;
    }

    /**
     * Returns whether one parameter group spans every dimension represented by
     * the applicable fitting unit.
     *
     * @return true for GLOBAL and PER_SERIES
     */
    public boolean combinesDimensions() {
        return this == GLOBAL
                || this == PER_SERIES;
    }

    /**
     * Returns whether this scope is currently executable.
     *
     * @return true for every declared scope
     */
    public boolean isImplemented() {
        return true;
    }

    /**
     * Throws if this scope is not implemented.
     *
     * <p>All currently declared scopes are implemented. This method is retained
     * as a stable validation hook for callers and future extension points.</p>
     */
    public void requireImplemented() {
        if (!isImplemented()) {
            throw new UnsupportedOperationException(
                    "Standardization scope "
                            + this
                            + " is recognized but is not yet implemented."
            );
        }
    }

    /**
     * Returns the number of reusable training-statistic groups needed for this
     * scope.
     *
     * <p>This operation applies only to scopes that use training statistics.
     * Per-series scopes are rejected because their parameter groups are local
     * to each realized series and are represented by per-series transformation
     * state rather than {@link StandardizationStats}.</p>
     *
     * @param dimensionCount number of realized dimensions or tabular features
     * @return number of reusable statistic groups
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
                                    + " calculates local per-series "
                                    + "parameters and does not use a fixed "
                                    + "set of reusable training-statistic "
                                    + "groups."
                    );
        };
    }

    /**
     * Returns the number of local parameter groups required for one realized
     * series.
     *
     * <p>This operation applies only to per-series scopes. A univariate series
     * has a dimension count of one.</p>
     *
     * @param dimensionCount number of dimensions in the realized series
     * @return one group for PER_SERIES, or one per dimension for
     *         PER_SERIES_PER_DIMENSION
     */
    public int perSeriesGroupCount(
            int dimensionCount
    ) {
        if (dimensionCount <= 0) {
            throw new IllegalArgumentException(
                    "dimensionCount must be positive, but received: "
                            + dimensionCount
            );
        }

        return switch (this) {
            case PER_SERIES ->
                    1;
            case PER_SERIES_PER_DIMENSION ->
                    dimensionCount;
            case GLOBAL,
                    PER_DIMENSION ->
                    throw new UnsupportedOperationException(
                            "Scope "
                                    + this
                                    + " uses reusable training statistics, "
                                    + "not local per-series parameter groups."
                    );
        };
    }
}
