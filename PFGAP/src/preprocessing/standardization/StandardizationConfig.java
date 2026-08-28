package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * Immutable configuration describing how standardization should be fitted
 * or applied.
 *
 * This class contains standardization policy, not fitted numeric statistics.
 * Fitted centers and scales are stored separately in
 * {@link StandardizationStats}.
 *
 * Phase 1 supports:
 *
 *     method:
 *         NONE
 *         Z_SCORE
 *
 *     scope:
 *         GLOBAL
 *         PER_DIMENSION
 *
 *     variance convention:
 *         POPULATION
 *         SAMPLE
 *
 * Future phases may additionally support:
 *
 *     MIN_MAX
 *     ROBUST
 *     PER_SERIES
 *     PER_SERIES_PER_DIMENSION
 *     JSON statistics loading and writing
 *     streaming fitting for lazy datasets
 *
 * Configuration examples:
 *
 *     StandardizationConfig.disabled()
 *
 *     StandardizationConfig.zScorePerDimension()
 *
 *     StandardizationConfig.builder()
 *         .setMethod(StandardizationMethod.Z_SCORE)
 *         .setScope(StandardizationScope.GLOBAL)
 *         .setVarianceConvention(VarianceConvention.SAMPLE)
 *         .build()
 */
public final class StandardizationConfig
        implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * Standardization method to apply.
     */
    private final StandardizationMethod method;

    /**
     * Scope over which statistics are fitted.
     */
    private final StandardizationScope scope;

    /**
     * Variance denominator convention used for z-score fitting.
     */
    private final VarianceConvention varianceConvention;

    /**
     * Optional path to externally supplied standardization statistics.
     *
     * This field is retained in the configuration now so the public API does
     * not need to change when JSON support is added in Phase 2.
     *
     * A null value means that no external statistics file was supplied.
     */
    private final String statisticsPath;

    /**
     * Whether fitted statistics should later be written to a file.
     *
     * Phase 1 stores fitted statistics in memory only. Phase 2 can use this
     * flag together with statisticsOutputPath.
     */
    private final boolean saveFittedStatistics;

    /**
     * Optional output path for fitted statistics.
     *
     * A null value allows the application layer to choose a default output
     * location when saveFittedStatistics is true.
     */
    private final String statisticsOutputPath;

    private StandardizationConfig(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            String statisticsPath,
            boolean saveFittedStatistics,
            String statisticsOutputPath
    ) {
        this.method =
                Objects.requireNonNull(
                        method,
                        "StandardizationMethod cannot be null."
                );

        this.scope =
                Objects.requireNonNull(
                        scope,
                        "StandardizationScope cannot be null."
                );

        this.varianceConvention =
                Objects.requireNonNull(
                        varianceConvention,
                        "VarianceConvention cannot be null."
                );

        this.statisticsPath =
                normalizeNullablePath(
                        statisticsPath
                );

        this.saveFittedStatistics =
                saveFittedStatistics;

        this.statisticsOutputPath =
                normalizeNullablePath(
                        statisticsOutputPath
                );

        validate();
    }

    /**
     * Returns a configuration that disables standardization.
     *
     * @return disabled standardization configuration
     */
    public static StandardizationConfig disabled() {
        return new StandardizationConfig(
                StandardizationMethod.NONE,
                StandardizationScope.PER_DIMENSION,
                VarianceConvention.POPULATION,
                null,
                false,
                null
        );
    }

    /**
     * Returns the recommended initial z-score configuration:
     *
     *     method              = Z_SCORE
     *     scope               = PER_DIMENSION
     *     variance convention = POPULATION
     *
     * @return per-dimension population z-score configuration
     */
    public static StandardizationConfig zScorePerDimension() {
        return new StandardizationConfig(
                StandardizationMethod.Z_SCORE,
                StandardizationScope.PER_DIMENSION,
                VarianceConvention.POPULATION,
                null,
                false,
                null
        );
    }

    /**
     * Returns a global population z-score configuration.
     *
     * @return global population z-score configuration
     */
    public static StandardizationConfig zScoreGlobal() {
        return new StandardizationConfig(
                StandardizationMethod.Z_SCORE,
                StandardizationScope.GLOBAL,
                VarianceConvention.POPULATION,
                null,
                false,
                null
        );
    }

    /**
     * Returns a new configuration builder.
     *
     * @return standardization configuration builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Returns a builder initialized from an existing configuration.
     *
     * @param config source configuration
     * @return initialized builder
     */
    public static Builder builder(
            StandardizationConfig config
    ) {
        Objects.requireNonNull(
                config,
                "StandardizationConfig cannot be null."
        );

        return new Builder()
                .setMethod(config.getMethod())
                .setScope(config.getScope())
                .setVarianceConvention(
                        config.getVarianceConvention()
                )
                .setStatisticsPath(
                        config.getStatisticsPath()
                )
                .setSaveFittedStatistics(
                        config.shouldSaveFittedStatistics()
                )
                .setStatisticsOutputPath(
                        config.getStatisticsOutputPath()
                );
    }

    /**
     * Returns the configured standardization method.
     *
     * @return standardization method
     */
    public StandardizationMethod getMethod() {
        return method;
    }

    /**
     * Returns the configured standardization scope.
     *
     * @return standardization scope
     */
    public StandardizationScope getScope() {
        return scope;
    }

    /**
     * Returns the configured variance convention.
     *
     * @return variance convention
     */
    public VarianceConvention getVarianceConvention() {
        return varianceConvention;
    }

    /**
     * Returns the path to externally supplied statistics, or null when no
     * statistics file was supplied.
     *
     * @return external statistics path or null
     */
    public String getStatisticsPath() {
        return statisticsPath;
    }

    /**
     * Returns whether an external statistics path was supplied.
     *
     * @return true when statistics should be loaded externally
     */
    public boolean hasStatisticsPath() {
        return statisticsPath != null;
    }

    /**
     * Returns whether fitted statistics should be written for later reuse.
     *
     * @return true when fitted statistics should be saved
     */
    public boolean shouldSaveFittedStatistics() {
        return saveFittedStatistics;
    }

    /**
     * Returns the requested statistics output path, or null when the
     * application should choose a default path.
     *
     * @return statistics output path or null
     */
    public String getStatisticsOutputPath() {
        return statisticsOutputPath;
    }

    /**
     * Returns whether standardization is enabled.
     *
     * @return true unless method is NONE
     */
    public boolean isEnabled() {
        return method != StandardizationMethod.NONE;
    }

    /**
     * Returns whether standardization is disabled.
     *
     * @return true when method is NONE
     */
    public boolean isDisabled() {
        return !isEnabled();
    }

    /**
     * Returns whether fitted statistics should be loaded from an external
     * source rather than calculated from training data.
     *
     * @return true when an external statistics path is configured
     */
    public boolean shouldLoadStatistics() {
        return isEnabled()
                && hasStatisticsPath();
    }

    /**
     * Returns whether statistics must be fitted from training data.
     *
     * Per-series scopes calculate statistics at transformation time rather
     * than fitting reusable training statistics. Those scopes are reserved
     * for a later implementation phase.
     *
     * @return true when reusable statistics must be fitted
     */
    public boolean shouldFitStatistics() {
        return isEnabled()
                && !hasStatisticsPath()
                && scope.usesTrainingStatistics()
                && method.requiresFittedStatistics();
    }

    /**
     * Returns whether statistics are calculated separately for each series
     * during transformation.
     *
     * @return true for a per-series scope
     */
    public boolean usesPerSeriesStatistics() {
        return isEnabled()
                && scope.usesPerSeriesStatistics();
    }

    /**
     * Validates that this configuration is supported by the current
     * implementation phase.
     *
     * This method may be called by PFApplication or ExperimentRunner before
     * reading or transforming data.
     *
     * JSON statistics paths are recognized by the configuration model but
     * are not implemented until Phase 2.
     */
    public void requireImplemented() {
        if (isDisabled()) {
            return;
        }

        method.requireImplemented();
        scope.requireImplemented();
    }

    /**
     * Validates that fitted statistics match this configuration.
     *
     * @param stats fitted or externally loaded statistics
     */
    public void validateStatistics(
            StandardizationStats stats
    ) {
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );

        if (isDisabled()) {
            throw new IllegalStateException(
                    "Cannot validate fitted statistics against a disabled "
                            + "standardization configuration."
            );
        }

        if (stats.getMethod() != method) {
            throw new IllegalArgumentException(
                    "Standardization method mismatch. Configuration uses "
                            + method
                            + ", but statistics use "
                            + stats.getMethod()
                            + "."
            );
        }

        if (stats.getScope() != scope) {
            throw new IllegalArgumentException(
                    "Standardization scope mismatch. Configuration uses "
                            + scope
                            + ", but statistics use "
                            + stats.getScope()
                            + "."
            );
        }

        if (method == StandardizationMethod.Z_SCORE
                && stats.getVarianceConvention()
                != varianceConvention) {

            throw new IllegalArgumentException(
                    "Variance convention mismatch. Configuration uses "
                            + varianceConvention
                            + ", but statistics use "
                            + stats.getVarianceConvention()
                            + "."
            );
        }
    }

    private void validate() {
        if (method == StandardizationMethod.NONE) {
            if (statisticsPath != null) {
                throw new IllegalArgumentException(
                        "A statistics path cannot be supplied when "
                                + "standardization method is NONE."
                );
            }

            if (saveFittedStatistics) {
                throw new IllegalArgumentException(
                        "Fitted statistics cannot be saved when "
                                + "standardization method is NONE."
                );
            }

            if (statisticsOutputPath != null) {
                throw new IllegalArgumentException(
                        "A statistics output path cannot be supplied when "
                                + "standardization method is NONE."
                );
            }

            return;
        }

        /*
         * Recognized future methods and scopes are allowed in the immutable
         * configuration object. requireImplemented() is responsible for
         * rejecting them before execution.
         */

        if (!saveFittedStatistics
                && statisticsOutputPath != null) {

            throw new IllegalArgumentException(
                    "statisticsOutputPath was supplied, but "
                            + "saveFittedStatistics is false."
            );
        }

        if (statisticsPath != null
                && saveFittedStatistics) {

            throw new IllegalArgumentException(
                    "A standardization configuration cannot both load "
                            + "precomputed statistics and save newly fitted "
                            + "statistics in the same operation."
            );
        }
    }

    private static String normalizeNullablePath(
            String path
    ) {
        if (path == null) {
            return null;
        }

        String trimmed =
                path.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    @Override
    public boolean equals(
            Object other
    ) {
        if (this == other) {
            return true;
        }

        if (!(other instanceof StandardizationConfig that)) {
            return false;
        }

        return saveFittedStatistics
                == that.saveFittedStatistics
                && method == that.method
                && scope == that.scope
                && varianceConvention == that.varianceConvention
                && Objects.equals(
                statisticsPath,
                that.statisticsPath
        )
                && Objects.equals(
                statisticsOutputPath,
                that.statisticsOutputPath
        );
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                method,
                scope,
                varianceConvention,
                statisticsPath,
                saveFittedStatistics,
                statisticsOutputPath
        );
    }

    @Override
    public String toString() {
        return "StandardizationConfig{"
                + "method=" + method
                + ", scope=" + scope
                + ", varianceConvention=" + varianceConvention
                + ", statisticsPath='" + statisticsPath + '\''
                + ", saveFittedStatistics=" + saveFittedStatistics
                + ", statisticsOutputPath='"
                + statisticsOutputPath
                + '\''
                + '}';
    }

    /**
     * Builder for immutable StandardizationConfig instances.
     */
    public static final class Builder {

        private StandardizationMethod method =
                StandardizationMethod.NONE;

        private StandardizationScope scope =
                StandardizationScope.PER_DIMENSION;

        private VarianceConvention varianceConvention =
                VarianceConvention.POPULATION;

        private String statisticsPath =
                null;

        private boolean saveFittedStatistics =
                false;

        private String statisticsOutputPath =
                null;

        private Builder() {
        }

        /**
         * Sets the standardization method.
         *
         * @param method standardization method
         * @return this builder
         */
        public Builder setMethod(
                StandardizationMethod method
        ) {
            this.method =
                    Objects.requireNonNull(
                            method,
                            "StandardizationMethod cannot be null."
                    );

            return this;
        }

        /**
         * Parses and sets the standardization method.
         *
         * @param method user-facing method name
         * @return this builder
         */
        public Builder setMethod(
                String method
        ) {
            return setMethod(
                    StandardizationMethod.fromString(
                            method
                    )
            );
        }

        /**
         * Sets the fitting scope.
         *
         * @param scope standardization scope
         * @return this builder
         */
        public Builder setScope(
                StandardizationScope scope
        ) {
            this.scope =
                    Objects.requireNonNull(
                            scope,
                            "StandardizationScope cannot be null."
                    );

            return this;
        }

        /**
         * Parses and sets the fitting scope.
         *
         * @param scope user-facing scope name
         * @return this builder
         */
        public Builder setScope(
                String scope
        ) {
            return setScope(
                    StandardizationScope.fromString(
                            scope
                    )
            );
        }

        /**
         * Sets the variance convention.
         *
         * @param varianceConvention population or sample convention
         * @return this builder
         */
        public Builder setVarianceConvention(
                VarianceConvention varianceConvention
        ) {
            this.varianceConvention =
                    Objects.requireNonNull(
                            varianceConvention,
                            "VarianceConvention cannot be null."
                    );

            return this;
        }

        /**
         * Parses and sets the variance convention.
         *
         * @param varianceConvention user-facing convention name
         * @return this builder
         */
        public Builder setVarianceConvention(
                String varianceConvention
        ) {
            return setVarianceConvention(
                    VarianceConvention.fromString(
                            varianceConvention
                    )
            );
        }

        /**
         * Sets an optional externally supplied statistics path.
         *
         * @param statisticsPath statistics path or null
         * @return this builder
         */
        public Builder setStatisticsPath(
                String statisticsPath
        ) {
            this.statisticsPath =
                    statisticsPath;

            return this;
        }

        /**
         * Sets whether newly fitted statistics should later be saved.
         *
         * @param saveFittedStatistics whether to save fitted statistics
         * @return this builder
         */
        public Builder setSaveFittedStatistics(
                boolean saveFittedStatistics
        ) {
            this.saveFittedStatistics =
                    saveFittedStatistics;

            return this;
        }

        /**
         * Sets an optional output path for fitted statistics.
         *
         * @param statisticsOutputPath output path or null
         * @return this builder
         */
        public Builder setStatisticsOutputPath(
                String statisticsOutputPath
        ) {
            this.statisticsOutputPath =
                    statisticsOutputPath;

            return this;
        }

        /**
         * Builds and validates an immutable configuration.
         *
         * This performs structural validation but does not require every
         * recognized method, scope, or JSON operation to be implemented.
         * Call requireImplemented() before execution.
         *
         * @return immutable standardization configuration
         */
        public StandardizationConfig build() {
            return new StandardizationConfig(
                    method,
                    scope,
                    varianceConvention,
                    statisticsPath,
                    saveFittedStatistics,
                    statisticsOutputPath
            );
        }
    }
}