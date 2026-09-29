package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * Immutable policy describing how numeric data should be standardized.
 *
 * <p>GLOBAL and PER_DIMENSION use reusable {@link StandardizationStats}
 * fitted from training data or loaded from JSON. PER_SERIES and
 * PER_SERIES_PER_DIMENSION calculate local parameters from each realized
 * series during transformation and may retain
 * {@link PerSeriesStandardizationState} for later inverse transformation.</p>
 */
public final class StandardizationConfig implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Mathematical transformation applied to accepted numeric values. */
    private final StandardizationMethod method;

    /** Grouping policy used to fit or calculate transformation parameters. */
    private final StandardizationScope scope;

    /** Variance denominator convention used by z-score fitting. */
    private final VarianceConvention varianceConvention;

    /** Optional JSON path containing reusable training statistics. */
    private final String statisticsPath;

    /** Whether newly fitted reusable statistics should be written to JSON. */
    private final boolean saveFittedStatistics;

    /** Optional output path for newly fitted reusable statistics. */
    private final String statisticsOutputPath;

    private StandardizationConfig(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            String statisticsPath,
            boolean saveFittedStatistics,
            String statisticsOutputPath
    ) {
        this.method = Objects.requireNonNull(
                method,
                "StandardizationMethod cannot be null."
        );
        this.scope = Objects.requireNonNull(
                scope,
                "StandardizationScope cannot be null."
        );
        this.varianceConvention = Objects.requireNonNull(
                varianceConvention,
                "VarianceConvention cannot be null."
        );
        this.statisticsPath = normalizeNullablePath(statisticsPath);
        this.saveFittedStatistics = saveFittedStatistics;
        this.statisticsOutputPath =
                normalizeNullablePath(statisticsOutputPath);
        validate();
    }

    /** Returns a configuration that disables standardization. */
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

    /** Returns reusable population z-score standardization per feature or dimension. */
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

    /** Returns reusable global population z-score standardization. */
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
     * Returns the usual univariate time-series z-normalization policy.
     */
    public static StandardizationConfig zScorePerSeries() {
        return new StandardizationConfig(
                StandardizationMethod.Z_SCORE,
                StandardizationScope.PER_SERIES,
                VarianceConvention.POPULATION,
                null,
                false,
                null
        );
    }

    /**
     * Returns the usual per-instance, per-channel multivariate policy.
     */
    public static StandardizationConfig zScorePerSeriesPerDimension() {
        return new StandardizationConfig(
                StandardizationMethod.Z_SCORE,
                StandardizationScope.PER_SERIES_PER_DIMENSION,
                VarianceConvention.POPULATION,
                null,
                false,
                null
        );
    }

    /** Returns a new configuration builder with disabled defaults. */
    public static Builder builder() {
        return new Builder();
    }

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
                .setVarianceConvention(config.getVarianceConvention())
                .setStatisticsPath(config.getStatisticsPath())
                .setSaveFittedStatistics(
                        config.shouldSaveFittedStatistics()
                )
                .setStatisticsOutputPath(
                        config.getStatisticsOutputPath()
                );
    }

    /** @return configured mathematical method */
    public StandardizationMethod getMethod() {
        return method;
    }

    /** @return configured fitting or local-calculation scope */
    public StandardizationScope getScope() {
        return scope;
    }

    /** @return configured variance convention */
    public VarianceConvention getVarianceConvention() {
        return varianceConvention;
    }

    public String getStatisticsPath() {
        return statisticsPath;
    }

    public boolean hasStatisticsPath() {
        return statisticsPath != null;
    }

    public boolean shouldSaveFittedStatistics() {
        return saveFittedStatistics;
    }

    public String getStatisticsOutputPath() {
        return statisticsOutputPath;
    }

    public boolean isEnabled() {
        return method != StandardizationMethod.NONE;
    }

    public boolean isDisabled() {
        return !isEnabled();
    }

    /**
     * Returns whether reusable statistics should be loaded before readers are
     * constructed. Per-series scopes cannot load dataset-level statistics.
     */
    public boolean shouldLoadStatistics() {
        return isEnabled()
                && scope.usesTrainingStatistics()
                && hasStatisticsPath();
    }

    /**
     * Returns whether reusable statistics must be fitted from eager training
     * data.
     */
    public boolean shouldFitStatistics() {
        return isEnabled()
                && scope.usesTrainingStatistics()
                && !hasStatisticsPath()
                && method.requiresFittedStatistics();
    }

    /** @return true when training-set statistics are fitted or loaded and reused */
    public boolean usesReusableTrainingStatistics() {
        return isEnabled() && scope.usesTrainingStatistics();
    }

    /** @return true when parameters are calculated independently per realized series */
    public boolean usesPerSeriesStatistics() {
        return isEnabled() && scope.usesPerSeriesStatistics();
    }

    public void requireImplemented() {
        if (isDisabled()) {
            return;
        }
        method.requireImplemented();
        scope.requireImplemented();
    }

    /**
     * Validates reusable fitted or externally loaded statistics.
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
                    "Cannot validate statistics against disabled "
                            + "standardization."
            );
        }
        if (!usesReusableTrainingStatistics()) {
            throw new IllegalStateException(
                    "Scope "
                            + scope
                            + " calculates local per-series parameters and "
                            + "does not use StandardizationStats."
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
        if (method.usesVarianceConvention()
                && stats.getVarianceConvention() != varianceConvention) {
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
            requireNoPersistenceOptions(
                    "standardization method is NONE"
            );
            return;
        }

        if (scope.usesPerSeriesStatistics()) {
            requireNoPersistenceOptions(
                    "scope " + scope + " uses local per-series parameters"
            );
            return;
        }

        if (!saveFittedStatistics && statisticsOutputPath != null) {
            throw new IllegalArgumentException(
                    "statisticsOutputPath was supplied, but "
                            + "saveFittedStatistics is false."
            );
        }
        if (statisticsPath != null && saveFittedStatistics) {
            throw new IllegalArgumentException(
                    "A configuration cannot both load precomputed "
                            + "statistics and save newly fitted statistics."
            );
        }
    }

    private void requireNoPersistenceOptions(
            String reason
    ) {
        if (statisticsPath != null) {
            throw new IllegalArgumentException(
                    "A reusable statistics path cannot be supplied because "
                            + reason
                            + "."
            );
        }
        if (saveFittedStatistics) {
            throw new IllegalArgumentException(
                    "Reusable fitted statistics cannot be saved because "
                            + reason
                            + "."
            );
        }
        if (statisticsOutputPath != null) {
            throw new IllegalArgumentException(
                    "A statistics output path cannot be supplied because "
                            + reason
                            + "."
            );
        }
    }

    private static String normalizeNullablePath(
            String path
    ) {
        if (path == null) {
            return null;
        }
        String trimmed = path.trim();
        if (trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")) {
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
        return saveFittedStatistics == that.saveFittedStatistics
                && method == that.method
                && scope == that.scope
                && varianceConvention == that.varianceConvention
                && Objects.equals(statisticsPath, that.statisticsPath)
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

    /** Builder for immutable standardization configurations. */
    public static final class Builder {
        private StandardizationMethod method = StandardizationMethod.NONE;
        private StandardizationScope scope =
                StandardizationScope.PER_DIMENSION;
        private VarianceConvention varianceConvention =
                VarianceConvention.POPULATION;
        private String statisticsPath;
        private boolean saveFittedStatistics;
        private String statisticsOutputPath;

        private Builder() {
        }

        public Builder setMethod(
                StandardizationMethod method
        ) {
            this.method = Objects.requireNonNull(
                    method,
                    "StandardizationMethod cannot be null."
            );
            return this;
        }

        public Builder setMethod(
                String method
        ) {
            return setMethod(StandardizationMethod.fromString(method));
        }

        public Builder setScope(
                StandardizationScope scope
        ) {
            this.scope = Objects.requireNonNull(
                    scope,
                    "StandardizationScope cannot be null."
            );
            return this;
        }

        public Builder setScope(
                String scope
        ) {
            return setScope(StandardizationScope.fromString(scope));
        }

        public Builder setVarianceConvention(
                VarianceConvention varianceConvention
        ) {
            this.varianceConvention = Objects.requireNonNull(
                    varianceConvention,
                    "VarianceConvention cannot be null."
            );
            return this;
        }

        public Builder setVarianceConvention(
                String varianceConvention
        ) {
            return setVarianceConvention(
                    VarianceConvention.fromString(varianceConvention)
            );
        }

        public Builder setStatisticsPath(
                String statisticsPath
        ) {
            this.statisticsPath = statisticsPath;
            return this;
        }

        public Builder setSaveFittedStatistics(
                boolean saveFittedStatistics
        ) {
            this.saveFittedStatistics = saveFittedStatistics;
            return this;
        }

        public Builder setStatisticsOutputPath(
                String statisticsOutputPath
        ) {
            this.statisticsOutputPath = statisticsOutputPath;
            return this;
        }

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
