package output;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Stable, JSON-friendly representation of the results from one PFGAP
 * repetition.
 *
 * This DTO deliberately does not contain:
 *
 *     ProximityForest
 *     ProximityTree
 *     ListObjectDataset
 *     prediction arrays
 *     outlier-score arrays
 *     proximity matrices
 *
 * Those objects are either too large, implementation-specific, or written
 * as separate output artifacts.
 *
 * The result record instead contains compact experiment metadata:
 *
 *     dataset identity
 *     repetition number
 *     forest identity and mode
 *     evaluation metrics
 *     timing information
 *     aggregate forest-structure statistics
 *     preprocessing metadata
 *     paths to separately written artifacts
 *
 * Maps use LinkedHashMap internally so Gson produces deterministic and
 * readable field ordering.
 *
 * This class is immutable after construction. Nested map values are copied
 * defensively and exposed through unmodifiable views.
 */
public final class ExperimentResultRecord {

    /**
     * Version of the JSON-facing result-record schema.
     *
     * This is independent of Java serialization because this DTO is intended
     * primarily for JSON output.
     */
    public static final int CURRENT_FORMAT_VERSION =
            1;

    private final int formatVersion;

    private final String dataset;

    /**
     * One-based repetition number for user-facing output.
     */
    private final int repetition;

    private final int forestId;

    /**
     * Classification, regression, or isolation.
     */
    private final String forestMode;

    /**
     * Named evaluation metrics.
     *
     * Examples:
     *
     *     accuracy
     *     errorRate
     *     rmse
     *     mae
     *     r2
     *
     * Isolation runs may have an empty metrics map when scores are written
     * as a separate per-instance artifact.
     */
    private final Map<String, Double> metrics;

    /**
     * Named integer counts.
     *
     * Examples:
     *
     *     correct
     *     errors
     *     predictionCount
     *     trainingInstanceCount
     *     testingInstanceCount
     */
    private final Map<String, Long> counts;

    /**
     * Named timing values measured in milliseconds.
     *
     * Examples:
     *
     *     trainingMilliseconds
     *     testingMilliseconds
     *     proximityMilliseconds
     */
    private final Map<String, Double> timingMilliseconds;

    /**
     * Compact forest statistics.
     *
     * Examples:
     *
     *     numTrees
     *     meanNodesPerTree
     *     standardDeviationNodesPerTree
     *     meanDepthPerTree
     *     standardDeviationDepthPerTree
     *     meanWeightedDepthPerTree
     *     standardDeviationWeightedDepthPerTree
     */
    private final Map<String, Double> forestStatistics;

    /**
     * Model and preprocessing configuration relevant to interpreting the
     * result.
     *
     * Examples:
     *
     *     standardizationMethod
     *     standardizationScope
     *     varianceConvention
     *     proximityType
     *     trainingReaderType
     *     testingReaderType
     *
     * Values are strings so enum values and concise descriptive metadata can
     * be represented without coupling this DTO to application classes.
     */
    private final Map<String, String> configuration;

    /**
     * Paths to separately written output artifacts.
     *
     * Examples:
     *
     *     predictions
     *     outlierScores
     *     trainingProximities
     *     testTrainProximities
     *     model
     *     standardizationStatistics
     *
     * Paths should preferably be relative to the experiment output
     * directory when possible.
     */
    private final Map<String, String> artifacts;

    private ExperimentResultRecord(
            Builder builder
    ) {
        this.formatVersion =
                CURRENT_FORMAT_VERSION;

        this.dataset =
                requireNonblank(
                        builder.dataset,
                        "dataset"
                );

        if (builder.repetition <= 0) {
            throw new IllegalArgumentException(
                    "Repetition must be one-based and positive, "
                            + "but received: "
                            + builder.repetition
            );
        }

        this.repetition =
                builder.repetition;

        this.forestId =
                builder.forestId;

        this.forestMode =
                requireNonblank(
                        builder.forestMode,
                        "forestMode"
                );

        this.metrics =
                immutableNumericMap(
                        builder.metrics,
                        "metrics",
                        true
                );

        this.counts =
                immutableCountMap(
                        builder.counts,
                        "counts"
                );

        this.timingMilliseconds =
                immutableNumericMap(
                        builder.timingMilliseconds,
                        "timingMilliseconds",
                        false
                );

        this.forestStatistics =
                immutableNumericMap(
                        builder.forestStatistics,
                        "forestStatistics",
                        true
                );

        this.configuration =
                immutableStringMap(
                        builder.configuration,
                        "configuration"
                );

        this.artifacts =
                immutableStringMap(
                        builder.artifacts,
                        "artifacts"
                );
    }

    /**
     * Returns a new result-record builder.
     *
     * @return builder
     */
    public static Builder builder() {
        return new Builder();
    }

    public int getFormatVersion() {
        return formatVersion;
    }

    public String getDataset() {
        return dataset;
    }

    public int getRepetition() {
        return repetition;
    }

    public int getForestId() {
        return forestId;
    }

    public String getForestMode() {
        return forestMode;
    }

    public Map<String, Double> getMetrics() {
        return metrics;
    }

    public Map<String, Long> getCounts() {
        return counts;
    }

    public Map<String, Double> getTimingMilliseconds() {
        return timingMilliseconds;
    }

    public Map<String, Double> getForestStatistics() {
        return forestStatistics;
    }

    public Map<String, String> getConfiguration() {
        return configuration;
    }

    public Map<String, String> getArtifacts() {
        return artifacts;
    }

    /**
     * Returns a named metric or null when it is not present.
     *
     * @param metricName metric key
     * @return metric value or null
     */
    public Double getMetric(
            String metricName
    ) {
        return metrics.get(metricName);
    }

    /**
     * Returns whether this record contains a named metric.
     *
     * @param metricName metric key
     * @return true when present
     */
    public boolean hasMetric(
            String metricName
    ) {
        return metrics.containsKey(
                metricName
        );
    }

    /**
     * Returns a named artifact path or null when it is not present.
     *
     * @param artifactName artifact key
     * @return artifact path or null
     */
    public String getArtifact(
            String artifactName
    ) {
        return artifacts.get(
                artifactName
        );
    }

    private static Map<String, Double> immutableNumericMap(
            Map<String, Double> source,
            String mapName,
            boolean allowNonfinite
    ) {
        LinkedHashMap<String, Double> copied =
                new LinkedHashMap<>();

        if (source == null) {
            return Collections.unmodifiableMap(
                    copied
            );
        }

        for (Map.Entry<String, Double>
                entry : source.entrySet()) {

            String key =
                    requireNonblank(
                            entry.getKey(),
                            mapName + " key"
                    );

            Double value =
                    entry.getValue();

            if (value == null) {
                throw new IllegalArgumentException(
                        mapName
                                + " value for key '"
                                + key
                                + "' cannot be null."
                );
            }

            if (!allowNonfinite
                    && !Double.isFinite(value)) {

                throw new IllegalArgumentException(
                        mapName
                                + " value for key '"
                                + key
                                + "' must be finite, but received "
                                + value
                                + "."
                );
            }

            copied.put(
                    key,
                    value
            );
        }

        return Collections.unmodifiableMap(
                copied
        );
    }

    private static Map<String, Long> immutableCountMap(
            Map<String, Long> source,
            String mapName
    ) {
        LinkedHashMap<String, Long> copied =
                new LinkedHashMap<>();

        if (source == null) {
            return Collections.unmodifiableMap(
                    copied
            );
        }

        for (Map.Entry<String, Long>
                entry : source.entrySet()) {

            String key =
                    requireNonblank(
                            entry.getKey(),
                            mapName + " key"
                    );

            Long value =
                    entry.getValue();

            if (value == null) {
                throw new IllegalArgumentException(
                        mapName
                                + " value for key '"
                                + key
                                + "' cannot be null."
                );
            }

            if (value < 0L) {
                throw new IllegalArgumentException(
                        mapName
                                + " value for key '"
                                + key
                                + "' cannot be negative, but received "
                                + value
                                + "."
                );
            }

            copied.put(
                    key,
                    value
            );
        }

        return Collections.unmodifiableMap(
                copied
        );
    }

    private static Map<String, String> immutableStringMap(
            Map<String, String> source,
            String mapName
    ) {
        LinkedHashMap<String, String> copied =
                new LinkedHashMap<>();

        if (source == null) {
            return Collections.unmodifiableMap(
                    copied
            );
        }

        for (Map.Entry<String, String>
                entry : source.entrySet()) {

            String key =
                    requireNonblank(
                            entry.getKey(),
                            mapName + " key"
                    );

            String value =
                    entry.getValue();

            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(
                        mapName
                                + " value for key '"
                                + key
                                + "' cannot be null or blank."
                );
            }

            copied.put(
                    key,
                    value.trim()
            );
        }

        return Collections.unmodifiableMap(
                copied
        );
    }

    private static String requireNonblank(
            String value,
            String description
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    description
                            + " cannot be null or blank."
            );
        }

        return value.trim();
    }

    @Override
    public boolean equals(
            Object other
    ) {
        if (this == other) {
            return true;
        }

        if (!(other instanceof ExperimentResultRecord that)) {
            return false;
        }

        return formatVersion == that.formatVersion
                && repetition == that.repetition
                && forestId == that.forestId
                && dataset.equals(that.dataset)
                && forestMode.equals(that.forestMode)
                && metrics.equals(that.metrics)
                && counts.equals(that.counts)
                && timingMilliseconds.equals(
                that.timingMilliseconds
        )
                && forestStatistics.equals(
                that.forestStatistics
        )
                && configuration.equals(
                that.configuration
        )
                && artifacts.equals(
                that.artifacts
        );
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                formatVersion,
                dataset,
                repetition,
                forestId,
                forestMode,
                metrics,
                counts,
                timingMilliseconds,
                forestStatistics,
                configuration,
                artifacts
        );
    }

    @Override
    public String toString() {
        return "ExperimentResultRecord{"
                + "formatVersion=" + formatVersion
                + ", dataset='" + dataset + '\''
                + ", repetition=" + repetition
                + ", forestId=" + forestId
                + ", forestMode='" + forestMode + '\''
                + ", metrics=" + metrics
                + ", counts=" + counts
                + ", timingMilliseconds=" + timingMilliseconds
                + ", forestStatistics=" + forestStatistics
                + ", configuration=" + configuration
                + ", artifacts=" + artifacts
                + '}';
    }

    /**
     * Builder for one immutable result record.
     */
    public static final class Builder {

        private String dataset;

        private int repetition =
                1;

        private int forestId =
                -1;

        private String forestMode =
                "classification";

        private final Map<String, Double> metrics =
                new LinkedHashMap<>();

        private final Map<String, Long> counts =
                new LinkedHashMap<>();

        private final Map<String, Double> timingMilliseconds =
                new LinkedHashMap<>();

        private final Map<String, Double> forestStatistics =
                new LinkedHashMap<>();

        private final Map<String, String> configuration =
                new LinkedHashMap<>();

        private final Map<String, String> artifacts =
                new LinkedHashMap<>();

        private Builder() {
        }

        public Builder setDataset(
                String dataset
        ) {
            this.dataset =
                    dataset;

            return this;
        }

        /**
         * Sets the one-based repetition number.
         */
        public Builder setRepetition(
                int repetition
        ) {
            this.repetition =
                    repetition;

            return this;
        }

        public Builder setForestId(
                int forestId
        ) {
            this.forestId =
                    forestId;

            return this;
        }

        public Builder setForestMode(
                String forestMode
        ) {
            this.forestMode =
                    forestMode;

            return this;
        }

        public Builder addMetric(
                String name,
                double value
        ) {
            metrics.put(
                    requireNonblank(
                            name,
                            "metric name"
                    ),
                    value
            );

            return this;
        }

        public Builder addMetrics(
                Map<String, Double> values
        ) {
            if (values != null) {
                metrics.putAll(values);
            }

            return this;
        }

        public Builder addCount(
                String name,
                long value
        ) {
            if (value < 0L) {
                throw new IllegalArgumentException(
                        "Count '"
                                + name
                                + "' cannot be negative: "
                                + value
                );
            }

            counts.put(
                    requireNonblank(
                            name,
                            "count name"
                    ),
                    value
            );

            return this;
        }

        public Builder addCounts(
                Map<String, Long> values
        ) {
            if (values != null) {
                counts.putAll(values);
            }

            return this;
        }

        public Builder addTimingMilliseconds(
                String name,
                double value
        ) {
            if (!Double.isFinite(value) || value < 0.0) {
                throw new IllegalArgumentException(
                        "Timing value '"
                                + name
                                + "' must be finite and nonnegative, "
                                + "but received "
                                + value
                                + "."
                );
            }

            timingMilliseconds.put(
                    requireNonblank(
                            name,
                            "timing name"
                    ),
                    value
            );

            return this;
        }

        public Builder addTimingNanoseconds(
                String name,
                long nanoseconds
        ) {
            if (nanoseconds < 0L) {
                throw new IllegalArgumentException(
                        "Timing value '"
                                + name
                                + "' cannot be negative: "
                                + nanoseconds
                );
            }

            return addTimingMilliseconds(
                    name,
                    nanoseconds / 1_000_000.0
            );
        }

        public Builder addForestStatistic(
                String name,
                double value
        ) {
            forestStatistics.put(
                    requireNonblank(
                            name,
                            "forest statistic name"
                    ),
                    value
            );

            return this;
        }

        public Builder addForestStatistics(
                Map<String, Double> values
        ) {
            if (values != null) {
                forestStatistics.putAll(values);
            }

            return this;
        }

        public Builder addConfiguration(
                String name,
                Object value
        ) {
            if (value == null) {
                return this;
            }

            configuration.put(
                    requireNonblank(
                            name,
                            "configuration name"
                    ),
                    String.valueOf(value)
            );

            return this;
        }

        public Builder addConfigurationValues(
                Map<String, String> values
        ) {
            if (values != null) {
                configuration.putAll(values);
            }

            return this;
        }

        public Builder addArtifact(
                String name,
                Path path
        ) {
            if (path == null) {
                return this;
            }

            return addArtifact(
                    name,
                    path.toString()
            );
        }

        public Builder addArtifact(
                String name,
                String path
        ) {
            if (path == null || path.isBlank()) {
                return this;
            }

            artifacts.put(
                    requireNonblank(
                            name,
                            "artifact name"
                    ),
                    path.trim()
            );

            return this;
        }

        public Builder addArtifacts(
                Map<String, String> values
        ) {
            if (values != null) {
                artifacts.putAll(values);
            }

            return this;
        }

        public ExperimentResultRecord build() {
            return new ExperimentResultRecord(
                    this
            );
        }
    }
}