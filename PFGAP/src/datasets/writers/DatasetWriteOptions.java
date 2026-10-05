package datasets.writers;

import preprocessing.standardization.PerSeriesStandardizationState;
import preprocessing.standardization.StandardizationStats;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Immutable options shared by dataset writer implementations.
 *
 * <p>This class carries output metadata and optional inverse-standardization
 * parameters. It deliberately does not identify a concrete file format. The
 * selected {@link DatasetWriter} owns format-specific encoding rules, while a
 * writer factory maps the original reader type to its mirror writer.</p>
 *
 * <p>Exactly one inverse-parameter source may be present:</p>
 *
 * <ul>
 *     <li>{@link StandardizationStats} for reusable GLOBAL or PER_DIMENSION
 *     transformation parameters.</li>
 *     <li>A dataset-aligned list of
 *     {@link PerSeriesStandardizationState} objects for PER_SERIES or
 *     PER_SERIES_PER_DIMENSION transformation parameters.</li>
 *     <li>Neither when standardization is disabled or the source values are
 *     already in output coordinates.</li>
 * </ul>
 *
 * <p>Writers should use these parameters to inverse-transform values while
 * emitting output. They must not mutate the source dataset or construct a
 * complete inverse-transformed numeric copy.</p>
 */
public final class DatasetWriteOptions {

    /**
     * Describes the logical interpretation of materialized dataset instances.
     *
     * <p>The same primitive representation can have different meanings. A
     * {@code double[]} may be one fixed-width tabular row or one potentially
     * variable-length univariate time series. Writers use this value for schema
     * validation and layout selection rather than inferring semantics solely
     * from the Java array type.</p>
     */
    public enum DataLayout {
        /** One instance is a fixed-schema feature row. */
        TABULAR,

        /** One instance is one univariate series, potentially variable length. */
        UNIVARIATE_SERIES,

        /** One instance is a dimension-major multivariate series. */
        MULTIVARIATE_SERIES,

        /** One observation per record with explicit instance/time metadata. */
        LONG_FORM,

        /** Layout is unknown and must be inferred or rejected by the writer. */
        AUTO
    }

    /** Placement of an embedded target column in a delimited record. */
    public enum TargetPlacement {
        FIRST,
        LAST
    }

    private final Path outputPath;
    private final DataLayout dataLayout;
    private final String entrySeparator;
    private final String arraySeparator;
    private final boolean includeHeader;
    private final List<String> featureNames;
    private final boolean embedLabels;
    private final List<?> labels;
    private final String targetName;
    private final TargetPlacement targetPlacement;
    private final StandardizationStats reusableStatistics;
    private final List<PerSeriesStandardizationState> perSeriesStates;

    private DatasetWriteOptions(
            Builder builder
    ) {
        this.outputPath = Objects.requireNonNull(
                builder.outputPath,
                "Dataset output path cannot be null."
        );
        this.dataLayout = Objects.requireNonNull(
                builder.dataLayout,
                "Dataset data layout cannot be null."
        );
        this.entrySeparator = normalizeRequiredSeparator(
                builder.entrySeparator,
                "entrySeparator"
        );
        this.arraySeparator = normalizeOptionalSeparator(
                builder.arraySeparator
        );
        this.includeHeader = builder.includeHeader;
        this.featureNames = immutableStrings(
                builder.featureNames,
                "featureNames"
        );
        this.embedLabels = builder.embedLabels;
        this.labels = builder.labels == null
                ? List.of()
                : List.copyOf(builder.labels);
        this.targetName = normalizeTargetName(builder.targetName);
        this.targetPlacement = Objects.requireNonNull(
                builder.targetPlacement,
                "Target placement cannot be null."
        );
        this.reusableStatistics = builder.reusableStatistics;
        this.perSeriesStates = builder.perSeriesStates == null
                ? List.of()
                : List.copyOf(builder.perSeriesStates);

        validate();
    }

    public static Builder builder(
            Path outputPath
    ) {
        return new Builder(outputPath);
    }

    public static Builder builder(
            DatasetWriteOptions options
    ) {
        Objects.requireNonNull(
                options,
                "DatasetWriteOptions cannot be null."
        );
        return new Builder(options.getOutputPath())
                .setDataLayout(options.getDataLayout())
                .setEntrySeparator(options.getEntrySeparator())
                .setArraySeparator(options.getArraySeparator())
                .setIncludeHeader(options.shouldIncludeHeader())
                .setFeatureNames(options.getFeatureNames())
                .setEmbedLabels(options.shouldEmbedLabels())
                .setLabels(options.getLabels())
                .setTargetName(options.getTargetName())
                .setTargetPlacement(options.getTargetPlacement())
                .setReusableStatistics(options.getReusableStatistics())
                .setPerSeriesStates(options.getPerSeriesStates());
    }

    public Path getOutputPath() {
        return outputPath;
    }

    public DataLayout getDataLayout() {
        return dataLayout;
    }

    public String getEntrySeparator() {
        return entrySeparator;
    }

    public String getArraySeparator() {
        return arraySeparator;
    }

    public boolean hasArraySeparator() {
        return arraySeparator != null;
    }

    public boolean shouldIncludeHeader() {
        return includeHeader;
    }

    public List<String> getFeatureNames() {
        return featureNames;
    }

    public boolean hasFeatureNames() {
        return !featureNames.isEmpty();
    }

    public boolean shouldEmbedLabels() {
        return embedLabels;
    }

    public List<?> getLabels() {
        return labels;
    }

    public boolean hasLabels() {
        return !labels.isEmpty();
    }

    public String getTargetName() {
        return targetName;
    }

    public TargetPlacement getTargetPlacement() {
        return targetPlacement;
    }

    public boolean isTargetFirst() {
        return targetPlacement == TargetPlacement.FIRST;
    }

    public StandardizationStats getReusableStatistics() {
        return reusableStatistics;
    }

    public boolean hasReusableStatistics() {
        return reusableStatistics != null;
    }

    public List<PerSeriesStandardizationState> getPerSeriesStates() {
        return perSeriesStates;
    }

    public boolean hasPerSeriesStates() {
        return !perSeriesStates.isEmpty();
    }

    public boolean requiresInverseTransformation() {
        return hasReusableStatistics() || hasPerSeriesStates();
    }

    /**
     * Returns the single-character entry separator required by FastCSV-backed
     * writers.
     *
     * @throws IllegalArgumentException if the configured separator expands to
     *         more than one character or is a line delimiter
     */
    public char getEntrySeparatorCharacter() {
        return parseSingleCharacterSeparator(
                entrySeparator,
                "entrySeparator"
        );
    }

    /**
     * Returns the single-character array separator when configured.
     *
     * @throws IllegalStateException if no array separator is configured
     * @throws IllegalArgumentException if it is not one character
     */
    public char getArraySeparatorCharacter() {
        if (arraySeparator == null) {
            throw new IllegalStateException(
                    "No array separator is configured."
            );
        }
        return parseSingleCharacterSeparator(
                arraySeparator,
                "arraySeparator"
        );
    }

    private void validate() {
        if (hasReusableStatistics() && hasPerSeriesStates()) {
            throw new IllegalArgumentException(
                    "Dataset output cannot use reusable standardization "
                            + "statistics and per-series states at the same "
                            + "time."
            );
        }
        if (includeHeader && featureNames.isEmpty()) {
            throw new IllegalArgumentException(
                    "Feature names are required when includeHeader is true."
            );
        }
        if (!embedLabels && !labels.isEmpty()) {
            throw new IllegalArgumentException(
                    "Labels were supplied, but embedLabels is false. Use a "
                            + "separate label writer or enable embedded labels."
            );
        }
        if (embedLabels && labels.isEmpty()) {
            throw new IllegalArgumentException(
                    "Embedded-label output requires a nonempty label list."
            );
        }
        if (dataLayout == DataLayout.MULTIVARIATE_SERIES
                && arraySeparator == null) {
            throw new IllegalArgumentException(
                    "MULTIVARIATE_SERIES output requires an array separator "
                            + "unless a concrete writer uses a format with "
                            + "native nested structure."
            );
        }
        if (arraySeparator != null
                && arraySeparator.equals(entrySeparator)) {
            throw new IllegalArgumentException(
                    "Entry and array separators must differ."
            );
        }
    }

    private static String normalizeRequiredSeparator(
            String separator,
            String name
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    name + " cannot be null or empty."
            );
        }
        return separator;
    }

    private static String normalizeOptionalSeparator(
            String separator
    ) {
        if (separator == null || separator.isEmpty()) {
            return null;
        }
        return separator;
    }

    private static char parseSingleCharacterSeparator(
            String separator,
            String name
    ) {
        String normalized = switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    name
                            + " must contain exactly one character for this "
                            + "writer, but received: '"
                            + separator
                            + "'."
            );
        }
        char value = normalized.charAt(0);
        if (value == '\n' || value == '\r') {
            throw new IllegalArgumentException(
                    name + " cannot be a line delimiter."
            );
        }
        return value;
    }

    private static List<String> immutableStrings(
            List<String> values,
            String fieldName
    ) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        for (int index = 0; index < values.size(); index++) {
            String value = values.get(index);
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(
                        fieldName
                                + " contains a null or blank value at index "
                                + index
                                + "."
                );
            }
        }
        return List.copyOf(values);
    }

    private static String normalizeTargetName(
            String targetName
    ) {
        if (targetName == null || targetName.isBlank()) {
            return "target";
        }
        return targetName.trim();
    }

    @Override
    public String toString() {
        return "DatasetWriteOptions{"
                + "outputPath=" + outputPath
                + ", dataLayout=" + dataLayout
                + ", entrySeparator='" + entrySeparator + '\''
                + ", arraySeparator='" + arraySeparator + '\''
                + ", includeHeader=" + includeHeader
                + ", featureNames=" + featureNames
                + ", embedLabels=" + embedLabels
                + ", labelCount=" + labels.size()
                + ", targetName='" + targetName + '\''
                + ", targetPlacement=" + targetPlacement
                + ", reusableStatistics="
                + (reusableStatistics == null ? "absent" : "present")
                + ", perSeriesStateCount=" + perSeriesStates.size()
                + '}';
    }

    /** Builder for immutable dataset-writing options. */
    public static final class Builder {
        private final Path outputPath;
        private DataLayout dataLayout = DataLayout.AUTO;
        private String entrySeparator = ",";
        private String arraySeparator;
        private boolean includeHeader;
        private List<String> featureNames = List.of();
        private boolean embedLabels;
        private List<?> labels = List.of();
        private String targetName = "target";
        private TargetPlacement targetPlacement = TargetPlacement.LAST;
        private StandardizationStats reusableStatistics;
        private List<PerSeriesStandardizationState> perSeriesStates = List.of();

        private Builder(
                Path outputPath
        ) {
            this.outputPath = Objects.requireNonNull(
                    outputPath,
                    "Dataset output path cannot be null."
            );
        }

        public Builder setDataLayout(
                DataLayout dataLayout
        ) {
            this.dataLayout = Objects.requireNonNull(
                    dataLayout,
                    "Dataset data layout cannot be null."
            );
            return this;
        }

        public Builder setEntrySeparator(
                String entrySeparator
        ) {
            this.entrySeparator = entrySeparator;
            return this;
        }

        public Builder setArraySeparator(
                String arraySeparator
        ) {
            this.arraySeparator = arraySeparator;
            return this;
        }

        public Builder setIncludeHeader(
                boolean includeHeader
        ) {
            this.includeHeader = includeHeader;
            return this;
        }

        public Builder setFeatureNames(
                List<String> featureNames
        ) {
            this.featureNames = featureNames;
            return this;
        }

        public Builder setEmbedLabels(
                boolean embedLabels
        ) {
            this.embedLabels = embedLabels;
            return this;
        }

        public Builder setLabels(
                List<?> labels
        ) {
            this.labels = labels;
            return this;
        }

        public Builder setTargetName(
                String targetName
        ) {
            this.targetName = targetName;
            return this;
        }

        public Builder setTargetPlacement(
                TargetPlacement targetPlacement
        ) {
            this.targetPlacement = targetPlacement;
            return this;
        }

        public Builder setReusableStatistics(
                StandardizationStats reusableStatistics
        ) {
            this.reusableStatistics = reusableStatistics;
            return this;
        }

        public Builder setPerSeriesStates(
                List<PerSeriesStandardizationState> perSeriesStates
        ) {
            this.perSeriesStates = perSeriesStates;
            return this;
        }

        public DatasetWriteOptions build() {
            return new DatasetWriteOptions(this);
        }
    }
}
